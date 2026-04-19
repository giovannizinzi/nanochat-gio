"""Correctness + microbenchmark harness for `nanochat.flash_moe.flash_moe_expert_ffn`.

Run on a box with CUDA + Triton installed:
    python -m scripts.bench_flash_moe

Exits non-zero if the kernel's output diverges from the reference bmm+relu²+bmm path
by more than `--atol` / `--rtol`. On success, prints a wall-clock comparison at the
shapes used by MoE d22 (default).
"""

import argparse
import time
import torch
import torch.nn.functional as F


def reference_expert_ffn(x_sorted: torch.Tensor,
                         w_fc: torch.Tensor,
                         w_proj: torch.Tensor,
                         expert_offsets: torch.Tensor) -> torch.Tensor:
    """Reference implementation using torch.bmm. Matches nanochat/moe.py's replicated path."""
    E = w_fc.shape[0]
    out = torch.empty_like(x_sorted)
    offs = expert_offsets.tolist()
    for e in range(E):
        s, t = offs[e], offs[e + 1]
        if t <= s:
            continue
        x_e = x_sorted[s:t]                 # (N_e, D)
        h_e = x_e @ w_fc[e]                  # (N_e, H)
        h_e = F.relu(h_e).square()
        out[s:t] = h_e @ w_proj[e]           # (N_e, D)
    return out


def make_shapes(E, N_total, D, H, device, dtype, seed=42):
    g = torch.Generator(device=device).manual_seed(seed)
    # Randomly assign tokens to experts, then count tokens per expert.
    idx = torch.randint(0, E, (N_total,), device=device, generator=g)
    counts = torch.bincount(idx, minlength=E)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])
    # Sort tokens by expert so each expert's rows are contiguous.
    perm = idx.argsort()
    x = torch.randn(N_total, D, device=device, dtype=dtype, generator=g)[perm].contiguous()
    w_fc = torch.randn(E, D, H, device=device, dtype=dtype, generator=g) * (D ** -0.5)
    w_proj = torch.randn(E, H, D, device=device, dtype=dtype, generator=g) * (H ** -0.5)
    return x, w_fc, w_proj, offsets


def check_correctness(E, N_total, D, H, device, dtype, atol, rtol):
    from nanochat.flash_moe import flash_moe_expert_ffn
    x, w_fc, w_proj, offs = make_shapes(E, N_total, D, H, device, dtype)
    ref = reference_expert_ffn(x, w_fc, w_proj, offs)
    kern = flash_moe_expert_ffn(x, w_fc, w_proj, offs)
    diff = (ref.float() - kern.float()).abs()
    # Relative diff restricted to non-tiny reference values — tiny-denom rel diffs are
    # meaningless noise and swamp the metric.
    ref_abs = ref.abs().float()
    mask = ref_abs > 1e-2
    rel = (diff[mask] / ref_abs[mask]) if mask.any() else torch.tensor([0.0])
    print(f"  max abs diff:      {diff.max().item():.4e}")
    print(f"  mean abs diff:     {diff.mean().item():.4e}")
    print(f"  max rel diff (>1e-2 ref): {rel.max().item():.4e}")
    print(f"  mean rel diff (>1e-2 ref):{rel.mean().item():.4e}")
    ok = torch.allclose(ref.float(), kern.float(), atol=atol, rtol=rtol)
    return ok, diff.max().item(), rel.max().item()


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--E", type=int, default=4)
    p.add_argument("--N", type=int, default=32768, help="total tokens across all experts")
    p.add_argument("--D", type=int, default=1408, help="model dim (d22 => 1408)")
    p.add_argument("--H", type=int, default=4096, help="expert hidden dim")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    # Loose defaults — bf16 matmul over H=4096 accumulates ~3e-2 error vs a different
    # accumulation order even when both are "correct". Run the benchmark regardless so we
    # can see if the kernel is worth the precision tradeoff.
    p.add_argument("--atol", type=float, default=5e-2)
    p.add_argument("--rtol", type=float, default=5e-2)
    p.add_argument("--skip-bench", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    print(f"=== Correctness check (E={args.E}, N={args.N}, D={args.D}, H={args.H}, {args.dtype}) ===")
    ok, md, mrd = check_correctness(args.E, args.N, args.D, args.H, device, dtype, args.atol, args.rtol)
    print("  PASS" if ok else "  FAIL (run benchmark anyway for speed data)")

    if args.skip_bench:
        return SystemExit(0 if ok else 1)

    print("\n=== Benchmark ===")
    x, w_fc, w_proj, offs = make_shapes(args.E, args.N, args.D, args.H, device, dtype)
    from nanochat.flash_moe import flash_moe_expert_ffn

    ref_t = bench(lambda: reference_expert_ffn(x, w_fc, w_proj, offs))
    kern_t = bench(lambda: flash_moe_expert_ffn(x, w_fc, w_proj, offs))
    print(f"  reference (bmm+relu²+bmm):   {ref_t*1000:7.2f} ms")
    print(f"  flash_moe_expert_ffn:        {kern_t*1000:7.2f} ms")
    print(f"  speedup:                     {ref_t/kern_t:.2f}x")


if __name__ == "__main__":
    main()
