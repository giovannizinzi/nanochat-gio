# FlashMoE Triton kernel — dev log

Goal: a fused Triton kernel for the MoE expert FFN (`bmm + relu² + bmm`) fast enough to close the ~450 ms/step overhead vs dense d26 FP8 described in `dev/MOE_WHY_DENSE_WINS.md`. If we can get expert FFN runtime below `torch.bmm`, MoE could plausibly beat dense d26 on time-to-CORE-0.2565.

Reference point on H100 (bf16, E=4, N=32768, D=1408, H=4096):
- `torch.bmm + relu² + torch.bmm`: **1.46 ms** (≈ 258 TFLOPs, ~80% of H100 peak)
- That's the number to beat.

## v1 — initial kernel (commit `f38837a`, superseded)

First attempt. Used `tl.load` over the full (BLOCK_M, D) row at once. Flagged as SRAM-risky — D=1408 × BLOCK_M=64 × 2 bytes = 176 KB, close to the H100 SRAM budget with no room for accumulators. Never benchmarked.

## v2 — proper 3-level tiling (commit `fd423dd`)

Rewrote with inner `BLOCK_D` tiling so x_chunk is always (BLOCK_M, BLOCK_D) — small and safe.

Config:
```
BLOCK_M = 64, BLOCK_D = 128, BLOCK_H = 128
grid = (cdiv(N_e, BLOCK_M), cdiv(D, BLOCK_D))
```

Per (pid_m, pid_d) block: for each of H/BLOCK_H=32 h-slabs, inner-tile over D/BLOCK_D=11 chunks. Accumulate (BLOCK_M, BLOCK_D) out tile over H slabs.

**Result — v56 bench (2026-04-18)** on 1×H100:
```
E=4, N=32768, D=1408, H=4096, bf16
  max abs diff:      3.1250e-02   (within bf16 noise at H=4096)
  mean abs diff:     4.0208e-03
  max rel diff (>1e-2 ref): 1.92
  PASS (correctness)
  reference bmm:   1.46 ms
  flash kernel:   18.34 ms
  speedup:         0.08x   ← 12.5× SLOWER
```

**Analysis**:
- Achieved ~20 TFLOPs vs cuBLAS 258 TFLOPs = 8% of peak.
- Kernel is correct but massively under-utilizes Tensor Cores.
- Likely causes: tile sizes not optimal for TC (BLOCK_M=64 is small), no `num_warps`/`num_stages` pipelining pragmas, each (pid_m, pid_d) block re-loads the same w_fc chunks H/BLOCK_H=32× without L2 cache hints.
- HBM traffic analysis: for each output tile we re-read x_chunk 32× (once per h-slab). That's 32× redundant reads from HBM of the same data. L2 likely absorbs most of this, but the inner loop structure is wasteful.

## Lessons

1. **Beating cuBLAS on bf16 matmul is hard.** torch.bmm is at 80% of H100 peak; Triton needs careful tile tuning + pipelining to match it, let alone beat it. A naive tiled kernel will be 5-10× slower.

2. **Fusion value is small when bmm is near-peak.** The theoretical HBM-saving win from fusing `bmm1 + relu² + bmm2` is:
   - Hidden buffer (E, cap, H) = 4 * 8192 * 4096 * 2 bytes = 256 MB
   - Round-trip (BMM1 write + RELU read/write + BMM2 read) = ~3 × 256 MB = 768 MB
   - At 3 TB/s HBM = 256 μs = 0.25 ms savings best-case
   - That's 17% of the 1.46 ms reference time.
   
   So even a *perfect* fused kernel only saves ~0.25 ms per layer per step. Across 22 layers × 2 (forward + backward) = 11 ms/step. Against the ~450 ms/step MoE overhead, that's 2.5% — not a step-change.

3. **The kernel is the wrong lever for closing the MoE-vs-dense gap.** The 450 ms/step MoE overhead breaks down as:
   - Expert bmm forward: 130 ms — cuBLAS is already near-peak here
   - Expert bmm backward: 200 ms — ditto (plus lacks FP8)
   - Dispatch scatter/gather: 140 ms — this IS a memory-bound kernel, Triton CAN help
   - Aux loss + router: 10 ms — too small to matter
   - Python overhead + NCCL: 140 ms — not addressable via fusion
   
   The actionable kernel target is **dispatch+combine fusion**, not FFN fusion. That's a different kernel (scatter-gather, not matmul) and much more likely to beat a Python-managed bmm chain.

## Status

v2 kernel is **correct but too slow to use**. Leaving `nanochat/flash_moe.py` + `scripts/bench_flash_moe.py` in tree as a starting point for someone who wants to pursue Triton matmul tuning, but **not integrating into moe.py**. Per nanochat's "keep it simple, no bloat" principle, we don't ship a slower code path behind a flag.

## Future directions (not attempted here)

1. **Triton autotune**: let Triton explore `num_warps ∈ {4, 8}`, `num_stages ∈ {2, 3, 4}`, `BLOCK_M ∈ {64, 128}`, `BLOCK_H ∈ {64, 128}`. Standard `triton.autotune` decorator. Might get us to 2-3× slower than cuBLAS, still not helpful.
2. **Dispatch-only fusion**: write a Triton kernel that does `argsort → scatter → per-expert slice` in one pass. Attack the 140 ms dispatch/combine overhead directly without competing with cuBLAS on matmul.
3. **Grouped launch**: collapse the per-expert Python loop into one kernel launch, where grid.z = expert_id and per-expert token ranges are loaded from a device-side offset table. Standard ScatterMoE approach — eliminates launch overhead from 4 → 1 kernel call.
4. **FP8 grouped backward**: the MoE gap doesn't come from bmm forward (already FP8 via torch._scaled_grouped_mm); it comes from bmm backward being bf16. Custom autograd that does both directions in FP8 would cut ~100 ms/step. This is a PyTorch-level change, not a Triton kernel.
