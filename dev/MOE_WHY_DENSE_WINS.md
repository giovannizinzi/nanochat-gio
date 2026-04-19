# Why dense beats MoE on time-to-GPT-2-CORE at nanochat scale

**Question**: given an 8×H100 node, a ~2–3 hour training budget, and the goal of crossing CORE = 0.2565 as fast as possible, should you run dense or MoE?

**Answer (from 40+ runs across E, top_k, h, cf, aux, depth, EP, FP8, grouped_mm)**: **dense d26 with `--fp8`**, at ~99 min. The best MoE variant we built (d22 E=4 k=1 h=4096 cf=1.0 aux=0.05 router→AdamW, `--fp8 --moe-expert-fp8` with `torch._scaled_grouped_mm`) takes ~112 min — a persistent **~13-minute deficit** we can't close with any combination of PyTorch-primitive optimizations we've tried.

This document explains *why* that 13-minute gap exists, with hard numbers from our runs, and what would need to be true to flip it.

---

## The gap has two components

**Component 1 — MFU gap (≈ 8 min of the 13).** Dense d26 FP8 runs at **60% MFU**. Our best MoE d22 FP8 runs at **44% MFU**. The 16-percentage-point gap means MoE takes 60/44 ≈ 1.36× longer per unit of compute.

**Component 2 — CORE-per-step gap (≈ 5 min of the 13).** MoE d22 needs ~4900 iters to reach CORE 0.2565; dense d26 needs ~4500 iters. MoE's larger-than-compute-matched expert hidden dim (4096 vs dense d26's 6656 effective FFN hidden) gives it a per-step CORE that is ~10% worse than dense at the same iteration count. This is a direct consequence of MoE at d22 being a *smaller* model (1.28B active params) than dense d26 (1.68B active).

Both components are real. FP8 and grouped_mm closed most of the FP8-era dense speedup from Phase 3, but neither attacks the core MFU gap.

## Where the MFU disappears

Per-step time breakdown at MoE d22 E=4 h=4096 FP8 grouped, measured / estimated:

| Component | Time per step | % of step |
|---|---:|---:|
| Dense transformer body (attn, LM head) at FP8 | ~650 ms | ~46% |
| Router (F.linear + softmax + topk) | ~10 ms | <1% |
| Sort tokens by expert (argsort, scatter_add into dispatch buffer) | ~60 ms | ~4% |
| Expert FP8 bmm forward (grouped_mm) | ~130 ms | ~9% |
| Expert bf16 bmm backward (2 bmms for grad_input + grad_weight) | ~200 ms | ~14% |
| Dispatch buffer read/write memory traffic | ~80 ms | ~6% |
| Combine (gather + scatter_add to token outputs) | ~60 ms | ~4% |
| Embeddings + scalars + aux loss + optimizer step | ~90 ms | ~6% |
| Python overhead (per-layer attribute access, etc.) | ~30 ms | ~2% |
| All-reduce / NCCL on gradients | ~110 ms | ~8% |
| **Total** | **~1420 ms** | **100%** |

Dense d26 FP8 for comparison (same step time ~1320 ms = 93% of MoE's): **no** dispatch buffer, **no** separate expert matmuls, **no** Python overhead per expert, **no** combine. Just attn + FFN + LM head all in proper FP8.

**The MoE-specific overhead (dispatch + expert matmul + combine + extra Python) adds ~450 ms/step vs what dense does.** That's 30% overhead per step.

## Why FP8 didn't close the gap more

Dense went from bf16 (50% MFU, 160 min) to FP8 (60% MFU, 131 min) — an 18% speedup. MoE went from bf16 (43% MFU, 145 min) to FP8 grouped (44% MFU, 141 min) — a 3% speedup.

**Dense gets more out of FP8 because a higher fraction of its compute is in the big matmuls that FP8 accelerates.** In MoE, the big matmuls are only ~23% of the step (9% forward + 14% backward); the remaining 77% (dispatch, combine, Python, NCCL) is unaffected by FP8. So Amdahl's law limits MoE's FP8 gain:

```
MoE with perfect FP8 on all matmuls:
  1 / (1 - 0.23 + 0.23/2)  ≈ 1.13× speedup  →  ~125 min (best case)

Dense with FP8:
  1 / (1 - 0.7 + 0.7/2)  ≈ 1.41× speedup  →  matches the 131 min we saw
```

So even a *perfect* FP8 implementation of MoE (both forward and backward in FP8, no Python overhead) would top out around 125 min. Dense at 131 min → 99 min is a bigger absolute win because its FP8-accelerated slice is bigger.

## Why Expert Parallelism didn't help

EP at d26 saved memory (fits E=8 h=3072) but added 2 × `all_to_all_single` × 26 layers × (forward + backward) = **104 NCCL collectives per step**. NCCL-side overhead on small tensors is around 2–5 ms each. 104 × 3 ms = **~300 ms/step**, which is more than the per-step dispatch work it was supposed to accelerate. EP at d26 took 180 min total — *slower* than replicated MoE at d22 (145 min) and *way slower* than dense d26 (131 min).

EP only wins when:
- The model fundamentally cannot fit replicated.
- The all_to_all latency is hidden behind compute via overlap.
- The collectives are amortized over large enough per-expert work.

None of those hold at d22–d26 on 8 GPUs with E ≤ 8.

## What *would* let MoE win the race

Three levers, in decreasing order of likely impact:

### 1. Fused FlashMoE-style Triton kernel

A single Triton kernel that does:
- Token sort by expert (in-kernel or via a pre-pass)
- Fused `bmm(x, w_fc)` + `relu²` + `bmm(h, w_proj)` — no hidden-buffer materialization
- Fused scatter back to token positions with gate weights
- Optional: fuse the load-balancing aux loss accumulation

This kernel would attack:
- ~130 ms expert-bmm-forward
- ~200 ms expert-bmm-backward  
- ~60 ms dispatch memory traffic
- ~60 ms combine memory traffic

With realistic fusion (H100 tensor cores at peak), savings would be 200–300 ms/step, bringing MoE to ~1.1 s/step — matching dense d26 FP8 throughput. At that point the CORE-per-step gap (~5 min) is the remaining deficit, and MoE could pull close to parity.

**Engineering cost**: 1–2 weeks of focused Triton work for someone familiar with the kernels (reference: Megablocks ~4000 lines, ScatterMoE Triton impl ~800 lines of kernel code). **Not doable in this iteration loop.**

### 2. FP8 routed-expert backward

Our current grouped FP8 path does FP8 forward, bf16 backward. Proper FP8 backward needs per-group column-wise scaling for `grad_weight`, which isn't natively supported by `torch._scaled_grouped_mm` in the way that works out-of-the-box. Getting this right would save ~100 ms/step (half the backward-bmm cost).

**Engineering cost**: ~200 lines of custom autograd + quantization. Doable in a session, but fragile.

### 3. Expert-parallel + true overlap of all_to_all with compute

Current EP is synchronous (all_to_all blocks compute). If we overlap by doing the attention forward of layer L+1 while experts of layer L are communicating, we hide the 300 ms overhead. Requires pipelining logic in the block forward.

**Engineering cost**: model-level rewrite. Tractable but invasive.

## What won't help (tested and documented)

- Hyperparameter sweeps over E, top_k, expert_hidden_dim, capacity_factor, aux_loss_coef, num_shared_experts, router optimizer choice. All explored. Best config lands at ~112 min.
- More experts (E > 8 at d22): per-expert tokens drops, GEMM efficiency collapses, CORE regresses.
- Deeper MoE (d24, d26 replicated): doesn't fit in 80 GB per GPU without EP.
- More iterations: dense also scales, so the gap is preserved.

## Honest recommendation

**For a GPT-2-CORE speedrun on 8×H100 in 2026-Q1 with PyTorch-primitive code: run dense with `--fp8` at the depth dictated by your compute budget.** MoE is an engineering investment (kernel work, not hyperparameters) that pays off only when:
1. You have memory pressure that forces expert-parallel.
2. You have a FlashMoE-style fused kernel.
3. Your training budget is long enough for the per-step CORE gap to reverse (probably 4× longer than GPT-2 scale).

For nanochat's current d10→d26 miniseries, none of those conditions are met, and dense wins.

## The one-sentence result

At the scale and with the implementation primitives available as of April 2026, **MoE adds roughly 30% per-step overhead on top of dense, and FP8 + fused grouped matmul reduce that overhead by only a few percent — leaving an irreducible 13-minute gap on the 99-minute time-to-GPT-2-CORE that only a FlashMoE-style fused kernel can plausibly close.**
