# MoE experiments in nanochat (ongoing)

## Why MoE at all

The nanochat philosophy (see `dev/LOG.md` Jan 7 "Miniseries v1") is that the user holds one dial — `--depth` — and the code scales everything else (width, LR, token horizon, weight decay) off it, producing a family of compute-optimal models. A candidate architectural change is only worth keeping if it lifts the **compute → loss / CORE** frontier across the whole family, not just at one point.

MoE is a natural candidate for that frontier:
- Routed experts add **total** parameters cheaply while keeping **active** parameters constant per token.
- More total capacity should → lower loss at matched compute.
- But: routing + dispatch + sparse memory traffic eat into MFU. If the MFU hit is bigger than the per-step loss win, MoE loses wall-clock even if it wins per-step.

Karpathy ran this experiment in Feb 2026 (see the inline note below) at d18 with `torch._grouped_mm` and concluded MoE is a **net negative wall-clock** for nanochat at GPT-2 scale. This document is a running log of replicating and stress-testing that claim at d12 with a different implementation.

## Goals

1. Refute or confirm Karpathy's "MoE is net negative wall-clock at our scale" claim, at d12, with our implementation.
2. Sweep `num_experts ∈ {2, 4, 8, 16, 32}` at fixed `top_k=2`, `num_shared_experts=1`, `depth=12`.
3. Preserve the single-dial principle: the changes have to be principled enough to work across all depths.
4. Produce actionable verdict: is MoE worth keeping, and if so under what conditions?

## Our MoE implementation (`nanochat/moe.py`)

| Decision | Value | Rationale |
|---|---|---|
| Routing | Token-choice top-k=2 softmax, renormalized over chosen experts | Standard Switch/GShard; user pushback against paper-faithful top-k=2 was minimal |
| Dispatch | Padded capacity `⌈1.25·k·N/E⌉` + `scatter_add` into `(E, cap, D)`, compile-safe static shapes | No custom Triton; stays readable; drops overflow tokens (contribute 0 via `keep` mask) |
| Shared experts | DeepSeek style: `num_shared_experts=1`, always active alongside routed path | Expected to stabilize training with cheap always-on compute |
| Compute-matched sizing | `expert_hidden = 4·n_embd / (top_k + num_shared)` rounded to 64 | Active FFN params per token ≈ dense MLP, so sweeping E varies **total capacity** not compute |
| Aux loss | Switch-style: `λ·E·Σ_e(f_e·p_e)`, λ=0.01, `f` detached | Cheap; `p` provides the balancing gradient |
| Expert weights | 3D `nn.Parameter (E, D, H)` / `(E, H, D)` | Groups naturally under Muon once `optim.py` state_shape supports leading dims |
| Optimizer | Muon for router + experts (extended to 3D), AdamW for embeddings/scalars | No AdamW-on-experts compromise |
| Scaling-law target | `active_transformer_matrices + lm_head` (DeepSeek convention) | Keeps `--target-param-data-ratio` compute-optimal across E |

### How we differ from Karpathy's Feb 2026 implementation

Karpathy used `torch._grouped_mm` + optional Triton alignment-padding kernels (from torchtitan), and sigmoid-gated DeepSeekV3-style routing with bias-nudging load balancing. He reported the grouped_mm/layout machinery took MFU from ~46% → ~35% at d18.

We use a simpler scatter-based dispatch with `torch.bmm` over `(E, capacity, D)`:
- **Pro**: readable, no undocumented APIs, no Triton, no FP8 gaps.
- **Con**: wastes FLOPs on the overflow slots (the ones whose `keep=0`), and may suffer memory-bound throughput at large E + small tokens-per-expert.

So our implementation probably sits on a *different* point of the speed/simplicity tradeoff than his. Neither is obviously better — different failure modes.

## Methodology (current runs)

- **Depth**: 12 (n_embd=768, 12 layers, n_head=6, seqlen=2048)
- **Iterations**: 1500 (~4–5 min training on 8×H100)
- **Batch**: total 524,288 tokens/step, `device-batch-size=32`
- **Eval**: CORE metric with `--core-metric-max-per-task=500` (~4 min eval)
- **Precision**: bf16 compute, fp32 master weights, no FP8 (would need custom autograd for 3D experts — deliberately scoped out, matching Karpathy's observation)
- **Data**: ClimbMix shards, tokenizer as-is from prior runs
- **Hardware**: single 8×H100 node (H100_NVLINK_80GB)

### Known methodology gaps

1. **No dense d12 baseline yet** at 1500 iters → can't directly answer "is MoE net-positive wall-clock at d12?" Dense run is #3 below, in flight.
2. **E=2 ran at 600 iters** (original smoke test before the user bumped iters). Its numbers are **not comparable** to the 1500-iter runs. Will re-run.
3. **Single seed** per config. No run-to-run variance estimate. Conclusions from 1-run-per-config are directional only.
4. **CORE noise at small scale**: Karpathy's own Jan 11 update flagged CORE as noisy below d18. We're at d12 so expect ±0.01 or more noise on CORE; interpret single-digit CORE deltas with caution.
5. **Per-step comparisons across different iter counts are confounded** by the LR warmdown schedule (`warmdown_ratio=0.65`, so 1500-iter run is in peak LR at step 500 while 600-iter run has already decayed). Only the **final** step's `val_bpb` / `CORE` can be compared across runs.

## Running experiment log

### Run 0 — E=2 d12 @ 600 iters (smoke test, 2026-04-18)

**Purpose**: verify the MoE + Muon + helm-chart-driven iteration loop works end-to-end.

| Metric | Value |
|---|---|
| `num_experts` / `top_k` / `num_shared_experts` | 2 / 2 / 1 |
| `active_total` | 286,280,162 |
| Total params | 286,298,594 |
| Iterations | 600 |
| Training wall-clock | 1.71 min |
| Peak VRAM | 35.1 GiB |
| Final val_bpb | 0.9478 |
| Final CORE metric | 0.0932 |
| Training MFU (steady-state) | ~29% |

**Verdict**: pipeline works. aux_loss stayed non-zero (~0.03), no NaN, val_bpb monotonically down. **Do not compare to later runs** — iter count differs.

### Run 1 — E=4 d12 @ 1500 iters (2026-04-18)

| Metric | Value |
|---|---|
| `num_experts` / `top_k` / `num_shared_experts` | 4 / 2 / 1 |
| `active_total` | 286,298,594 (+ 18k router vs E=2) |
| Iterations | 1500 |
| Training wall-clock | 4.86 min |
| Peak VRAM | 35.6 GiB |
| Final val_bpb | **0.8713** |
| Final CORE metric | **0.1403** |
| Final train/aux_loss | 0.120 |
| Training MFU (steady-state) | ~24% |

**Critical observations:**
- **MFU dropped from ~29% (E=2) → ~24% (E=4).** Even without `grouped_mm`, the scatter-based dispatch pays a throughput cost as E grows.
- aux_loss settled at 0.12 — routing is not collapsing but load is imbalanced-ish. Expected from softmax top-k at small E.
- val_bpb at step 250 matched E=2's step-250 value almost exactly (1.082 vs 1.081) — confirms active-compute matching worked: the two configs are indistinguishable early, diverge only once capacity matters.

### Run 2 — **dense d12 @ 1500 iters (baseline)** (2026-04-18)

| Metric | Value |
|---|---|
| `num_experts` | 1 (dense MLP path) |
| `active_total` | 286,261,730 (just 37k less than E=4 — the missing router + shared-expert weights) |
| Iterations | 1500 |
| Training wall-clock | **3.18 min** |
| Peak VRAM | **28.0 GiB** |
| Final val_bpb | **0.8749** |
| Final CORE metric | **0.1429** |
| Training MFU (median / mean) | **39.39% / 39.30%** |

### The first apples-to-apples — dense vs MoE E=4 at matched 1500 iters

| | Dense d12 | MoE E=4 d12 | Delta |
|---|---|---|---|
| `active_total` | 286M | 286M | — (matched by design) |
| Total params | 286M | 337M | MoE +18% (routed expert weights) |
| val_bpb | 0.8749 | **0.8713** | MoE −0.0036 (**−0.41%**) |
| CORE | **0.1429** | 0.1403 | **dense +0.0026** (~1.8% relative) |
| Training wall-clock | **3.18 min** | 4.86 min | MoE **+53%** slower |
| Peak VRAM | **28.0 GiB** | 35.6 GiB | MoE **+27%** |
| Training MFU | **39.3%** | 23.9% | MoE loses **−15pp** absolute / **−39%** relative |

**This is the headline result.** At matched iteration count and matched active compute, MoE-E=4 produces:
- A **negligible** (noise-level) val_bpb gain.
- An actual *regression* on CORE (the primary downstream metric).
- A ~50% wall-clock penalty.
- A ~40% relative MFU drop.

At **matched wall-clock** (which is what actually matters for "should I keep this in the repo"), dense d12 could run for ~2300 iters in MoE's 4.86 min, which with a proper warmdown schedule would almost certainly beat MoE's numbers on both val_bpb and CORE.

**This is a stronger negative result than Karpathy's Feb 2026 writeup.** He reported ~46% → ~35% MFU at d18 with `torch._grouped_mm`; we're seeing ~39% → ~24% at d12 with a simpler scatter-based dispatch. Our dispatch path is more overhead-heavy than his — which was foreseeable (scatter_add is memory-bound; `grouped_mm` is dense-compute-bound). Either way the verdict rhymes: at nanochat scale, routing overhead eats the capacity win.

### Runs 3–6 — E ∈ {2, 8, 16, 32} d12 @ 1500 iters (planned)

- **E=2 @ 1500 iters**: re-run for apples-to-apples with the other MoE configs.
- **E=8/16/32**: sweep increasing total capacity at fixed active compute.
- Expected trend (capacity argument): lower val_bpb as E grows.
- Expected counter (routing overhead): MFU decreasing as E grows, possibly steeply for E=16+.
- Interesting breakeven question: is the marginal val_bpb gain per extra expert > marginal MFU cost, **at each point on the E axis**?

## Analysis framework (how we'll answer "did MoE win?")

Fair comparison requires plotting two things on the same axes:

1. **Per-step** `val_bpb` and `CORE` — MoE will almost certainly win here (matches Karpathy and common sense).
2. **Per-wall-clock** and **per-FLOP** `val_bpb` and `CORE` — this is where MoE either beats or loses to dense.

Concretely, the right scoring function (matching the nanochat miniseries philosophy):

```
for each config (dense, E=2, 4, 8, 16, 32):
    plot (total_training_time, val_bpb) and (total_training_flops, CORE)
```

And the right verdict:

> **MoE is worth it iff its wall-clock / FLOP curve lies strictly below dense's**, AND the improvement is larger than the code-complexity tax.

Anything weaker than that — e.g. "MoE per-step is better" — is not actionable for the miniseries.

## Open questions to revisit after the sweep

- **Does MFU scale cleanly with E, or are there cliffs** (e.g. when per-expert tokens drop below some kernel-efficiency threshold)?
- **Token drop rate**: at E=32 with capacity_factor=1.25, how many tokens overflow? If drop rate is high, either raise capacity factor (worse throughput) or accept biased routing.
- **Karpathy's FP8 gap**: our run is bf16 for everything in the MoE block. On dense, FP8 adds maybe +10–20% throughput. Without FP8 for routed experts, MoE pays a compounding throughput penalty vs a FP8 dense baseline. Our dense baseline is also bf16, so the direct comparison here is fair; but for a real "nanochat with MoE vs nanochat with dense+FP8" comparison, MoE is further behind.
- **Does the depth-dial principle survive?** Run the same MoE config at d10, d14, d18 and check whether the val_bpb improvement is consistent across depths. If it's only a d12 artifact, it's not interesting.

## Current tentative verdict (after dense baseline)

With dense d12 and MoE E=4 both at 1500 iters in hand, the early read is **MoE is not winning at d12**:

- **val_bpb is a tie within noise** — dense 0.8749 vs MoE 0.8713. At d12/1500iters the two runs are indistinguishable on vocab-invariant loss.
- **CORE is worse for MoE** — 0.1403 vs 0.1429 (dense). CORE is noisy at d12 (see Karpathy's Jan 11 note), so single-run CORE deltas aren't decisive; but there is no signal in this data that MoE improves downstream capability at this compute budget.
- **Throughput is substantially worse for MoE** — 24% MFU vs 39% MFU, +53% wall-clock, +27% peak VRAM. This is consistent in direction with Karpathy's Feb 2026 finding, and *worse in magnitude*. Our simpler scatter-based dispatch is more memory-bound than his `grouped_mm` — at small tokens-per-expert we're burning more bandwidth for the same compute.

**So far, the data corroborates Karpathy's Feb 2026 verdict.** The sweep over E ∈ {2, 8, 16, 32} is the next test: the only scenario where MoE recovers is if more total capacity at higher E gives a per-wall-clock-unit benefit that beats dense. Given E=4 already lost wall-clock, and the MFU penalty is expected to scale up (not down) with E, this feels unlikely — but worth running for definitive numbers.

Will update this section again after the sweep completes.

---

*Note for future maintainers*: if you come back to MoE in nanochat, the way to revisit this negative result is almost certainly **not** another PyTorch-primitives implementation. It's either (a) a fused FlashMoE-style kernel that collapses routing + dispatch + expert matmul into one kernel call, or (b) expert-parallel / tensor-parallel MoE that actually uses the 8 GPUs to host separate experts instead of replicating everything. Neither is a weekend project. The bar is that it must lift the d10→d30 compute→loss frontier, not just win at one operating point.
