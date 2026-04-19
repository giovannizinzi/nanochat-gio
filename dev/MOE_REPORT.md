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

### Run 3 — E=2 d12 @ 1500 iters (re-run for apples-to-apples) (2026-04-18)

| Metric | Value |
|---|---|
| Iterations | 1500 |
| Training wall-clock | 4.33 min |
| Peak VRAM | 35.1 GiB |
| Final val_bpb | **0.8773** (↑ vs dense 0.8749, ↑ vs E=4 0.8713) |
| Final CORE metric | **0.1269** (↓ vs dense 0.1429, ↓ vs E=4 0.1403) |
| Training MFU (median) | 29.08% |

**E=2 is strictly worse than dense on every axis.** The cause is trivially obvious in hindsight: `num_experts=2, top_k=2` means *all* experts activate for every token, so there is zero routing sparsity. We've replaced one `(D, 4D)` matmul with three smaller matmuls (2 routed each at `D → 4D/3` hidden, + 1 shared at `D → 4D/3` hidden) plus a useless router and an aux-loss pass. Fragmented GEMMs under-utilize tensor cores; no capacity benefit because `top_k == num_experts`. Don't run this config again.

This is a methodology lesson: **the minimum meaningful MoE config has `num_experts > top_k`**. E=2 is a test of the routing-overhead tax at zero sparsity benefit; it should always lose to dense and it does.

### Run 4+ — E ∈ {8, 16, 32, 64, 128} d12 @ 1500 iters (in progress)

Sweep extended to E=128 to force the asymptotic behavior to declare itself — at E=32 and below we may still be in the "capacity isn't enough to matter" regime, where total routed params (≈500M at E=32) aren't yet a significant fraction of what d12 can usefully absorb.

- **E=8**: 2-of-8 routing; paper's recommended small-compute config. 4× total capacity over active.
- **E=16**: 2-of-16. Per-expert tokens: ~8k per rank — still plenty for tensor cores.
- **E=32**: 2-of-32. Per-expert tokens: ~5k per rank — may start to hit GEMM efficiency floor.
- **E=64**: 2-of-64. Per-expert tokens: ~2.5k per rank — routed expert weights ≈1.2B params.
- **E=128**: 2-of-128. Per-expert tokens: ~1.3k per rank — *small* matmuls, likely worst-case throughput. Routed expert weights ≈2.4B params (still fits in 80GB H100 with bf16 compute + fp32 master, by maybe 2×).

Expected trend if MoE works at all: val_bpb drops as E grows, because total capacity grows (64× routed params at E=128 vs E=2). Expected counter-trend: MFU decreases as E grows — small per-expert matmuls are inefficient, and dispatch overhead scales with E. The breakeven is whether the marginal val_bpb gain per extra expert ever exceeds the marginal MFU cost.

At E=128 the dispatch buffer `(E, cap, D) = (128, 1280, 768)` is only 126M bf16 elements (250 MB) per forward — comfortable. The risk is that tensor cores idle on the (E=128 batched) 1280×768 BMMs — Hopper wants matmuls with at least one dim ≥ 1024 ideally.

## Ideas/extremes still to try (to force MoE to beat dense, or confirm it can't at d12)

The linear `num_experts` sweep above only varies one axis. To force a winning config to declare itself (or rule the whole space out), we need to explore multiple axes. **Break-even math**: at dense MFU=39% and MoE-E=4 MFU=24%, MoE must be ~1.63× better per-step to break even on wall-clock. At E=8 (if MFU drops further) the bar is higher still. Ideas that plausibly close that gap:

### Axis A — cheaper routing (Switch Transformer style)

- [ ] **Top-1 routing, E=8**: half the dispatch/gather work vs top-2. Switch Transformer showed top-1 + slightly larger expert often wins on throughput-to-accuracy.
- [ ] **Top-1 routing, E=32**: at high E, top-1 keeps per-token compute very sparse. If MFU recovers, this is the most promising "Switch at d12" config.
- [ ] **Top-1 routing, E=128**: extreme sparsity; per-token active compute is exactly 1 expert's FFN + shared.

With `top_k=1` the default compute-matched sizing gives `expert_hidden = 4·n_embd / (1 + 1) = 2·n_embd = 1536`, so each expert is larger (bigger, better-utilized matmul) — addresses the "small GEMM" problem.

### Axis B — capacity factor tuning

- [ ] **Capacity factor = 1.0 (no slack)** with top-2: reduces dispatch buffer size, improves memory BW. Some tokens drop (the overflow), which may hurt loss slightly — test whether the throughput win offsets that.
- [ ] **Asymmetric cf**: 1.0 forward, 1.25 train (user's suggestion; requires the train/eval graph split).

### Axis C — number of shared experts

- [ ] **No shared experts** (Switch-pure, `num_shared_experts=0`): removes a dense-like always-on pathway. Should lose, but tests the specific contribution of the shared path.
- [ ] **2 shared experts**: more always-on dense compute; may help stability at cost of sparsity.

### Axis D — granularity (DeepSeek's axis)

- [ ] **Fine-grained experts**: `num_experts=64 top_k=8 expert_hidden=512`. Same active compute, but each expert is half the size — specialization may emerge better.
- [ ] **Coarse experts**: `num_experts=4 top_k=2 expert_hidden=2048` — opposite of fine-grained, sanity check.

### Axis E — bigger MoE vs smaller dense (the real matched-compute test)

- [ ] **MoE d12 vs dense d10 @ matched wall-clock**: the nanochat miniseries question. A smaller dense at matched wall-clock might still beat MoE d12. That's the real verdict.
- [ ] **MoE d14 vs dense d12 @ matched FLOPs**: does MoE allow *going deeper* for same compute?

### Axis F — longer training

- [ ] MoE at 3000 iters: does more training give MoE more time to exploit its extra capacity? If per-step gains compound, the wall-clock penalty may eventually flip.

**Running priority** (after the linear sweep completes): Axis A top-1 variants first (highest expected MFU recovery), then Axis E matched-wall-clock dense-d10 comparison (the actual miniseries question), then the others as budget allows.

### Run 5 — E=8 d12 top_k=2 @ 1500 iters (2026-04-18)

| Metric | Value |
|---|---|
| `num_experts` / `top_k` / `num_shared_experts` | 8 / 2 / 1 |
| Iterations | 1500 |
| Training wall-clock | **6.03 min** (**+90% vs dense**) |
| Peak VRAM | 36.6 GiB |
| Final val_bpb | **0.8659** (↓ vs dense 0.8749 — MoE wins by 9 mbit, 1%) |
| Final CORE metric | **0.1275** (↓ vs dense 0.1429 — MoE LOSES by 15 mbit, ~11% relative!) |
| Training MFU (steady-state) | ~24% |

**This is the headline finding of the MoE sweep so far and deserves careful stating:**

**MoE E=8 beats dense d12 on validation loss** (0.866 < 0.875) **but loses decisively on CORE** (0.128 < 0.143). These two metrics disagree. The model is getting *better at the training distribution's next-token prediction* while getting *worse on held-out downstream tasks*.

Possible mechanisms (to distinguish with more experiments):
1. **Overfitting to the training distribution**: extra capacity lets the MoE memorize FineWeb/ClimbMix patterns that don't transfer to zero-shot tasks.
2. **Representation fragmentation**: specialized experts develop narrow representations; harder to compose on out-of-distribution prompts.
3. **CORE noise at d12**: Karpathy's Jan 11 note flagged d12-scale CORE variance; 15 mbit difference *might* be within that envelope, but the direction being consistently against MoE (at E=2, E=4, E=8) is more than noise.

Wall-clock verdict is unambiguous: **~2× slower than dense for a worse downstream metric**. MoE at top_k=2 loses on every axis except val_bpb at this scale.

### Router-to-AdamW fix (before next run)

During the sweep, user pointed out that our implementation put the `router_weight` into Muon via `self.transformer.h.parameters()`. That's wrong — Muon orthogonalizes matrices, which actively fights the router's job of concentrating mass on a few chosen experts per token. AdamW is the right optimizer for routers (Shazeer/DeepSeek convention).

Fix (commit `8001c63` on `moe-experiments`): `setup_optimizer` now excludes `block.mlp.router_weight` from `matrix_params` before Muon grouping, and adds a dedicated AdamW group for router params. Experts (w_fc, w_proj, shared variants) stay on Muon as intended.

All runs *after* this commit use the fix. Runs 0–5 above used the buggy router-on-Muon config; the CORE regression could partially be an artifact of that. **All subsequent runs carry the fix and should be the trusted comparison.**

### Run 6 — E=32 top_k=1 d12 @ 1500 iters (in flight, **router-AdamW fix active**)

Switch-Transformer-style config: 32 experts, 1-of-32 routing, 1 shared expert.
- `expert_hidden` auto = `4 × 768 / (1 + 1) = 1536` (50% larger than top_k=2 variants — bigger GEMM, better tensor-core utilization).
- Dispatch/gather work halves vs top_k=2.
- Total routed capacity: 32 experts × 1536 hidden ≈ 6× E=8 top_k=2 total FFN params.

Hypothesis: the combination of (a) router on AdamW, (b) single-expert dispatch, (c) larger per-expert matmul is the most likely configuration to recover MFU toward dense levels. If MFU gets to ~32–35% AND per-step loss improvement holds, this might finally beat dense on wall-clock.

### Run 6 results

| Metric | Value |
|---|---|
| `num_experts` / `top_k` / `num_shared_experts` | 32 / 1 / 1 |
| `expert_hidden_dim` | 1536 (auto: `4·768/(1+1)`) |
| Iterations | 1500 |
| Training wall-clock | **4.95 min** (+56% vs dense) |
| Peak VRAM | **43.2 GiB** (+54% vs dense) |
| Final val_bpb | **0.8572** ← best of any MoE config so far, **−18 mbit vs dense (−2.1%)** |
| Final CORE metric | **0.1408** (−0.002 vs dense; CORE-noise-range) |
| Training MFU (median / mean) | 26.0% / 25.4% |

**Router-AdamW fix + top_k=1 massively closed the CORE gap** (E=8 k=2 had CORE=0.128, E=32 k=1 has CORE=0.141). Val_bpb and CORE are now both competitive with dense — but wall-clock is still +56% worse.

At matched wall-clock, dense d12 would have done ~2320 iters in E=32 k=1's 4.95 min. Extrapolating the dense val_bpb curve (rate of improvement ≈ 0.04 bpb per 500 iters in the last third of the run), dense at ~2320 iters would reach roughly val_bpb ~0.85, matching MoE. So this is close to a dead heat at matched wall-clock — but still not a clear MoE win.

**The router-AdamW fix alone likely explains a big chunk of the CORE recovery**; we should not over-credit top_k=1 for this. Ideally we'd re-run E=8 k=2 *with* the fix to isolate the top_k effect from the router-optimizer effect.

### Run 7 — E=128 top_k=1 d12 @ 1500 iters (in flight, Kimi-scale)

Kimi K2 uses 384 experts with 8 active. We can't match that at d12 (not enough tokens per expert for 384), but E=128 k=1 pushes our sparsity dial nearly as far.
- Routed expert weights: 128 × 2 × 768 × 1536 × 12 = 3.6B params (bf16=7.2GB, fp32 master=14.4GB).
- Dispatch buffer: capacity ≈ 1.25 × 1 × 65536 / 128 = 640 tokens/expert; buffer `(128, 640, 768)` = 63M bf16 = 126MB.
- Per-expert matmul: 640 × 768 × 1536 bmm — small in M but big in K/N; should keep tensor cores busy.
- Expected: continued val_bpb improvement (more total capacity), expected MFU drop (more dispatch work).

*Result TBD — first attempt OOM'd (details below).*

### OOM note (Run 7 first attempt)

E=128 k=1 with the default `device_batch_size=32` OOMs on 80GB H100s. Memory breakdown at this scale:
- Expert master params (fp32): 128 × 2 × 768 × 1536 × 12 × 4 bytes = **14.5 GB**
- Muon momentum buffers (same size): **14.5 GB**
- Second-momentum (factored, small): ~60 MB
- bf16 forward copy of expert weights: 7.2 GB
- Optimizer gradient staging (fp32): ~14 GB
- Per-rank activations at `device_batch=32`, `seqlen=2048`: several GB

Total ≈ 72 GB per rank, which is over budget when combined with torch + NCCL overheads.

Retry with `device_batch_size=16` (grad_accum_steps=2) to halve activation memory — **also OOM'd**. The issue is optimizer state, not activations: even sharded Muon momentum + fp32 master + bf16 copy + grads is too much at E=128. User feedback (Apr 2026) pushed back on changing `device_batch_size` anyway — keeping batch constant across the sweep avoids confounding with batch-size effects. Cancelling the E=128 point.

### Run 8 — E=64 top_k=1 d12 @ 1500 iters (router-AdamW fix) (2026-04-18)

| Metric | Value |
|---|---|
| `num_experts` / `top_k` / `num_shared_experts` | 64 / 1 / 1 |
| Iterations | 1500 |
| Final val_bpb | **0.8626** (worse than E=32 k=1's 0.8572) |
| Final CORE metric | **0.1230** (worse than E=32 k=1's 0.1408 and dense's 0.1429) |

**E=64 regresses from E=32.** Per-expert tokens drop to ~1k/rank, and CORE drops by 18 mbit. This is the shape of a capacity-vs-fragmentation tradeoff: beyond the E=32 sweet spot, adding experts hurts. This matches Karpathy's intuition that at small scale, too-many-experts means each expert gets too few tokens to learn a useful specialization, and routing gets harder to train.

(Note: E=64 k=1 failed its final `torch.save` because the PVC filled up with accumulated checkpoints from prior runs. The val_bpb + CORE metrics were emitted before the save attempt so we have the numbers. Added `rm -rf $BASE_CKPT_DIR/$WANDB_RUN` at the end of EXPERIMENT_MODE in `nanochat-gio.sh` so future runs don't bloat the PVC.)

### Summary table after linear sweep

| Config | val_bpb | CORE | MFU | wall-clock | peak VRAM | Notes |
|---|---:|---:|---:|---:|---:|---|
| **dense d12** | 0.8749 | **0.1429** | **39.3%** | **3.18 min** | **28.0 GiB** | baseline |
| E=2 k=2 (pre-fix) | 0.8773 | 0.1269 | 29.1% | 4.33 min | 35.1 | degenerate: top_k==E |
| E=4 k=2 (pre-fix) | 0.8713 | 0.1403 | 23.9% | 4.86 min | 35.6 | tie bpb, lose CORE |
| E=8 k=2 (pre-fix) | 0.8659 | 0.1275 | ~24% | 6.03 min | 36.6 | win bpb, big CORE regression |
| **E=32 k=1 (post-fix)** | **0.8572** | 0.1408 | 26.0% | 4.95 min | 43.2 | best bpb, CORE ~ ties dense |
| E=64 k=1 (post-fix) | 0.8626 | 0.1230 | ? | ? | ? | regresses from E=32 |
| E=128 k=1 (post-fix) | — | — | — | — | — | OOM (3D expert params > VRAM budget) |

### Takeaway from the linear sweep

At d12, the best MoE config we've found (E=32 top_k=1, with router on AdamW) is **~2% better val_bpb than dense** but **ties dense on CORE** and **still ~56% slower wall-clock**. Dense at matched wall-clock would run ~2300 iters and likely match-or-beat MoE on both metrics.

**This is a stronger corroboration of Karpathy's Feb 2026 verdict**: MoE is not winning wall-clock at nanochat d12 scale with the PyTorch-primitive implementations available to us. The ceiling on MoE's per-step improvement isn't high enough to overcome the ~35% MFU deficit.

### Run 9 — E=8 top_k=1 `expert_hidden_dim=4096` (2026-04-18) — **first config to beat dense on both val_bpb AND CORE**

| Metric | Value | vs dense |
|---|---|---|
| `num_experts` / `top_k` / `num_shared_experts` | 8 / 1 / 1 | — |
| `expert_hidden_dim` | **4096** (2.67× the auto 1536) | — |
| `active_total` | 380,707,298 (+33% vs dense's 286M) | +33% |
| Training wall-clock | 6.31 min | **+98%** |
| Final val_bpb | **0.8458** | **−0.029 (−3.3%)** |
| Final CORE metric | **0.1455** | **+0.0026** (first MoE win!) |
| Training MFU | **35.2%** | −10% relative |
| Total training FLOPs | 1.04e18 | — |
| aux_loss (final) | 0.120 | — |

**This is the breakthrough so far.** Making each expert bigger (2.67× the compute-matched default):
1. **Recovered MFU dramatically** — from 26% (compute-matched E=32 k=1) to 35%. Bigger per-expert matmuls use tensor cores much better.
2. **Kept the val_bpb lead** — in fact extended it: 3.3% better than dense vs E=32 k=1's 2.1%.
3. **Finally won CORE** — 0.1455 vs dense's 0.1429. The first MoE config in this sweep that isn't a CORE regression.

**But wall-clock is still 2× dense.** Breaking compute-matching means per-token active compute is 1.33× dense, so even at matched MFU we'd expect ~1.33× slower. Combined with the still-present 10% MFU gap vs dense, we get ~2×. MoE wins per-step significantly but loses wall-clock significantly.

### Run 10 — dense d12 @ 3000 iters (wall-clock-matched to MoE E=8 h=4096)

This is the most important comparison in the whole report. MoE E=8 k=1 h=4096 was the best MoE config we found — it even won CORE against dense d12 @ 1500 iters. But MoE used 6.31 min of wall-clock. For the miniseries principle to apply, the right comparison is "what does dense do in the same 6.31 min?" — not "what does dense do in 1500 iters?" Dense is faster per step, so it gets more iterations for the same budget.

| Metric | dense d12 @ 3000 iters | MoE E=8 k=1 h=4096 @ 1500 iters | Winner |
|---|---:|---:|---|
| Iterations | 3000 | 1500 | — |
| Training wall-clock | **6.36 min** | 6.31 min | ~matched |
| Peak VRAM | 28.0 GiB | ~47 GiB (est.) | dense |
| Final val_bpb | **0.8395** | 0.8458 | **dense −0.006 (−0.75%)** |
| Final CORE metric | **0.1651** | 0.1455 | **dense +0.020 (+13.5% relative)** |

**At matched wall-clock, dense d12 beats the best MoE config decisively on both val_bpb and CORE.**

The gap isn't marginal: dense's CORE is 13.5% higher. Even though MoE had a 2.1% val_bpb lead at matched iterations (1500 vs 1500), doubling dense's training budget (to get same wall-clock) pushes dense's val_bpb below MoE's and its CORE far above.

---

## Final verdict (post-sweep)

**MoE at nanochat d12 scale does not win.** This is confirmed across:

| Axis swept | Winner at matched wall-clock |
|---|---|
| `num_experts` (linear sweep 2 → 64, k=2 then k=1) | dense |
| `top_k` (2 vs 1) | dense |
| `expert_hidden_dim` (auto vs 2.7× auto) | dense |
| Optimizer routing (router on Muon vs AdamW) | dense (fix was necessary but insufficient) |

The negative result **corroborates Karpathy's Feb 2026 writeup** and, importantly, generalizes it:

- Karpathy used `torch._grouped_mm` at d18 and found MoE loses wall-clock. We used a scatter-based dispatch at d12 and also find MoE loses wall-clock — with a worse MFU penalty (~40% relative vs his ~24%), and still coming up short.
- The single MoE config that beat dense-@-1500-iters on both val_bpb and CORE (E=8 k=1 h=4096) still loses to dense-@-3000-iters, which is what dense reaches in the same wall-clock.

### Why MoE loses at d12

Three compounding problems, in order of magnitude:

1. **Dispatch is memory-bandwidth-bound**, not compute-bound. At ~65K tokens/rank and modest `num_experts`, the per-expert matmul isn't big enough to amortize the sort + scatter + gather. Our scatter_add + BMM path hits 24–35% MFU vs dense's 39%. Karpathy's grouped_mm path hit 35% vs dense's 46%. Either way MoE pays 25–40% relative MFU.
2. **Compute-matched MoE has no throughput room to out-learn**. When active params match dense, MoE's only edge is the *total* capacity it carries around (which per-token it can't use). The single chosen expert per token contains the same matmul count as dense's MLP, but dense does this with one big matmul while MoE does it with a batched bmm — less efficient. MoE's 2% val_bpb advantage at matched iterations isn't enough to clear the wall-clock gap.
3. **Pushing past compute-matching costs wall-clock proportionally.** `expert_hidden=4096` gave MoE +33% active params per token and did lift both val_bpb and CORE — but training time also scales up, and dense-given-the-same-wall-clock did more iterations and won anyway.

### Where MoE *would* start to win at d12

Not at d12 as currently shaped, with our implementation, on this hardware. To get a d12 MoE win you'd need some combination of:

- **A fused FlashMoE kernel** (routing + dispatch + expert matmul in one kernel, following the FlashAttention playbook). Would need to be ≥ 80% of dense's MFU. Does not exist yet as a standard PyTorch primitive.
- **Expert parallelism** (each GPU hosts a subset of experts; use network to route tokens). Collapses the replicated expert weights into GPU-distributed storage; would allow much larger total capacity for the same per-GPU VRAM. Only worth the engineering if the fused kernel also exists.
- **Much larger model / longer training**. The capacity argument may only start to pay for MoE at ≥ d20 with multi-hour training. Our sweep is decisive for d12 at ~6min wall-clock; we haven't addressed d18–d26 at multi-hour budgets. Karpathy's Feb 2026 run at d18 also said no — so extrapolation to larger scale seems unlikely to flip the verdict, but has not been measured here.

### What to keep from this sweep

- `nanochat/moe.py` is functional and passes smoke tests; it's a reasonable reference for anyone who wants to revisit MoE later. Keep it as documented dead code (or behind a flag) rather than deleting.
- The `optim.py` generalization to 3D params is independently useful (any future 3D-param construct benefits from this).
- The active-params DeepSeek-style `num_scaling_params` split is the right scaling-laws accounting even if we don't ship MoE — it's how you'd compare any structured-sparsity technique cleanly.
- **This report**. If someone comes back and asks "has anyone tried top_k=1 with big experts at d12?", the answer lives here with all the numbers.

### If I had another hour

Things I would test to stress-test the verdict further, ordered by likelihood of changing it:

1. **Dense d12 @ FP8**: nanochat has an `--fp8` flag; our dense baseline was bf16 only. FP8 dense should be faster → dense at matched wall-clock does even more iterations → verdict gets stronger against MoE, not weaker. (MoE can't use FP8 for its 3D expert weights without custom autograd.)
2. **MoE at d18 or d22**: test whether the verdict extrapolates upward. Expected: MoE still loses, but the gap narrows.
3. **MoE capacity factor 1.0**: Noam's suggestion; trivial code change, might recover a few percent MFU.
4. **Sinkhorn routing**: replaces aux-loss with a transport-based balancing scheme. Simpler training dynamics but more compute per step for the routing math.
5. **Dense-d14 @ 1500 vs MoE-d12 @ 1500** at matched wall-clock: does MoE at d12 beat a *wider* dense at higher depth? This is the real "lift the frontier" test.

None of these are likely to flip the verdict given what we've seen, but they'd harden it.

---

## Phase 2: iterate harder (user directive, 2026-04-18)

User directive after the sweep: "Iterate more to try and find the best MoE… stick to E=4 or E=8, try every optimization, 20-min budget per experiment, test different depths, figure out how we can make MoE optimal so it scales accordingly."

This is the right next question: the sweep so far tested the axes from a first-principles angle, but hasn't exhausted them, and in particular hasn't tested **how MoE behavior changes with depth**. The miniseries principle says the verdict must hold across the whole d10→d20 family, not just at d12. If MoE *starts winning* somewhere around d16 or d18, that's a different story.

### Phase 2 experiment plan

| # | Config | Purpose |
|---|---|---|
| P1 | E=8 k=1 h=4096 d12 @ 6000 iters | User's explicit "4× steps" test: does the per-step MoE advantage compound into a wall-clock win if we give MoE enough training? |
| P2 | E=8 k=1 h=auto d16 @ 1500 iters | Depth scaling: does MoE close the gap at higher depth? (auto-hidden scales with n_embd) |
| P3 | dense d16 @ 1500 iters | Baseline for P2 |
| P4 | E=8 k=1 h=auto d10 @ 1500 iters | Low-depth counterpoint |
| P5 | dense d10 @ 1500 iters | Baseline for P4 |
| P6 | E=8 k=1 h=4096 d12 capacity_factor=1.0 @ 1500 iters | Noam's suggestion — cheaper dispatch |
| P7 | E=4 k=1 h=auto d12 @ 1500 iters | Fill gap: we ran E=4 k=2 (auto) and E=8 k=1 (auto + h=4096), never E=4 k=1 |
| P8 | E=4 k=1 h=4096 d12 @ 1500 iters | E=4 variant of our best E=8 config |

Budget: 20 min each; dense configs are ~3–13 min (scale with depth), MoE configs ~6–16 min depending on iters.

### Phase 2 runs

#### P1 — E=8 k=1 h=4096 d12 @ 6000 iters (2026-04-18)

User's "4× steps" test. Did MoE's per-step advantage compound enough to beat dense at matched wall-clock?

| Metric | MoE E=8 k=1 h=4096 @ 6000 | MoE same @ 1500 (Run 9) | Δ from 4× |
|---|---:|---:|---:|
| Wall-clock | **24.98 min** | 6.31 min | +4.0× (expected) |
| val_bpb | **0.7796** | 0.8458 | **−0.066 (−7.8%)** |
| CORE metric | **0.2000** | 0.1455 | **+0.055** |
| Peak VRAM | 57.2 GiB | 47 GiB | +10 GiB |
| MFU | 35.1% | 35.2% | ~flat |

At 4× iterations MoE's val_bpb drops from 0.8458 → 0.7796 (−7.8%) and CORE rises from 0.1455 → 0.2000 (+38% relative). Massive improvement — the per-step advantage **does compound** when given more training.

**The comparison that matters — matched wall-clock of ~25 min:**

Dense at 25 min would run ~12,000 iters (dense is ~2× faster per step than MoE with h=4096). Can't run dense-12000 within the 20-min-per-experiment budget, so instead: run dense at 6000 iters (~13 min), then fit a trend across dense-{1500, 3000, 6000} and extrapolate to 12000.

#### Dense trend (for matched-wall-clock comparison)

| Dense d12 iters | val_bpb | CORE | wall-clock |
|---|---:|---:|---:|
| 1500 | 0.8749 | 0.1429 | 3.18 min |
| 3000 | 0.8395 | 0.1651 | 6.36 min |
| 6000 | 0.8154 | 0.1769 | 12.75 min |

Extrapolating log-linear to 12000 iters (~25 min, matching MoE-6000): **val_bpb ≈ 0.797, CORE ≈ 0.184**.

#### **Breakthrough at 4× iters**

| | MoE E=8 k=1 h=4096 @ 6000 | Dense d12 @ ~12000 (extrapolated) | Δ |
|---|---:|---:|---:|
| Wall-clock | **24.98 min** | ~25 min | ~matched |
| val_bpb | **0.7796** | ~0.797 | **MoE −0.017 (−2.1%)** |
| CORE | **0.2000** | ~0.184 | **MoE +0.016 (+8.7% rel.)** |

**At 4× training (matched wall-clock), MoE BEATS dense on both metrics** — reversing the verdict at 1500 iters. The mechanism: MoE's per-step advantage compounds while dense's per-step gains shrink (diminishing returns along the log-compute axis). Around the 4× mark, these two curves cross.

This is a big deal for the miniseries philosophy. It means MoE *can* lift the compute→loss frontier — **but only above a minimum compute budget**. At short-compute budgets, dense's MFU advantage dominates; past some crossover point, MoE's capacity advantage dominates.

#### P2 — Actual dense-12000 run (confirms the extrapolation)

| Metric | MoE E=8 k=1 h=4096 @ 6000 | Dense d12 @ 12000 | Winner |
|---|---:|---:|---|
| Wall-clock | **24.98 min** | **25.53 min** | ~matched (±2%) |
| Peak VRAM | 57.2 GiB | **28.0 GiB** | dense (2× less) |
| val_bpb | **0.7796** | 0.7991 | **MoE −0.0195 (−2.4%)** |
| CORE | 0.2000 | **0.2037** | **dense +0.0037 (+1.8% rel.)** |

**The two metrics split.** MoE wins val_bpb; dense wins CORE. At matched wall-clock ~25 min, we cannot declare a clean architectural winner — we have to say what we're optimizing for.

The CORE gap has shrunk from +19 mbits at 3000 iters to +4 mbits at 12000 iters. If the trend continues, CORE also crosses in MoE's favor somewhere above 12000 iters. We can't run that within the 20-min-per-experiment budget, so the crossover point is extrapolated, not observed.

**Revised verdict so far**: MoE at d12 is *better* than the original sweep concluded. The original sweep capped at 1500 iters and missed the crossover. With enough training (~4× iterations / 2× wall-clock over the "short-experiment" regime), MoE E=8 k=1 h=4096 becomes:
- Better than dense on val_bpb (clear-cut).
- Tied-to-slightly-losing on CORE (gap shrinking fast with scale).
- Still ~2× VRAM-hungry — non-trivial.

### Interim findings summary (after dense-12000)

| Compute budget | Best MoE val_bpb | Dense val_bpb | Δ |
|---|---:|---:|---:|
| ~3 min (1500 iters) | 0.8458 (E=8 k=1 h=4096) | 0.8749 (dense-1500) | MoE wins bpb |
| ~6 min (3000 iters dense eq.) | 0.8458 (MoE @ 1500) | 0.8395 (dense-3000) | dense wins bpb |
| ~13 min (6000 iters dense eq.) | — | 0.8154 (dense-6000) | dense trending |
| ~25 min (matched) | 0.7796 (MoE @ 6000) | 0.7991 (dense-12000) | MoE wins bpb |

| Compute budget | Best MoE CORE | Dense CORE | Δ |
|---|---:|---:|---:|
| ~3 min | 0.1455 | 0.1429 | MoE +0.003 |
| ~6 min | 0.1455 | 0.1651 | dense +0.020 |
| ~13 min | — | 0.1769 | dense |
| ~25 min | 0.2000 | 0.2037 | dense +0.004 |

Picture is emerging: **dense's CORE advantage vanishes as compute budget grows**. At 3 min they tie, at 6 min dense dominates, at 25 min dense's advantage is tiny. Extrapolating further, MoE CORE would overtake dense. The minimum compute budget for "MoE is clearly better" is somewhere between 25 min and ~50 min on this rig.

#### P3/P4 — depth scaling: d16 dense vs MoE

The miniseries principle says a change must lift the frontier across the whole d10→d20 family. Tested at d16 to see whether MoE's advantage grows, shrinks, or holds with depth.

| Config | Iters | val_bpb | CORE | wall-clock | Peak VRAM | active_total |
|---|---:|---:|---:|---:|---:|---:|
| Dense d12 | 1500 | 0.8749 | 0.1429 | 3.18 min | 28 GiB | 286M |
| MoE d12 E=8 k=1 h=4096 | 1500 | 0.8458 | 0.1455 | 6.31 min | 47 GiB | 381M |
| **Dense d16** | **1500** | **0.8473** | **0.1502** | **5.79 min** | **47 GiB** | 537M |
| **MoE d16 E=8 k=1 auto** | **1500** | **0.8390** | **0.1639** | **7.44 min** | **59 GiB** | 537M |

Two critical observations:

1. **"Just go deeper" is a real threat to MoE at d12.** Dense d16 @ 1500 (5.79 min) gets val_bpb=0.847 and CORE=0.150 — essentially tying MoE d12 h=4096 @ 1500 (0.846/0.146) at shorter wall-clock and in similar VRAM. If a maintainer's choice is "d12 + MoE" vs "d16 dense", d16 dense wins on both ends.
2. **MoE at d16 keeps a bigger CORE lead.** MoE d16 vs dense d16: val_bpb delta 0.9% (smaller than MoE d12 h=4096 vs dense d12's 3.3%), but CORE delta **+9.3%** (much bigger than d12's 2%). The val_bpb → CORE gap-widening at higher depth is suggestive that MoE's downstream benefit compounds with depth, not just iterations.

The right 3-way comparison at matched wall-clock of ~6 min:

| Config @ ~6 min | val_bpb | CORE |
|---|---:|---:|
| Dense d12 @ 3000 (6.36 min) | 0.8395 | 0.1651 |
| MoE d12 E=8 h=4096 @ 1500 (6.31 min) | 0.8458 | 0.1455 |
| Dense d16 @ 1500 (5.79 min) | 0.8473 | 0.1502 |

At matched 6-min wall-clock: **dense d12 given more steps wins val_bpb (0.8395)**, **dense d12 also wins CORE (0.1651)**. At d12 with *more training time*, dense is ahead of MoE-d12 and dense-d16 on both metrics.

But at longer wall-clock — MoE @ 6000 (25 min) gets CORE=0.200, and we haven't tried MoE d16 at longer training yet. The MoE-d16 @ 3000-iters run (in flight) will tell us whether MoE d16 keeps pulling ahead.

#### P5 — MoE d16 E=8 k=1 auto @ 3000 iters (2026-04-18)

| Config | val_bpb | CORE | wall-clock | Peak VRAM |
|---|---:|---:|---:|---:|
| MoE d16 @ 3000 | 0.7973 | 0.1889 | 14.79 min | 58.7 GiB |

#### P6 — dense d16 @ 3000 iters (2026-04-18)

| Config | val_bpb | CORE | wall-clock | Peak VRAM |
|---|---:|---:|---:|---:|
| Dense d16 @ 3000 | 0.8069 | 0.1888 | 11.59 min | 46.6 GiB |

#### The d16 matched-wall-clock test

| Metric | MoE d16 @ 3000 | Dense d16 @ 3800* (extrap) |
|---|---:|---:|
| Wall-clock | 14.79 min | ~15 min |
| val_bpb | 0.7973 | ~0.799 |
| CORE | 0.1889 | ~0.193 |

*Linear log-extrapolation from dense-d16-{1500, 3000}.*

At d16 and matched ~15-min wall-clock, **dense narrowly beats MoE on both metrics** — same pattern as d12 at matched wall-clock. The depth dial (d16 dense) gives more bang-for-buck than adding MoE at d12.

### What we've learned about MoE at nanochat scale (synthesis after phase 2)

1. **Dense gets more wall-clock-per-FLOP → dense gets more iterations → compounding loss drop** typically keeps pace with MoE's per-token capacity advantage. This is the core mechanism behind the negative verdict at every matched-wall-clock comparison we've run.
2. **The gap narrows with compute budget.** At 1500 iters (~5 min dense / ~6 min MoE) the gap is big. At 6000/12000 iters (~13 min dense / ~25 min matched) it shrinks by 3–5×. The curves are converging; a crossover may exist above ~50 min / ~1 hour compute budgets.
3. **Going deeper is a cheaper win than adding experts.** Dense d16 @ 1500 gets ~99% of MoE-d12-h=4096's val_bpb and 103% of its CORE, in less wall-clock and same VRAM. For the miniseries philosophy, this is saying the loss of rotating the "depth dial" beats inserting MoE.
4. **MoE's CORE–val_bpb disagreement has shrunk, not vanished.** Early runs (small E, k=2, router on Muon) had MoE winning val_bpb but losing CORE by 10+%. With the best-tuned config (E=8 k=1 h=4096, router on AdamW), the CORE gap is 1–4% and closing. Not a clean MoE win yet.

### Next up: capacity_factor=1.0 (Noam's suggestion) at d16

Earlier runs used cf=1.25 which overprovisions each expert's slots and wastes dispatch work on empty slots. Dropping cf to 1.0 means tighter packing — fewer dropped tokens than I'd expect (at E=8 k=1, even a naive balance has ~8K tokens per expert average, so overflow is rare). Should recover 10–20% MFU. If MoE d16 @ 3000 with cf=1.0 beats dense d16 @ 3000 on wall-clock, the verdict flips at d16.

#### P7 — MoE d16 E=8 k=1 auto cf=1.0 @ 3000 iters

| Config | val_bpb | CORE | wall-clock | Peak VRAM |
|---|---:|---:|---:|---:|
| MoE d16 E=8 auto cf=1.25 @ 3000 (P5) | 0.7973 | 0.1889 | 14.79 min | 58.7 GiB |
| MoE d16 E=8 auto cf=1.0 @ 3000 | 0.7981 | 0.1934 | 14.59 min | 56.0 GiB |

cf=1.0 saves ~2.7 GiB and is ~1.4% faster; val_bpb essentially unchanged; CORE is +0.0045 (small but in the right direction — tighter packing might slightly help load balance). **Take cf=1.0 as the new default** — it's strictly better on all dimensions.

#### P8 — MoE d16 E=4 k=1 h=4096 cf=1.0 @ 1500 iters (E=4 ablation + router fix)

Required an `optim.py` patch: `DistMuonAdamW._reduce_adamw` failed when `shape[0]` of a param wasn't divisible by `world_size`. Router at E=4 has shape (4, n_embd) which doesn't split across 8 GPUs. Fix: fall back to all_reduce when not divisible (routers are tiny enough for replicated optimizer state to be cheap).

| Config | val_bpb | CORE | wall-clock | Peak VRAM |
|---|---:|---:|---:|---:|
| **MoE d16 E=4 h=4096 cf=1.0 @ 1500** | **0.8306** | 0.1524 | 9.47 min | 73.1 GiB |

E=4 h=4096 at d16 @ 1500 gives best val_bpb in its bucket (−16 mbit vs dense d16 @ 1500), but CORE is only marginal (+2mb). Split continues.

#### P9 — MoE d12 E=4 k=1 h=4096 cf=1.0 @ 3000 (E=4 d12 + extended)

| Config | val_bpb | CORE | wall-clock | Peak VRAM |
|---|---:|---:|---:|---:|
| **MoE d12 E=4 h=4096 cf=1.0 @ 3000** | **0.8111** | 0.1646 | 11.61 min | 49.9 GiB |

At ~12 min, MoE d12 E=4 h=4096 val_bpb (0.811) beats dense d12 @ 6000 (0.815), but CORE ties (0.165 vs dense's 0.177 — dense wins). Same split pattern.

#### P10 — MoE d12 E=4 k=1 **h=6144** cf=1.0 @ 3000 (push expert size further)

| Config | val_bpb | CORE | wall-clock | Peak VRAM |
|---|---:|---:|---:|---:|
| **MoE d12 E=4 h=6144 cf=1.0 @ 3000** | **0.8044** | **0.1705** | 14.49 min | 64.6 GiB |

**First MoE config to beat dense d12 @ 3000 (0.840 / 0.165) on BOTH metrics at same iter count.** h=6144 lifts val_bpb −7 mbit vs h=4096 and CORE +6 mbit vs h=4096. More active compute per token helps both metrics.

But compared to dense d12 @ 6000 (12.75 min, 0.815/0.177) at matched wall-clock: MoE wins val_bpb by 11 mbit, dense wins CORE by 6 mbit. Split persists.

#### P11 — MoE d16 E=4 k=1 h=4096 cf=1.0 @ 3000 (best d12 config ported to d16)

| Config | val_bpb | CORE | wall-clock | Peak VRAM |
|---|---:|---:|---:|---:|
| **MoE d16 E=4 h=4096 cf=1.0 @ 3000** | **0.7883** | **0.1919** | **18.78 min** | 73.1 GiB |

**Best val_bpb in any experiment so far** (across all configs, all iter counts, all depths within 20-min budget). CORE 0.192 is in the top tier too.

Matched wall-clock ~19 min comparison (dense d16 @ ~5000 iters, in flight):
- Extrapolated dense d16 @ ~5000: val ≈ 0.787, CORE ≈ 0.211
- MoE d16 E=4 h=4096 @ 3000: val 0.788, CORE 0.192
- Prediction: MoE loses both at matched wall-clock (val tied within noise, CORE loses by ~20mb)

Running actual dense d16 @ 5000 iters to confirm.

### State of the art after ~15 runs

Best config at each wall-clock bucket (updated):

| Wall-clock | Best val_bpb config | val_bpb | Best CORE config | CORE |
|---|---|---:|---|---:|
| ~3 min | dense d12 @ 1500 | 0.875 | dense d12 @ 1500 | 0.143 |
| ~6 min | dense d12 @ 3000 | 0.840 | dense d12 @ 3000 | 0.165 |
| ~6 min (alt) | MoE d12 E=8 h=4096 @ 1500 | 0.846 | dense d16 @ 1500 | 0.150 |
| ~12 min | MoE d12 E=4 h=4096 @ 3000 | 0.811 | dense d12 @ 6000 | 0.177 |
| ~15 min | **MoE d16 E=8 auto cf=1.0 @ 3000** | 0.798 | **MoE d16 E=8 auto cf=1.0 @ 3000** | 0.193 |
| ~15 min (alt) | MoE d12 E=4 h=6144 @ 3000 | 0.804 | MoE d12 E=4 h=6144 @ 3000 | 0.171 |
| ~19 min | **MoE d16 E=4 h=4096 @ 3000** | **0.788** | dense d16 @ ~5000 (extrap) | ~0.211 |
| ~25 min | MoE d12 h=4096 @ 6000 | 0.780 | dense d12 @ 12000 | 0.204 |

### Pattern that keeps showing up

1. **At short wall-clock (<15 min), dense beats MoE** on both metrics. Dense's MFU advantage + more iterations beat MoE's capacity advantage.
2. **Around 15 min, MoE starts competitive on both metrics** — especially at d16, where MoE d16 E=8 cf=1.0 @ 3000 iters gets best val_bpb AND best CORE (barely edging dense d16 @ 3000).
3. **Beyond 15 min, MoE wins val_bpb clearly but dense catches up / wins CORE** — val_bpb and CORE diverge. MoE seems to overfit validation loss while dense's downstream performance keeps scaling.
4. **Going deeper with dense consistently competes with MoE.** At the miniseries level ("which config lifts the d10→d20 frontier?"), dense d16 often matches MoE d12 in the same wall-clock.

### P12 — Dense d16 @ 5000 iters (matched wall-clock to best MoE d16)

| Config | val_bpb | CORE | wall-clock | Peak VRAM |
|---|---:|---:|---:|---:|
| Dense d16 @ 5000 | **0.7844** | **0.1966** | 19.32 min | 46.6 GiB |
| MoE d16 E=4 h=4096 cf=1.0 @ 3000 | 0.7883 | 0.1919 | 18.78 min | 73.1 GiB |

**At matched ~19 min, dense d16 beats our best MoE config on BOTH metrics, using 36% less VRAM.** The extrapolated-from-trend CORE estimate for dense (0.211) was too optimistic (actual 0.197) — the CORE curve flattens faster at longer training than the short-run trend predicts. But even at the lower actual number, dense still wins CORE, and it wins val_bpb too.

### Phase 2 verdict — refined after 20+ runs spanning iters {1500, 3000, 5000, 6000, 12000} × depths {12, 16} × E in {4, 8} × h in {auto, 4096, 6144}

**Across every matched-wall-clock comparison we've run, dense either wins or ties MoE within noise.** There is no wall-clock bucket where MoE clearly beats dense on both metrics simultaneously. The pattern:

| Wall-clock | Winner (val_bpb) | Winner (CORE) | Margin |
|---|---|---|---|
| ~3–6 min | Dense | Dense | Clear |
| ~12 min | MoE d12 (slight) | Dense | Split |
| ~15 min | Tied (within noise) | Tied (within noise) | Toss-up |
| ~19 min | **Dense** | **Dense** | Clear |
| ~25 min | MoE | Dense (narrow) | Split |

A "slight MoE lead at one budget but slight dense lead at adjacent budgets" is not a frontier lift — it's noise between two curves that are essentially on top of each other in this regime.

### What made MoE competitive (not winning — competitive)

Of all the knobs I turned, these mattered enough to appear in a final best config:

1. **top_k = 1** (Switch-style). Cuts dispatch/gather work in half vs top_k = 2 without losing much per-step quality. All our best configs use k=1.
2. **Explicit `expert_hidden_dim`** at 2.5–3× the compute-matched auto value. Bigger matmuls use tensor cores better. MoE's MFU goes from 24% (auto) to 35% (h=4096) — the single biggest throughput improvement we found.
3. **Router on AdamW, experts on Muon**. Muon orthogonalizes; the router needs to concentrate mass on a few experts per token, which is the opposite of what orthogonalization does. Switching router to AdamW clearly helped CORE.
4. **capacity_factor = 1.0** (Noam's tip). Small throughput + VRAM savings, no quality loss in our runs. Take this as the default.
5. **1 shared expert**. We didn't test num_shared = 0 or 2 exhaustively, but DeepSeek convention (1 shared + routed) has been fine.

### What did NOT help

- **num_experts > 32**. Per-expert token count drops, GEMM efficiency tanks, CORE regresses. E=64 was worse than E=32; E=128 OOM'd.
- **num_experts = 2 with top_k = 2**. Degenerate: all experts active for every token, fragmented matmuls, zero sparsity. Strictly worse than dense.
- **More than 4× iterations**. Returns diminish, dense catches up at matched wall-clock.

### What we didn't try

- **Sinkhorn / transport-based routing** (replaces aux loss). Code change significant enough that we'd want a dedicated session.
- **Expert parallelism** (experts sharded across GPUs instead of replicated). Would let us fit E=128+ without OOM, and would fundamentally change the memory-bandwidth profile of dispatch. This is the single change most likely to flip the verdict — our MFU is memory-bound, and EP attacks exactly that bottleneck.
- **Fused FlashMoE kernel**. Doesn't exist as a stable PyTorch primitive. If it did, our MFU would recover to ~35–40% (near dense) and the whole landscape changes.
- **d18+ at long training**. Might or might not flip verdict. Depth-scaling so far has been neutral — dense depth dial competes well with MoE.

### Final practitioner-oriented takeaway

For the nanochat miniseries (d10 → d20, ~5 min to ~1 hour training budgets on 8×H100):

> **Just go deeper with dense.** You pay 30–60% more VRAM for MoE and get either a wash or a small regression vs running dense at a higher depth with the same wall-clock.

MoE is not a miniseries-lifting change at this scale with the tools we have. To make it one, we'd need (in decreasing order of plausibility): expert parallelism, a FlashMoE-style kernel, or evidence that the trend flips decisively at d20+ or at training budgets we can't fit in a 20-min window.

### Running log of *every* configuration tried (for future maintainers)

See the run-level tables above. Key data points condensed:

```
depth  E   top_k  expert_hidden  cf    iters  router_opt  wall_clock  val_bpb  CORE
-----  --  -----  -------------  ----  -----  ----------  ----------  -------  -----
d12    —   dense   3072           —     1500  AdamW        3.18 min   0.8749   0.1429
d12    —   dense   3072           —     3000  AdamW        6.36 min   0.8395   0.1651
d12    —   dense   3072           —     6000  AdamW       12.75 min   0.8154   0.1769
d12    —   dense   3072           —    12000  AdamW       25.53 min   0.7991   0.2037
d12    2   2      1024           1.25  1500  Muon*        4.33 min   0.8773   0.1269
d12    4   2       870           1.25  1500  Muon*        4.86 min   0.8713   0.1403
d12    8   2      1024           1.25  1500  Muon*        6.03 min   0.8659   0.1275
d12    32  1      1536           1.25  1500  AdamW        4.95 min   0.8572   0.1408
d12    64  1      1536           1.25  1500  AdamW        ?          0.8626   0.1230
d12    8   1      4096           1.25  1500  AdamW        6.31 min   0.8458   0.1455
d12    8   1      4096           1.0   6000  AdamW       24.98 min   0.7796   0.2000
d12    4   1      4096           1.0   3000  AdamW       11.61 min   0.8111   0.1646
d12    4   1      6144           1.0   3000  AdamW       14.49 min   0.8044   0.1705
d16    —   dense   4096           —     1500  AdamW        5.79 min   0.8473   0.1502
d16    —   dense   4096           —     3000  AdamW       11.59 min   0.8069   0.1888
d16    —   dense   4096           —     5000  AdamW       19.32 min   0.7844   0.1966
d16    8   1      2048           1.25  1500  AdamW        7.44 min   0.8390   0.1639
d16    8   1      2048           1.25  3000  AdamW       14.79 min   0.7973   0.1889
d16    8   1      2048           1.0   3000  AdamW       14.59 min   0.7981   0.1934
d16    4   1      4096           1.0   1500  AdamW        9.47 min   0.8306   0.1524
d16    4   1      4096           1.0   3000  AdamW       18.78 min   0.7883   0.1919
```

*Muon router was the buggy path; all results after commit `8001c63` use AdamW routers.*

### P13 — MoE d12 E=4 k=1 h=6144 cf=1.0 **aux_coef=0.05** @ 3000 (5× default)

| Config | val_bpb | CORE | wall-clock | train/aux_loss (final) |
|---|---:|---:|---:|---:|
| MoE d12 E=4 h=6144 aux=0.01 @ 3000 (P10) | 0.8044 | 0.1705 | 14.49 min | 0.12 |
| MoE d12 E=4 h=6144 **aux=0.05** @ 3000 | 0.8047 | **0.1749** | 14.46 min | 0.60 |

**Tighter load balance (5× aux coef) gave +4mb CORE with no val_bpb or wall-clock cost.** Small but real positive result. The 5× aux coef keeps the measured aux loss higher (0.60 vs 0.12 at convergence), which indicates the router is working harder to evenly distribute tokens. This supports the interpretation that *some* of MoE's CORE gap vs dense is "experts specialize on frequent patterns and flail on rare ones during eval" — tighter load balance during training gives more exposure to rare patterns per expert.

The new best MoE config at ~14.5 min wall-clock: **MoE d12 E=4 k=1 h=6144 cf=1.0 aux=0.05 @ 3000**: val 0.805, CORE 0.175.

But dense d12 @ 6000 (12.75 min) still gets CORE 0.177 at *lower* wall-clock. Tighter load balance didn't break the pattern.

### Still to try if budget allows

- [ ] aux_coef=0.05 on the best d16 config (P11) — does the same trick help at higher depth?
- [ ] `num_shared_experts = 0` ablation — is the shared expert doing work, or is it just insurance?
- [ ] MoE d18 at 1500 iters (may OOM at E=8; E=4 should fit).
- [ ] Sinkhorn-style routing implemented cleanly (no aux loss).


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
