# Overnight writeup — speedrun attempt, 2026-04-21/22

## TL;DR

**Nothing beats the baseline speedrun recipe at its wall-clock budget.**

- **Baseline (v73) crosses GPT-2 CORE 0.2565 at ~82 min of pure training.** Beats current
  leaderboard #6 (99 min / 0.2626) on paper, but likely matches it once tokenizer +
  final-eval overhead is added back.
- Every hyperparameter deviation from baseline regresses CORE (often while matching
  val_bpb — the two metrics decouple).
- More compute is the only reliable lever: v117 (10k iter, 147 min) → CORE 0.2793,
  v118 (10k iter + aspect=80 + LR=0.03, 202 min) → CORE 0.2853 all-time best.
- None of the **Chinese OSS 2025-26 tricks** (MuonClip, SwiGLU, Scaled Embed,
  routed_scaling, data reuse, expert-choice routing, ScatterMoE) give a clean d22
  win. All were implemented but either neutral or negative.
- **MoE** best config (v75, E=4 K=1 sh=3 Dₑ=2048): CORE 0.2651 at 142 min. Still
  loses to dense d22 at matched wall-clock.

## The speedrun race

Leaderboard (as of user's last info):

| # | time | val_bpb | CORE | contributor |
|---|---:|---:|---:|---|
| 0 | 168 h | — | 0.2565 | Original GPT-2 |
| 6 | 1.65 h (99 min) | 0.71800 | 0.2626 | Karpathy autoresearch round 2 |

Our best measured d22 runs:

| run | time (pure training) | val_bpb | CORE | crosses GPT-2? |
|---|---:|---:|---:|:---:|
| **v73 baseline** | **~88 min** | 0.724 | **0.2694** | ✅ at est. ~82 min |
| v117 (10k iter) | 147 min | 0.709 | 0.2793 | ✅ |
| v118 (10k iter, aspect=80, LR=0.03) | 202 min | 0.697 | **0.2853** | ✅ best CORE |

v73's **82-min crossing** estimate comes from the v119 instrumented trajectory (CORE
eval every 500 steps), which showed CORE crossed 0.2565 somewhere between step 5500
(CORE 0.254) and step 6000 (CORE 0.269), linear-interpolated to step ~5583 ≈ 82 min.

## What we tried to speed up baseline crossing

All failed:

| experiment | change from baseline | CORE | vs v73 | notes |
|---|---|---:|---:|---|
| v120 | num-iterations=5000 | 0.2470 | −0.022 | fails to cross |
| v121 | num-iterations=5500 | 0.2539 | −0.016 | fails by 0.003 |
| v122 | depth=20 | hung | — | eval deadlock (infra) |
| v123 | total-batch=2M @ 3000 iter | 0.2435 | −0.026 | same tokens, half iters → fails |
| v124 | device-batch-size=32 | OOM | — | |
| v125 | matrix-lr=0.018 | 0.2651 | −0.004 | val_bpb matches, CORE drops |
| v126 | MuonClip + matrix-lr=0.025 | (killed) | — | trailing at step 2000 |
| v127 | warmdown-ratio=0.75 | 0.2509 | −0.019 | val_bpb matches, CORE drops |
| v116 (earlier) | warmdown-ratio=0.85 | 0.2588 | −0.011 | same pattern |
| v115 (earlier) | MuonClip tau=100 | 0.2485 | −0.021 | clipping hurts late training |
| v114 (earlier) | ffn-type=swiglu | 0.1949 → 1500-iter | — | tied/worse |
| v107 (earlier) | aspect=80 + matrix-lr=0.03 | 0.2649 | −0.005 | 121 min wall-clock |

## Big lesson: val_bpb ↔ CORE decouple

Multiple experiments land at identical val_bpb 0.724 as the baseline but measurably
lower CORE. This is a real and reproducible pattern:

- v125 (LR 0.018): val_bpb 0.724, CORE 0.2651 (−0.004)
- v127 (warmdown 0.75): val_bpb 0.724, CORE 0.2509 (−0.019)
- v115 (MuonClip): val_bpb 0.724, CORE 0.2485 (−0.021)
- v123 (2M batch): val_bpb 0.729, CORE 0.2435 (−0.026)

**val_bpb is not a sufficient proxy for CORE at this scale.** Fewer gradient steps and
alternative optimizer dynamics can produce similar next-token perplexity but materially
worse downstream capability.

Same decouple goes the other way for MoE:

- MoE d22 sh=1 (v58): val_bpb **0.712**, CORE 0.2568 (worse CORE despite lower val_bpb)
- MoE d22 sh=3 (v75): val_bpb **0.714**, CORE 0.2651
- dense d22 baseline: val_bpb 0.724, CORE 0.2694

MoE compresses bits better but trails on CORE.

## What's been added to the codebase (all behind default-off flags)

Preserved on `moe-experiments` branch, commits ca7838e through b819ede:

- `--muon-qk-clip-tau` (Kimi K2 QK-Clip, arxiv 2507.20534)
- `--moe-grad-checkpoint` (activation checkpointing for MoE blocks)
- `--ffn-type=swiglu` (Shazeer 2020 GLU variant)
- `--z-loss-coef` (PaLM-style z-loss)
- `--embed-norm` (toggle for Scaled Embed / Spike-No-More — the LN after wte was already default)
- `--moe-expert-choice` (Zhou 2022 routing)
- `--moe-auxfree-bias` (DeepSeek-V3 load balancing)
- `--moe-routed-scaling` (DeepSeek-V3 gate amplification)
- `--moe-scattermoe` (vendored ScatterMoE Triton kernels)
- `--moe-first-layer` (1dense+SE paper layout)
- `--max-train-shards` (data-reuse / multi-epoch)
- `nanochat/kernels/scattermoe/` (vendored from github.com/shawntan/scattermoe, Apache-2.0)
- `nanochat/flash_moe.py` + `scripts/bench_flash_moe.py` (Triton MoE kernel prototype, 12× slower than bmm)

All smoke-tested at build time on CPU; all default off so baseline bit-identical.

## Recommendations for next session

What's left to try, roughly in descending expected value:

1. **Data** — leaderboard history shows ClimbMix was −12 min. Is there a better data
   source/weighting? Per-domain curriculum? Perplexity-filtered subset?
2. **Attention changes** — MLA (DeepSeek), diff-attn, longer context with sliding windows.
   Requires FA3-compatible implementation for fair speed comparison.
3. **Better optimizer at scale** — Fantastic-Optimizers (arxiv 2509.02046) says Muon is
   optimal below ~4× Chinchilla, SOAP/Kron win above. We're likely below 4× at 6 B
   tokens.
4. **RLHF-style post-training** — not nanochat's training-only metric target.
5. **Architecture: Cut Cross-Entropy (CCE, apple/ml-cross-entropy)** — saves VRAM on
   unembedding, could unlock bigger batches. Worth 4-8h of wiring.

## Branches, artifacts, reproducibility

- `moe-experiments` (cluster pulls this): baseline recipe + all code-level additions
- `gradient-checkpoint-moe`: fork with just the grad-ckpt change
- `scaled-embed`: fork with embed_norm toggle
- `swiglu-zloss`: fork with SwiGLU + z-loss
- `muonclip`: fork with QK-clip

All configs stored as `sa-onboarding/nanochat-h100/values.*.yaml` (one file per
experiment). Each experiment's wandb run is `rs_*`. Research log at
`dev/RESEARCH_LOG.md`.
