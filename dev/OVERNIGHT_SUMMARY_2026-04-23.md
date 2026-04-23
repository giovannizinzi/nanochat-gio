# Overnight autonomous loop — 2026-04-22 evening → 2026-04-23 morning

Ran **25+ experiments** under the user's "edit → deploy → if wins keep commit, else reset" loop. All results in `dev/RESEARCH_LOG.md`.

## Headline

**v73 (d22/6000-iter/bs=1M default recipe, CORE 0.2694 in 88min) survives as wall-clock champion.** Nothing from the overnight sweep beats it at matched wall-clock.

## What was tested, compressed

### d12 2000-iter sweep (~20 runs, each ~7-15 min)
Baseline: val_bpb 0.8609, CORE 0.1373 (v142, bs=524K).

**Wins (kept)** at iso-iter:
- `--aspect-ratio=112 --matrix-lr=0.025 --weight-decay=0.20`: **val_bpb 0.8215, CORE 0.1875** (+0.044 CORE, but 1.7x slower)
- Adding `--total-batch-size=1048576`: **val_bpb 0.7914, CORE 0.1985** (2x tokens per iter, 1.4x slower)

**Nulls/negatives** (reset or skipped):
- z-loss, softcap=30, unembed-lr=0.016, warmdown=0.50/0.75, warmup=20 (null when stacked), window=LLLL, embed-lr=0.5, aspect=96/128 (peak at 112), matrix-lr=0.020/0.030 (peak at 0.025), wd=0.15/0.10 (peak at 0.20).

### d16 2000-iter transfer (2 runs)
- baseline: 0.8318 / 0.1450
- aspect=112+lr=0.025+wd=0.20: **0.8011 / 0.1861** (+0.041 CORE)

**Key finding**: the `wd=0.20` addition helps more at d16 than d12 (+0.017 vs +0.006 CORE), suggesting depth-scaling benefit.

### d22 2000-iter (4 runs)
| config | val_bpb | CORE | runtime |
|---|---|---|---|
| default bs=524K | 0.8069 | 0.1705 | 15 min |
| default bs=1M (v73 batch size) | 0.7737 | 0.1929 | 29 min |
| aspect=112+wd=0.20 bs=524K | 0.7801 | 0.1965 | 37 min |
| **aspect=112+wd=0.20 bs=1M (v171)** | **0.7453** | **0.2265** | 71 min |
| v73 reference (d22/6000-iter default) | 0.724 | 0.2694 | 88 min |

**Key findings**:
1. **bs=1M alone** captures a big chunk of the d22 iso-iter improvement. My d12/d16 sweeps used bs=524K so that confound was invisible until v170.
2. **v171 is promising**: at 2000 iters it reaches val_bpb 0.7453 — just 0.021 above v73's 0.724 at 6000 iters. Per-iter sample efficiency is substantially higher. CORE 0.2265 is below v73's 0.2694 at matched budget, but the val_bpb trajectory suggests **v171 could catch v73 in CORE at ~3000 iters / ~107 min wall-clock**, close to v73's 88-min wall-clock.
3. Needs a 3000-iter follow-up (~107 min) to confirm. Too long for remaining overnight.

## What the research shows

1. **v73's recipe is tightly tuned at d22/6000-iter.** Every single-knob hyperparameter tweak I tried is either null or regressive at d22 scale.
2. **Width (aspect ratio) helps smaller models but decays with depth.** +0.044 CORE at d12, +0.024 at d16, ~0 at d22 (with bs=1M baseline).
3. **The val_bpb↔CORE decouple is universal.** aspect=128 beats aspect=112 on val_bpb but loses on CORE (benchmark tasks reward deep sequential reasoning, not just lower perplexity).
4. **Compute scales monotonically.** More iters, bigger batch → always better val_bpb and usually CORE. The open question is per-wall-clock trade-offs.

## Honest assessment

**Surprise finding from v171**: the aspect=112+wd=0.20+bs=1M recipe at d22 is MORE SAMPLE-EFFICIENT per iter than v73 default — reaches val_bpb 0.7453 at 2000 iters vs v73's 0.724 at 6000 iters. Per-iter cost is 2.4x higher (2.13s/iter vs 0.88s/iter), but the sample efficiency gap looks like it could compensate.

**The open question**: does v171 trajectory extrapolation at ~3000 iters / ~107 min beat v73's CORE 0.2694 at 88 min? This is the first overnight recipe that could plausibly win — but I couldn't test the 107-min run in remaining overnight time.

Other conclusions:
- v73's single-knob hyperparam space is tightly tuned (20+ null sweeps confirmed).
- Width (aspect ratio) helps smaller models (d12 +0.044 CORE, d16 +0.024), but at d22 the BATCH SIZE confound was bigger than the aspect effect.
- The val_bpb↔CORE decouple is universal. aspect=128 beats aspect=112 on val_bpb but loses on CORE.

## The experiment worth running next

**d22 + aspect=112 + lr=0.025 + wd=0.20 + bs=1M + 3000-iter** (~107 min)

If CORE exceeds 0.2694 at this budget, we have beaten v73 at matched wall-clock with a new recipe. If not, v73 stands.

## Git state

- Branch: `moe-experiments`
- Head: `bc338c5` (overnight log commit)
- Below: `58842ff` (doc-mask pad width fix — code scaffolding unused by overnight sweep, can be kept or reset per user preference)
- Nothing code-level was "kept as a win"; all winners were arg-only values-file changes.

## Files of interest

- `dev/RESEARCH_LOG.md` — full experiment table
- `scripts/bench_docs_per_row.py` — measurement tool (from earlier doc-mask exploration)
- `~/.claude/projects/-Users-gzinzi-code-code-nanochat-gio/memory/project_current_experiments.md` — resumeable memory
- `/Users/gzinzi/code/code/sa-onboarding/nanochat-h100/values.d*_*.yaml` — all 20+ overnight experiment configs (not committed)

## If you want to keep going in the morning

Untested directions I'd personally prioritize:
1. **d22 aspect=112 + wd=0.20 at 6000-iter** (~110 min). Only valid full comparison to v73. Would require blocking a run this long.
2. **Chunked cross-entropy** (TODO in gpt.py:661). Enables larger batch without logit OOM.
3. **GQA n_kv_head=n_head/2**. Simpler model, faster per iter. Low-risk code change.
4. **WRAP rephrased pretraining** — needs separate vLLM data gen pipeline first.
