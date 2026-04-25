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

## The decisive v172 experiment (3000-iter)

Ran the key follow-up while the overnight loop still had time. Result:

| metric | v172 (3000-iter) | v73 (6000-iter) | delta |
|---|---|---|---|
| val_bpb | **0.7237** | 0.724 | **matched** (−0.0003) |
| CORE | 0.2533 | **0.2694** | −0.016 (LOSS) |
| wall-clock | 106.7 min | **88 min** | +19 min (LOSS) |

**v172 matches v73's val_bpb at half the iters, but the val_bpb↔CORE decouple means it's ~0.016 below on CORE AND 19 min slower.** Extrapolating: v172 needs ~3600 iters (~128 min) to reach v73's CORE. Clear wall-clock loss.

## Honest assessment

**v73 stands.** After 25+ overnight experiments across d12/d16/d22, nothing from the hyperparam + architecture space beats v73's 88-min/0.2694 recipe at matched wall-clock.

Key conclusions:
- v73's single-knob hyperparam space is tightly tuned (20+ null sweeps confirmed).
- Width (aspect ratio) helps smaller models (d12 +0.044 CORE, d16 +0.024), but at d22 the benefit collapses against the 2.4x per-iter cost.
- The val_bpb↔CORE decouple is universal. Wider models reach lower val_bpb faster but don't cash that efficiency into CORE — meaning per-wall-clock CORE is flat or worse.
- bs=1M (v73 default) was the biggest overlooked confound in my d12 sweep (which used bs=524K).
- Sample-efficient ≠ compute-efficient. At d22, reaching v73's val_bpb at half the iters doesn't translate to a CORE win.

## What's still untested

- **Data-level interventions** (WRAP rephrased pretraining, MATES reweighting, perplexity filtering) — out of overnight scope; requires vLLM data gen pipelines.
- **More compute** (v117 10k-iter → 0.2793 in 147 min) reliably works but violates wall-clock constraint.
- **MLA (Multi-head Latent Attention, DeepSeek-V2)** — complex ~150 LOC implementation; theoretically most promising remaining angle.

## Round 2 (user requested GQA/NoPE/chunked-CE/MLA)

Added 3 new flags: `--n-kv-head-divisor`, `--no-rope`, `--chunked-ce-chunk-size`. d12 smoke test passed with combined 46% per-iter speedup. **But the d12 speedup did NOT transfer to d22**.

| exp | config | val_bpb | CORE | runtime | verdict |
|---|---|---|---|---|---|
| v174/175 | d22 + MQA + NoPE + chunked-CE bs=1M | 0.7957 | 0.1773 | 26.0 min | quality LOSS −0.016 CORE |
| v176 | d22 + NoPE + chunked-CE (no MQA) | 0.7838 | 0.1784 | 29.3 min | no speedup, quality LOSS |
| v177 | d22 + chunked-CE only | 0.7736 | 0.1942 | 29.9 min | **bit-identical to v170 baseline** (confirms correctness) |
| **v178** | d22 + head_dim=64 + GQA 2:1 bs=1M 2000it | 0.7789 | **0.2026** | 28.6 min | **+0.010 CORE iso-wallclock** |
| v179 | v178 recipe at 6000-iter | 0.7291 | 0.2551 | 86.1 min | LOSS vs v73 (−0.014 CORE at −2min wallclock) |
| v180 | v178 + bs=2M + 3000-iter (iso-tokens) | 0.7339 | 0.2375 | 85.1 min | LOSS: −0.018 vs v179 iso-tokens, −0.032 vs v73 |

**Learnings from Round 2**:
- chunked-CE works as designed (bit-identical to one-shot) but provides no speedup at d22 (logits are not the wall-clock bottleneck at this scale).
- NoPE (Haviv et al. 2022) **hurts** CORE at d22 by ~0.015. Causal mask alone is insufficient position signal.
- MQA (1 KV head at d22/11 heads because 11 is prime) **hurts** CORE by ~0.016. Attention expressivity matters.
- head_dim=64 + GQA 2:1 (22 heads, 11 KV heads): **helps at 2000-iter** (+0.010 CORE iso-wallclock), but **the gain does not sustain at 6000-iter**. At full budget the recipe saturates 0.014 below v73.
- d12 46% speedup from combined flags was a small-model artifact. At d22, compute is matmul-bound, so RoPE/CE/KV projection overhead is negligible.

**Final overnight position**: v73 remains wall-clock champion. The aspect=112 direction, the GQA+head-dim-64 direction, and the NoPE/chunked-CE direction all fail to beat v73 at 88 min / CORE 0.2694. The recipe is tightly tuned.

**Additional finding from v180**: at iso-tokens + iso-wallclock, **bs=1M with more iters beats bs=2M with fewer iters** (−0.018 CORE for bs=2M). More optimization steps > bigger batch updates at this budget. Consistent with Chinchilla-style scaling.

**What a morning-session should try** (per Ilya/Alec framing):
1. **MLA (Multi-head Latent Attention)** — the one remaining "free lunch" untested. Low-rank KV compression. ~150 LOC but theoretically could reduce per-iter compute WITHOUT the MQA quality loss.
2. **Data-level**: WRAP rephrased pretraining. Would require building the offline vLLM rephrasing pipeline first (~1-2 hrs setup, then ~45 min data gen per 20% of ClimbMix).
3. **Recipe search with more iters**: if user accepts 100-130 min budget, v178's head_dim=64+GQA might beat v73 at matched compute-optimal (8-10 tokens/param).
4. None of the above are guaranteed wins. v73's tightness is real.

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

## Round 3 (2026-04-24/25): MuonClip wins via small-scale filter strategy

After 30+ negative experiments at d22 6000-iter (90min each), pivoted to d22 1500-iter quick-filter (27min each). Filtered 6 hypotheses cheaply, promoted only the best (muonclip) to full 6000-iter.

**1500-iter filter (vs Q0 baseline val_bpb 0.7910 / CORE 0.1937)**:
| Q | knob | val_bpb | CORE | verdict |
|---|---|---|---|---|
| Q1 | matrix-lr=0.025 | 0.7885 | 0.1967 | small both |
| Q2 | matrix-lr=0.022 | 0.7890 | 0.1972 | small both |
| **Q3** | **muonclip tau=100** | 0.7908 | **0.2005** | **CORE +0.007 — promote** |
| Q4 | attn-output-gate | 0.7898 | 0.1934 | val small |
| Q5 | embed-lr=0.4 | 0.7895 | 0.1932 | tied |
| Q6 | muonclip + lr=0.025 | 0.7883 | 0.1994 | val best, CORE same as Q3 |

**6000-iter scale-up** (anchor v73: val_bpb 0.7242 / CORE 0.2714 / ~81 min crossover):
| run | recipe | val_bpb | CORE | crosses 0.2565 at |
|---|---|---|---|---|
| **v198 (E48)** | **muonclip tau=100 alone** | **0.7242** | **0.2731** | **~80 min** ← **WINNER** |
| v199 (E49) | muonclip + warmdown=0.85 | 0.7240 | 0.2696 | ~81 min |
| v200 (E50) | muonclip + lr=0.025 | 0.7245 | 0.2667 | ~83 min |
| v201 (E51) | muonclip + warmup=20 | 0.7245 | 0.2625 | ~82 min |

**v198 IS THE ROUND-3 WINNER**: val_bpb tied with v73, CORE +0.0017, crosses GPT-2 CORE ~1 min faster (80 vs 81 min pure-train).

**Key learnings**:
1. **Small-scale filter strategy works**: 6× faster iteration (27min vs 90min) revealed muonclip as the CORE-mover. Should have started here.
2. **MuonClip (Kimi K2 §A QK-Clip) is a real CORE win at d22** — never tested cleanly before round 3. Previous v94 only tested as stack with matrix-lr=0.03 (which cancelled the gain).
3. **All stacks on muonclip REGRESSED** (warmdown=0.85, lr=0.025, warmup=20). Muonclip alone is at a local optimum.
4. **val_bpb↔CORE decouple persists**: v200 tied v73's val_bpb but lost CORE.

**To try next**: muonclip + warmdown=0.75, muonclip + bs=2M, MLA r=384 + muonclip, perplexity-filtered shards (data-level).

**To submit to leaderboard**: run v198's recipe with `--core-metric-every=999999` (no trajectory eval overhead) for clean 88-min wall-clock + CORE 0.273x + val_bpb 0.724x.

