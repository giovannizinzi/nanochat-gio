# Overnight research log — lowest val_bpb sweep

Started: 2026-04-20 evening. Plan: `/Users/gzinzi/.claude/plans/crystalline-soaring-aho.md`.

**Baselines at d12 1500 iter FP8**:
- dense d12 FP8 (v64): val_bpb **0.878**, CORE **0.1268** ← target to beat
- MoE d12 E=4 K=1 sh=1 (v65): val_bpb 0.871, CORE 0.1399

**Baselines at d22 6000 iter FP8** (for eventual scale-up):
- dense d22 FP8 (v73): val_bpb 0.724, CORE **0.2694** (88 min) ← target CORE
- MoE d22 sh=3 (v75): val_bpb 0.714, CORE 0.2651 (142 min)

## Scoreboard (rolling)

| v | depth | iter | config delta | val_bpb | CORE | dt (ms) | MFU | verdict |
|---|---|---|---|---|---|---|---|---|
| v64 | 12 | 1500 | dense FP8 baseline | 0.878 | 0.1268 | ~400 | — | baseline |
| v65 | 12 | 1500 | MoE E=4 K=1 sh=1 FP8 | 0.871 | 0.1399 | — | — | MoE wins d12 |
| v91 | 22 | 1500 | MoE E=16 K=2 sh=1 Dₑ=2048 bf16 +grad-ckpt | 0.785 | 0.1892 | 1820 | 29% | grad-ckpt validated; CORE below dense d22 |
| v92 | 12 | 1500 | dense FP8 +MuonClip tau=100 | 0.878 | **0.1329** | — | — | **+0.006 CORE**, val_bpb identical |
| v93 | 12 | 1500 | dense FP8 +Muon LR 0.03 (vs 0.02 default) | 0.876 | **0.1440** | — | — | **+0.0172 CORE win, biggest yet** |
| v94 | 12 | 1500 | dense FP8 +MuonClip tau=100 +Muon LR 0.03 | 0.876 | 0.1366 | — | — | Stack HURT — MuonClip cancels LR gain. Use either, not both. |
| v95 | 12 | 1500 | dense FP8 +Muon LR 0.05 | 0.878 | 0.1371 | — | — | LR 0.05 past peak |
| v96 | 12 | 1500 | dense FP8 +warmup-steps=200 | 0.878 | 0.1343 | — | — | +0.0075 CORE, small |
| v97 | 12 | 1500 | dense FP8 +warmdown-ratio=0.85 | 0.879 | 0.1293 | — | — | +0.0025, marginal |
| v98 | 12 | 1500 | dense FP8 +weight-decay=0.10 | 0.882 | 0.1335 | — | — | +0.0067, small |
| v99 | 12 | 1500 | dense FP8 +weight-decay=0.50 | 0.877 | **0.1397** | — | — | **+0.0129 win, 2nd biggest** |
| v100 | 12 | 1500 | dense FP8 +Muon LR 0.03 +WD 0.50 | 0.877 | 0.1373 | — | — | stack HURT — below either alone |
| v101 | 12 | 1500 | dense FP8 +aspect-ratio=80 (n_embd=960) | **0.857** | 0.1397 | 170 | 48% | **−0.021 val_bpb, same dt as baseline!** |
| v102 | 12 | 1500 | dense FP8 +aspect-ratio=96 (n_embd=1152) | **0.849** | **0.1424** | 203 | 50% | −0.029 val_bpb, 19% slower. Still big win @ matched wall-clock |
| v103 | 12 | 1500 | dense FP8 +aspect-ratio=112 (n_embd=1344) | **0.841** | 0.1459 | 254 | 57% | val_bpb peak. 50% slower. |
| v104 | 12 | 1500 | dense FP8 +aspect=80 +LR 0.03 | 0.856 | 0.1448 | | | small stack gain |
| v105 | 12 | 1500 | dense FP8 +aspect=96 +LR 0.03 | 0.850 | **0.1548** | | | best CORE until v106 |
| v106 | 12 | 1500 | dense FP8 +aspect=112 +LR 0.03 | **0.839** | **0.1607** | | | **NEW RECORD both metrics** |
| v107 | 22 | 6000 | dense FP8 +aspect=80 +LR 0.03 scale-up | 0.711 | 0.2649 | 1227 | 63% | **LOSE.** −0.013 val_bpb but CORE −0.005. val_bpb↔CORE decouple again. 121 min total. |
| v108 | 22 | aborted | dense FP8 +LR 0.03 only (no aspect) | — | — | | | killed at step 1500 (lagging 0.008 val_bpb vs v73) |
| v109 | 22 | aborted | dense FP8 +LR 0.015 | — | — | | | killed (worse than baseline) |
| v111 | 22 | 1500 | dense FP8 +MuonClip tau=100 | 0.791 | 0.1996 | | | +0.003 CORE at 1500-iter — only d22 transfer |
| v112 | 12 | 1500 | dense FP8 +SwiGLU | 0.878 | 0.1320 | | | +0.005 CORE at d12 |
| v113 | 12 | 1500 | dense FP8 +z-loss=1e-4 | 0.883 | 0.1297 | | | marginal |
| v114 | 22 | 1500 | dense FP8 +SwiGLU | 0.792 | 0.1949 | | | SwiGLU does NOT transfer to d22 |
| v115 | 22 | 6000 | dense FP8 +MuonClip tau=100 scale-up | 0.724 | 0.2485 | 880 | 57% | **REGRESSION.** Small 1500-iter win flipped to big 6000-iter loss. |
| v116 | 22 | 6000 | dense FP8 +warmdown-ratio=0.85 | 0.724 | 0.2588 | 880 | 57% | **REGRESSION.** Tied val_bpb, CORE −0.011. Schedule shape tuned at 0.65 default. |
| v117 | 22 | **10000** | dense FP8 baseline, longer training | **0.709** | **0.2793** | 880 | 57% | **WIN.** Just training 67% longer → −0.015 val_bpb, +0.010 CORE. Baseline was under-saturated at 6000 iter. 147 min total. |

## Best-ever d22 configs (updated)

| config | iters | wall-clock | val_bpb | CORE |
|---|---|---:|---:|---:|
| **v117 dense d22 FP8 (10k iter)** | 10000 | **147 min** | **0.709** | **0.2793** ← new king |
| v73 dense d22 FP8 (baseline) | 6000 | 88 min | 0.724 | 0.2694 |
| v75 MoE d22 sh=3 | 6000 | 142 min | 0.714 | 0.2651 |

At wall-clock-matched 88 min, dense d22 FP8 6000-iter (v73) is still the winner. Beyond
88 min, more training compounds: +0.010 CORE from +67% more tokens. Nothing else in our
sweep (hyperparameter, architecture, optimizer, MoE) gave ≥+0.005 CORE at d22. The path
to better val_bpb/CORE at this model size is **more compute**, not cleverer recipe.

## Leaderboard-aware analysis (after v119)

**The leaderboard is about time-to-CORE-0.2565 (crossing), not final CORE.** Current
leaderboard entry #6: 1.65 h (99 min), CORE 0.2626 (Karpathy autoresearch round 2).

### v119: instrumented CORE trajectory (baseline recipe, --core-metric-every=500, quick eval 500/task)

| step | val_bpb | CORE (quick) |
|---:|---:|---:|
| 500 | 0.914 | 0.117 |
| 1000 | 0.856 | 0.165 |
| 1500 | 0.834 | 0.186 |
| 2000 | 0.821 | 0.191 |
| 2500 | 0.804 | 0.186 |
| 3000 | 0.788 | 0.193 |
| 3500 | 0.774 | 0.219 |
| 4000 | 0.761 | 0.218 |
| 4500 | 0.750 | 0.236 |
| 5000 | 0.739 | 0.246 |
| 5500 | 0.731 | 0.254 |
| 6000 | 0.724 | 0.260 |

**Key calibration**: v73 at step 6000 (full eval) got CORE 0.2694; v119 quick eval got
0.260. Quick-eval underestimates full-eval by ~0.009. So the adjusted trajectory:

| step | CORE (full-eval adj) |
|---:|---:|
| 5000 | ~0.255 |
| 5250 | **~0.260 — crosses GPT-2 0.2565** |
| 5500 | ~0.263 |
| 6000 | 0.269 |

Crossing at step ~5250 = **77 min of pure training** (v73 ran 88 min for full 6000 iter).

### Time-to-GPT-2 estimates vs leaderboard

| recipe | final CORE | final wall-clock | est. crossing wall-clock |
|---|---:|---:|---:|
| Leaderboard #6 (Karpathy autoresearch 2) | 0.2626 | 99 min | ~99 min |
| **v73 / v119 baseline (6000 iter)** | **0.2694** | **88 min** | **~77-82 min** ← beats #6 by ~17-22 min |
| v117 baseline (10000 iter) | 0.2793 | 147 min | ~70-75 min (but eval only at end) |
| v118 aspect=80+LR=0.03 (10000 iter) | 0.2853 | 202 min | unknown |

### v120: test if compressed schedule crosses earlier (running)

Hypothesis: running baseline at 5000 iter (shorter LR schedule → more warmdown% late)
might cross 0.2565 at a wall-clock earlier than the 6000-iter recipe.

If v120 final CORE ≥ 0.2565 with wall-clock ≪ 77 min, we have a new time-to-GPT-2
record. If final CORE < 0.2565, the schedule-compression hypothesis fails.

Expected wall-clock for 5000 iter: 5000 × 0.88s = **73 min**. If final CORE ≥ 0.2565, we
set a tight budget floor and can explore further.

## Speedrun sweep results (overnight 2026-04-21)

Tested several candidates aimed at reducing time-to-CORE-0.2565 below v73's 82 min:

| v | config | iters | val_bpb | CORE | wall-clock | crosses? |
|---|---|---:|---:|---:|---:|:---:|
| v73 | baseline | 6000 | 0.724 | **0.2694** | **88 min** | ✅ ~82 min |
| v117 | baseline | 10000 | 0.709 | 0.2793 | 147 min | ✅ |
| v118 | aspect=80+LR=0.03 | 10000 | 0.697 | 0.2853 | 202 min | ✅ |
| v119 | baseline + CORE trajectory | 6000 | 0.724 | 0.2604 (quick) | 88+overhead | ✅ ~82 min |
| v120 | compressed schedule | 5000 | 0.731 | 0.2470 | 73 min | **❌ by 0.010** |
| v121 | mid compression | 5500 | 0.727 | 0.2539 | 81 min | **❌ by 0.003** |
| v122 | d20 smaller model | 6000 | 0.734 | hung | ~71 min | (probably ❌) |
| v123 | 2M batch × 3000 iter | 3000 | 0.729 | 0.2435 | 88 min | **❌ by 0.013** |
| v124 | device_batch=32 | — | — | OOM | — | — |
| v125 | Muon LR 0.018 | 6000 | 0.724 | 0.2651 | 88 min | ✅ but CORE −0.004 vs v73 |

**Findings for the speedrun leaderboard:**

1. **Baseline recipe (v73) crosses CORE 0.2565 at ~82 min of pure training** (inferred
   from v119 trajectory where step 5500 gave CORE ~0.254 and step 6000 gave ~0.269).

2. **Cutting iters fails**: 5000 iter (0.247) and 5500 iter (0.254) both miss the target
   despite reaching similar val_bpb to baseline. More gradient steps matter for CORE
   beyond what val_bpb captures.

3. **Larger batch fails**: 2M batch × 3000 iter has equivalent val_bpb (0.729) to baseline
   but CORE 0.2435 — final CORE correlates with #gradient-steps, not just total tokens.

4. **Smaller model fails**: d20 consistently +0.012 val_bpb worse than d22 per step.
   Model scale matters.

5. **LR fine-tune (0.018 vs 0.020) is a wash**: CORE −0.004 at same val_bpb.

6. **Speedrun recipe is optimal within single-recipe hyperparameter space**: v73's 6000
   iterations, 1M batch, Muon LR 0.02, default schedule is at a local optimum for
   time-to-CORE. The leaderboard-beating path requires either (a) a qualitatively
   different method (optimizer beyond Muon, novel architecture, better data) or (b)
   more compute (v117/v118 show CORE continues improving past 6000 iter).

## Final leaderboard comparison

| entry | time | val_bpb | CORE | notes |
|---|---:|---:|---:|---|
| Leaderboard #6 (Karpathy autoresearch 2) | 99 min (1.65h) | 0.71800 | 0.2626 | Mar 14 2026 |
| **v73 baseline (our estimate)** | **~82 min pure training** | 0.724 | 0.2694 | our recipe = ~speedrun.sh |
| v117 | 147 min | 0.709 | 0.2793 | more iters, off-budget |
| v118 | 202 min | 0.697 | 0.2853 | all-time best CORE |

v73 effectively matches leaderboard #6 once tokenizer + eval overhead is added back in
(~15 min). **Nothing in our sweep produces a strict leaderboard improvement** at the
pure-training budget.

## Research directions that remain open

All of these require code work not done in this sweep:
- **MuonClip QK-Clip** coupled with higher LR (our v115 regressed MuonClip alone; maybe
  MuonClip + LR 0.025 unlocks a better Pareto point we didn't test)
- **Longer context (seq_len=4096)** — untested; might hurt throughput but boost quality
- **Different data**: leaderboard shows ClimbMix was a win (-12 min); are there better
  sources or per-domain weighting schemes to try?
- **RoPE tuning**: base frequency, theta variants
- **Better expert init** for MoE (if revisited)
- **Expert parallel MoE with working CORE eval** (v91 showed grad-checkpoint unlocks
  configs but CORE was hurt)

None of these were implementable in this window without significant code work; logged
as concrete follow-ups.

## Additional attempts (post-summary) — nothing moves the needle

| v | config | val_bpb | CORE | vs baseline |
|---|---|---:|---:|---:|
| v126 | MuonClip + Muon LR 0.025 | (killed) | — | trailing +0.003 at step 2000, killed early |
| v127 | warmdown-ratio=0.75 | 0.724 | 0.2509 | **−0.019 CORE** |
| v128 | warmup-steps=80 | 0.724 | 0.2583 | **−0.011 CORE**, still crosses |
| v129 | ffn-type=swiglu | 0.726 | 0.2525 | **−0.017 CORE, FAILS to cross target**. SwiGLU's 8/3·D hidden is less effective than relu²'s 4·D. |

Baseline warmdown=0.65 is the right point. Going to 0.75 or 0.85 (v116, −0.011)
both regress CORE despite identical val_bpb. Same pattern as all other hyperparameter
changes — val_bpb tracks baseline, CORE drops.

Fully out of ideas in the hyperparameter space. **Baseline recipe (Muon LR 0.02,
warmdown 0.65, warmup 40, weight-decay 0.28, device batch 16, total batch 1M,
6000 iter, d22, FP8) is the speedrun champion at 82-88 min.**

## Final scoreboard at d22 6000-iter matched wall-clock

| config | wall-clock | val_bpb | CORE |
|---|---:|---:|---:|
| **v73 dense d22 FP8 (baseline)** | **88 min** | 0.724 | **0.2694** ← WINNER |
| v75 MoE d22 E=4 K=1 sh=3 Dₑ=2048 | 142 min | 0.714 | 0.2651 |
| v107 dense +aspect=80 +LR 0.03 | 121 min | 0.711 | 0.2649 |
| v115 dense +MuonClip tau=100 | 88 min | 0.724 | 0.2485 |

**No configuration beats the dense d22 FP8 baseline at matched wall-clock.**

Key takeaways from the overnight research:
1. **Aspect ratio is the biggest d12 lever** (up to +0.034 CORE) but it's just "bigger model fits free at low scale" — at d22 it costs wall-clock without gaining CORE.
2. **Muon LR 0.03** is a real d12 win (+0.017 CORE) but doesn't transfer: d22 needs LR 0.02 (default) which is already tuned.
3. **MuonClip** shows small transfer at d22 1500-iter (+0.003) but REGRESSES at 6000-iter (−0.021). Not worth using.
4. **SwiGLU and z-loss** barely move the needle at any scale.
5. **Hyperparameter stacking (LR 0.03 + WD 0.50) hurts** at d12; architecture+hyperparameter stacks (aspect + LR) give small boost.
6. **val_bpb ↔ CORE decouple is real**: wider dense model (v107) gets better val_bpb but worse CORE. Same pattern observed with MoE.
7. **The default nanochat dense d22 FP8 recipe is tightly tuned**. Genuine wins require compute-asymmetric levers (longer training, more data, different scale) that don't fit in a 88-min budget.

## Phase 1 queue — d12 single-knob sweeps

Each runs dense d12 FP8 for 1500 iter with ONE flag change vs v64 baseline.

| # | WANDB_RUN | knob | value |
|---|---|---|---|
| P1.1 | `rs_d12_muonlr03` | `--matrix-lr=0.03` | Muon LR higher |
| P1.2 | `rs_d12_muonlr05` | `--matrix-lr=0.05` | Muon LR extreme |
| P1.3 | `rs_d12_warmup200` | `--warmup-steps=200` | longer warmup |
| P1.4 | `rs_d12_warmdown085` | `--warmdown-ratio=0.85` | more warmdown |
| P1.5 | `rs_d12_wd010` | `--weight-decay=0.10` | less WD |
| P1.6 | `rs_d12_wd050` | `--weight-decay=0.50` | more WD |
| P1.7 | `rs_d12_aspect80` | `--aspect-ratio=80` | wider (n_embd 960) |
| P1.8 | `rs_d12_aspect48` | `--aspect-ratio=48` | narrower |
| P1.9 | `rs_d12_seq4096` | `--max-seq-len=4096` | 2× context |
| P1.10 | `rs_d12_gqa` | (needs head count change) | halve n_kv_head |
| P1.11 | `rs_d12_bs2m` | `--total-batch-size=2097152` | bigger batch |
| P1.12 | `rs_d12_winL` | `--window-pattern=LLLL` | all full context |
| P1.13 | `rs_d12_winS` | `--window-pattern=SSSS` | all sliding |
| P1.14 | `rs_d12_unembedlr` | `--unembedding-lr=0.004` | lower lm_head LR |
| P1.15 | `rs_d12_embedlr` | `--embedding-lr=0.5` | higher embed LR |

Scale to d16 anything that hits val_bpb < 0.868.

## Phase 2/3 — filled after Phase 1 results

---

## 2026-04-22 — data-level experiments, baseline recipe recap

### My v73 baseline vs Karpathy's `runs/speedrun.sh`

Both use: Muon matrix opt + AdamW embeddings, FP8 (tensorwise), `--device-batch-size=16`, total_batch_size=524288 (default), seq_len=2048 (default), ClimbMix-400B dataset.

**Differences:**
| | Karpathy `runs/speedrun.sh` | My v73 baseline |
|---|---|---|
| depth | **24** | 22 |
| iteration schedule | `--target-param-data-ratio=8` → auto ~7800 iter | explicit `--num-iterations=6000` |
| tokens seen | ~4.1 B (8× scaling_params) | ~3.1 B |
| runtime target | ~3 h (advertised speedrun) | 88 min (my measurement) |

Both are compute-optimal-ish around 8-10 tokens/param (vs Chinchilla 20). The d24+ratio=8 recipe trains slightly **undertrained** (Karpathy comment on line 72: "slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 to 8"). This is deliberate: fewer but bigger-model steps get to CORE 0.2565 fastest per-wall-clock.

My v73 d22 version happened to hit the same CORE target faster (88 min → 0.2694). Unclear until I run v135 whether this is because d22 is a better depth for 8×H100 throughput, or because my smaller step count got lucky.

### v135 launched: d24 + ratio=8 (Karpathy's exact recipe)

Config: `values.d24_karpathy_speedrun.yaml`. Iterations auto-computed from target-param-data-ratio=8 (EXPERIMENT_ITERATIONS=0 falls through to ratio path in `scripts/base_train.py:381-392`). CORE_MAX_PER_TASK=-1 (full eval).

Hypothesis: d24 will beat v73 d22 on CORE because it's Karpathy's tuned optimum for the speedrun. The question is whether the extra depth costs more than 88 min of wall-clock.

### Research agent findings — Jan-Apr 2026 data-level techniques

Top recommendations (ranked by expected CORE gain × low implementation cost):

**Tier 1 (ship tonight):**
1. **Rephrased pretraining (WRAP)** — Allen-Zhu arxiv 2401.16380 + Qwen3 Mar 2026 report. Replace 20-30% ClimbMix with Qwen-rephrased Wiki/StackExchange in Q&A+textbook styles. **+1.8 CORE at d12/1.5B tokens, 3× data-efficiency**. Zero training LOC; offline shard gen on 1 H100 ~45 min via vLLM. Highest single-bet ROI.
2. **MATES / DataComp-LM reweighting** — arxiv 2402.09739 + Kimi-K2 Feb 2026. Train 60M proxy on held-out val loss; reweight ClimbMix shards every 500 steps via influence proxy. +0.8-1.2 CORE. ~150 LOC. Orthogonal to ClimbMix (within-mixture vs between-mixture).
3. **BFD packing + cross-doc attention masking** — Krell et al. + DeepSeek-V3.5 Feb 2026. Replace naive concat-and-chunk; use `torch.nn.attention.flex_attention` document block mask. +0.4 CORE, +3% throughput. ~80 LOC. Fully orthogonal.

**Tier 2 (2 hours):**
4. Perplexity-correlations filtering (arxiv 2409.05816 + MiniMax-Text-01) — +0.5-0.9 CORE, 40% data reduction. Overlaps MATES.
5. Short-seq curriculum (Doubao-1.5 Lite Mar 2026) — first 30% at seq=1024, then 2048. +0.3 CORE, 8% faster. **10 LOC**. Free.
6. JEST joint example selection (DeepMind + Doubao Feb 2026, arxiv 2406.17711) — select batches by learnability score from reference model. +0.6 CORE, 13× less compute-per-CORE. Reference model fwd pass adds ~8% overhead.

**Skip for 1-hour sprint:** RegMix (too expensive), FineWeb-Edu v2 (subsumed), Phi-4 synthetic continuations (only shows at >3B).

**Plan after v135 finishes:**
1. Short-seq curriculum (10 LOC) — cheapest test, confirms agent's prediction before larger bets
2. BFD + doc masking (80 LOC) — orthogonal, needs flex_attention refactor
3. Rephrased pretraining — only if above land something; GPU-time heavy

### v135 result: d24 Karpathy speedrun recipe — LOSES to v73 d22

| metric | v135 d24 ratio=8 | v73 d22 6000iter (mine) | delta |
|---|---|---|---|
| val_bpb | **0.7189** | 0.724 | −0.005 (win) |
| CORE | 0.2611 | **0.2694** | −0.0083 (LOSE) |
| wall-clock | 99 min | 88 min | +11 min (LOSE) |

**Headline**: My v73 d22 explicit 6000-iter recipe beats Karpathy's canonical d24 speedrun recipe on CORE at matched wall-clock. The val_bpb↔CORE decouple strikes again — bigger d24 model wins val_bpb by 0.005 but loses CORE by 0.008.

**Final trajectory** (v135 d24):
| step | val_bpb |
|---|---|
| 250 | 1.026 |
| 500 | 0.910 |
| 1000 | 0.847 |
| 2000 | 0.809 |
| 3000 | 0.773 |
| 4000 | 0.746 |
| 5000 | 0.726 |
| 5568 | **0.719** |

Implication: The d22 vs d24 choice is CORE-dependent, not val_bpb-dependent. Future experiments should continue at d22 (where my baseline lives) rather than chase d24 for marginal val_bpb.

### v136 result: d22 seq=1024 at 6000 iter — big LOSS

Hypothesis: d12 seq=1024 gained +0.018 CORE at small scale; maybe the shorter seq is just better data-efficiency and it transfers to d22.

Result: **CORE 0.2434** at 98 min (5899s) — val_bpb 0.7319.

| metric | v136 d22 seq=1024 | v73 d22 seq=2048 (champ) | delta |
|---|---|---|---|
| val_bpb | 0.7319 | 0.724 | +0.008 (LOSE) |
| CORE | **0.2434** | **0.2694** | **−0.026 (big LOSE)** |
| wall-clock | 98 min | 88 min | +10 min (LOSE) |

Config delta from v73: `--max-seq-len=1024 --device-batch-size=32` (total_batch=1M held constant).

**Finding**: The d12→d22 transfer assumption fails for seq length. Shorter seq at d22 costs ~0.026 CORE and runs slower (memory-bound batch larger). Consistent with v107-era finding that wider/deeper benefits scale non-linearly with seq. At d22 the full 2048 context matters for CORE tasks (HellaSwag, ARC, etc. reward long-context reasoning).

**Conclusion**: stop chasing d12 wins at d22. The recipe space around v73 is truly saturated on single-knob axes.

### 2026-04-22 evening — v137 killed; pivot to principled correctness fix

Initial plan was v137 = `--max-train-shards=3` (multi-epoch on subset, paper 2506.12119 §6). **Killed before meaningful training**: this is a regime-specific trick that trades generalization for CORE-at-6000-iter. Multi-epoch on a narrower slice is textbook overfitting — it inflates a specific-compute metric at the cost of validation generalization. Not a recipe improvement worth pursuing.

Pivot: **cross-doc attention masking** as a correctness fix. The current BOS-bestfit dataloader packs multiple docs per row; the causal mask has no notion of doc boundaries; Doc-N tokens attend to unrelated Doc-(N-1) context. That's a train/test mismatch (at inference the model never sees packed cross-doc context) and a form of training-signal noise.

### 2026-04-22 — Docs-per-row measurement (v138 BENCH_MODE)

Before committing ~80 LOC to the FA3 varlen refactor, measured effect-size ceiling. `scripts/bench_docs_per_row.py` replays the BOS-bestfit packing on real ClimbMix shard 0, T=2048, 4096 rows.

| metric | value |
|---|---|
| docs_per_row | mean=3.94, p50=3, p90=6, p99=11, max=13 |
| bos_per_row | mean=3.94 (matches — each packed doc gets exactly one BOS) |
| crop_rate | **1.53%** tokens discarded (not 35% per the old docstring) |
| doc_length_tokens | mean=528, p50=538, p90=835, p99=1693 |

**Findings**:
1. **Cross-doc contamination is real** — median row packs 3 docs, 20% of rows pack 5+. Every training step, most tokens attend across unrelated docs. Inference never does this → explicit train/test mismatch.
2. **Token-recovery intervention is dead** — crop rate is 1.53%, not the 35% the docstring warned about. Best-fit + buffer=1000 is already near-optimal; nothing to recover.
3. **Expected magnitude of doc-mask fix**: moderate. Papers report +0.3-0.5 CORE from BFD+mask combined, but they bundle packing gains. Since our packing is already efficient, we only capture the mask piece → realistic expectation +0.1 to +0.3 CORE.

**Next**: implement `--doc-mask` via FA3 `flash_attn_varlen_func` + cu_seqlens. Validate at d12 2000-iter before scaling to d22.

