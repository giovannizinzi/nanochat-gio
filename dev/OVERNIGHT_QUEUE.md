# Overnight experiment queue (2026-04-19 → 2026-04-20)

Goal: narrow the 13-min time-to-CORE-0.2565 gap between MoE and dense d26 FP8, or prove it closed. Per user's "validate-then-scale" rule, each experiment is 2000 iters (~45 min) for hypothesis validation. Only the winner gets scaled to 6000 iters for the CORE race.

## Baseline for 2000-iter hypothesis validation

v58 (MoE d22 E=4 K=1 D_e=4096 shared=1 cf=1.0 aux=0.05 FP8 grouped) at step 2000 = **val_bpb 0.817**.

Any v59+ variant should be compared against this number at step 2000.

## Queue

| # | Hypothesis | Config delta from v58 | Expected effect | Paper ref |
|---|---|---|---|---|
| v59 | 1dense+SE beats all-MoE at same active params | `--moe-first-layer=1 --num-experts=8 --expert-hidden-dim=2816` (r_a≈22%, paper sweet spot) | Small quality gain + minor per-step save | 2506.12119 §4.2, Table 6 |
| v60 | Shared expert is load, not help | same as v58 but `--num-shared-experts=0 --expert-hidden-dim=5632` (match dense d22 FFN) | Lower step time, possibly worse quality | own audit |
| v61 | Fine-grained K=1 beats chunky K=1 | `--num-experts=32 --expert-hidden-dim=768 --num-shared-experts=1 --moe-first-layer=1` | Mixed; kernel efficiency risk on small matmuls | 2506.12119 §4.3, Table 8 |
| v62 | Higher depth pays with 1dense+SE savings | `--depth=24 --moe-first-layer=1 --num-experts=4 --expert-hidden-dim=4096` | d24 is deeper than d22, 1 layer dense offsets the dispatch tax | combine |
| v63 | Aux loss hurts k=1 quality | same as v58 but `--moe-aux-loss-coef=0.0` | Routing imbalance but maybe better loss | own |
| v64 | r_a≈12% is too sparse | `--num-experts=16 --expert-hidden-dim=2048 --num-shared-experts=1 --moe-first-layer=1` | Test paper's claim that <15% underperforms | 2506.12119 |
| v65 | Depth vs width: d18 wider | `--depth=18 --num-experts=4 --expert-hidden-dim=6144 --num-shared-experts=1 --moe-first-layer=1` | Same active params compressed in fewer layers | own |

After each run: log val_bpb at step 2000 + final loss + MFU + per-step dt to this file.

## Results

| # | Config | step 2000 val_bpb | Δ vs v58 (0.817) | MFU | Notes |
|---|---|---|---|---|---|
| v58 (baseline) | d22 E=4 K=1 Dₑ=4096 sh=1 cf=1.0 aux=0.05 | 0.817 | — | 44% | final @ 6000: val_bpb 0.712, **CORE 0.2568** at ~141 min. Barely beats GPT-2 0.2565; loses to dense d26 FP8 (0.2846 at 99 min) on matched wall-clock. |
| v59 | d22 E=8 K=1 Dₑ=2816 sh=1 **first_layer=1** aux=0.05 FP8 grouped | **0.766** | **−0.051** | 40% | **WINNER.** dt=1243 ms (v58: 1420 ms, 14% faster). CORE@2000=0.206 (quick eval). Scaling to 6000 iters as v60. |
| v60 (scale-up) | same as v59, 6000 iters, full CORE eval | final 0.716 | — | 40% | **LOSER.** CORE 0.2495 @ 124 min (below GPT-2 0.2565, below v58's 0.2568). v59's 2000-iter lead was LR-schedule artifact. 1dense+SE + r_a=22% + E=8 + D_e=2816 underperforms v58's E=4 D_e=4096 on absolute CORE despite being 18 min faster. |
| v61 | d24 E=4 K=1 Dₑ=4096 sh=1 first_layer=1 | CRASH OOM | — | — | d24 replicated MoE doesn't fit; would need EP. Skipped. |
| v62 | d22 E=4 K=1 Dₑ=4096 sh=1 **first_layer=1** aux=0.05 (= v58 + 1dense+SE only) | running | | | Isolates 1dense+SE alone vs v58's 0.817 baseline. |

## Winner protocol

When a 2000-iter run shows val_bpb notably below 0.817 (say ≥ 0.005 improvement), run it for 6000 iters with `CORE_MAX_PER_TASK=-1` to get the CORE number vs dense d26 FP8's 0.2846 (at 99 min).
