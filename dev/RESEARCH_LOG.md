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
| v115 | 22 | 6000 | dense FP8 +MuonClip tau=100 scale-up | 0.724 | 0.2485 | 880 | 57% | **REGRESSION.** Small 1500-iter win flipped to big 6000-iter loss. MuonClip hurts long training. |

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
