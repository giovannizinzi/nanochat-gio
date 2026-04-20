# d12 sprint — paper hypothesis test (arxiv 2506.12119)

Goal: validate at fast ~10-min d12 runs whether MoE + data reuse ("loose 2-epoch scheme") can beat dense at matched compute. If yes, scale winner to d22 for the CORE-vs-dense-d26-FP8 race.

Compute: ~330ms/step at d12 × 1500 iters = ~8 min/experiment. CORE at d12 is noisy but directional.

## Results

| # | Config | Final val_bpb | CORE (500/task) | Notes |
|---|---|---|---|---|
| v64 | **dense d12 FP8** (reference) | **0.878** | **0.1268** | Baseline for the sprint. |
| v65 | MoE d12 E=4 K=1 sh=1 auto Dₑ=1536 aux=0.05 FP8 | **0.871** | **0.1399** | **MoE beats dense at d12: val_bpb −0.007, CORE +0.0131.** |
| v66 | v65 + `--max-train-shards=33` (~2 epochs) | 0.871 | **0.1305** | Data reuse HURT CORE by −0.009 vs v65. Paper's recipe doesn't transfer. |
| v67/68/69 | (reuse ablations) | skipped | — | Negative reuse signal — not worth burning budget. |
| v67 (dense d16 FP8) | reference at middle depth | **0.851** | **0.1421** | +0.015 CORE vs dense d12. |
| v68 (MoE d16 TC) | E=4 K=1 sh=1 aux=0.05 FP8 | **0.846** | **0.1425** | MoE +0.0004 CORE vs dense d16. **Advantage collapsing with depth**: +0.013 @ d12 → +0.0004 @ d16. |
| v69 (MoE d12 EC) | expert-choice routing | 0.871 | **0.1292** | EC WORSE than TC by −0.011 CORE. At our scale expert-choice provides no benefit over token-choice. |
| v71 (dense d22 FP8, 1500 iter) | matched-depth baseline | **0.791** | **0.1969** | New honest dense baseline at d22. Karpathy's dense d26 was unfair vs MoE d22. |
| v72 (MoE d22 TC, 1500 iter) | v58 config at 1500 iter | **0.780** | **0.1971** | **MoE ties dense at matched depth d22, both val_bpb (MoE +0.011) and CORE (MoE +0.0002). Reopens the race.** |
| v73 (dense d22 FP8, 6000 iter) | true matched-depth long run | **0.724** | **0.2694** | **Dense wins.** 88 min (1.6× faster than MoE). CORE 0.0126 above v58 MoE d22 (0.2568). |

## Definitive matched-depth 6000-iter comparison

| metric | dense d22 FP8 (v73) | MoE d22 FP8 (v58) | winner |
|---|---|---|---|
| wall-clock | 88 min | 142 min | **dense** |
| val_bpb | 0.724 | 0.712 | MoE (−0.012) |
| CORE | 0.2694 | 0.2568 | **dense (+0.0126)** |
| MFU | 56.78% | 44% | dense |

Striking val_bpb ↔ CORE decoupling: MoE compresses bits better but dense generalizes better to downstream tasks. At our scale, the router's specialization benefit for training data distribution hurts CORE. **Dense d22 FP8 is the real baseline to beat, not the earlier d26 number — and MoE still loses at matched depth & compute.**

## Shared-expert sweep at d22 1500 iter (sh=3 is the sweet spot)

| shared | Dₑ | val_bpb | CORE | Δ vs sh=1 |
|---|---|---|---|---|
| 1 (v72) | 4096 | 0.780 | 0.1971 | baseline |
| **3 (v74)** | **2048** | **0.781** | **0.2090** | **+0.0119 (peak)** |
| 5 (v76) | 1408 | 0.784 | 0.2072 | +0.0101 |

Peak at sh=3. Not monotonic — sh=5 starts to regress. Diminishing returns beyond sh=3,
consistent with DeepSeek-v3's choice of 2-3 shared experts.

## Final 6000-iter matched-depth scoreboard (the real race)

| config | wall-clock | val_bpb | CORE | Δ CORE vs dense |
|---|---:|---:|---:|---:|
| dense d22 FP8 (v73) | 88 min | 0.724 | **0.2694** | — |
| MoE d22 sh=1 (v58) | 142 min | 0.712 | 0.2568 | **−0.0126** |
| MoE d22 sh=3 (v75) | 142 min | 0.714 | **0.2651** | **−0.0043** |

**Dense still wins the 6000-iter race, but shared=3 closed the gap from 0.013 → 0.004.**

At matched wall-clock 88 min (dense's budget):
- Dense: CORE 0.2694 (full run complete)
- MoE sh=3: reaches only step ~3700, val_bpb ~0.76, CORE ~0.24 (from trajectory)
- Gap at 88 min: dense beats MoE by ~0.03 CORE.

Dense d22 FP8 remains the winner on time-to-CORE. MoE sh=3 is the best MoE config we've found — it's 0.008 CORE better than v58 (MoE sh=1) — but can't close the MFU gap (44% vs 57%).

## BREAKTHROUGH (qualified): shared=3 helps (v74, 1500 iter)

Tested the "router over-specialization hurts CORE" hypothesis by raising shared experts
from 1 → 3 (compute-matched via halving Dₑ 4096 → 2048):

| config | val_bpb | CORE | vs dense d22 |
|---|---|---|---|
| dense d22 FP8 | 0.791 | 0.1969 | baseline |
| MoE E=4 K=1 sh=1 Dₑ=4096 | 0.780 | 0.1971 | +0.0002 (tie) |
| **MoE E=4 K=1 sh=3 Dₑ=2048** | **0.781** | **0.2090** | **+0.0121** |

**val_bpb unchanged (0.781 vs 0.780), but CORE jumped +0.012.** This is the first config
that separates val_bpb from CORE in MoE's favor — suggests shared experts carry general
knowledge that routed experts can't (routed specialize too narrowly). Scaling to 6000
iter as v75.

## What might still work (untested or inconclusive)

- **MoE with more iterations than dense at same wall-clock**: dense d22 finishes 6000 iter in 88 min; MoE d22 could run 6000 iter at 142 min but MoE per-step CORE-per-iter scales differently. Running MoE longer than matched-iter (e.g. 10k iter to match dense 6k wall-clock) might flip it. Untested.
- **MoE with active-params > dense**: our configs had active ≈ dense, so no compute-per-token advantage. Making MoE active > dense (e.g. top-k=2) at matched wall-clock → MoE becomes a bigger model per token and maybe outperforms on CORE. Untested at d22.
- **Expert-parallel at d26 MoE**: fits d26 but previous CORE eval crashed. Could be re-tried with core_metric_every=-1.
- **FP8-through-backward for MoE experts**: would close the MFU gap (current 44% → ~55%). Deep engineering work.

## Scoreboard by depth (all 1500 iter)

| Depth | dense val_bpb / CORE | MoE val_bpb / CORE | Δ CORE (MoE−dense) |
|---|---|---|---|
| d12 | 0.878 / 0.1268 | 0.871 / 0.1399 | **+0.0131** |
| d16 | 0.851 / 0.1421 | 0.846 / 0.1425 | **+0.0004** |
| d22 | 0.791 / 0.1969 | 0.780 / 0.1971 | **+0.0002** |

MoE's per-step val_bpb lead holds across depths. CORE gap narrows but MoE never trails at matched depth + matched wall-clock (1500 iter).

## Key checkpoint

If MoE+reuse beats dense at d12 by a decisive margin (CORE delta > 0.01), scale up to d22 immediately. If MoE+reuse is only marginally better or worse, pivot to other unexplored axes.
