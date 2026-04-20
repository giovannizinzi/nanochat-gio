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
| v73 (dense d22 FP8, 6000 iter) | true matched-depth long run | running | | Compare against v58 MoE d22 (CORE 0.2568). |

## Scoreboard by depth (all 1500 iter)

| Depth | dense val_bpb / CORE | MoE val_bpb / CORE | Δ CORE (MoE−dense) |
|---|---|---|---|
| d12 | 0.878 / 0.1268 | 0.871 / 0.1399 | **+0.0131** |
| d16 | 0.851 / 0.1421 | 0.846 / 0.1425 | **+0.0004** |
| d22 | 0.791 / 0.1969 | 0.780 / 0.1971 | **+0.0002** |

MoE's per-step val_bpb lead holds across depths. CORE gap narrows but MoE never trails at matched depth + matched wall-clock (1500 iter).

## Key checkpoint

If MoE+reuse beats dense at d12 by a decisive margin (CORE delta > 0.01), scale up to d22 immediately. If MoE+reuse is only marginally better or worse, pivot to other unexplored axes.
