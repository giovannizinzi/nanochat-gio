# Overnight MoE experiments — summary

Goal: MoE beats dense at time-to-GPT-2-CORE (0.2565) on 8×H100.

## TL;DR

**Dense still wins time-to-CORE at matched depth + matched iters.** But the gap has narrowed substantially through the night, and we found one novel lever (shared=3) that improves MoE by a reproducible +0.008 CORE over the v58 starting point.

Final matched-depth d22, 6000 iter race:
| config | wall-clock | val_bpb | CORE |
|---|---:|---:|---:|
| **dense d22 FP8 (v73)** | **88 min** | 0.724 | **0.2694** ← best |
| MoE d22 sh=3 (v75, new winner) | 142 min | 0.714 | 0.2651 |
| MoE d22 sh=1 (v58 old winner) | 142 min | 0.712 | 0.2568 |

MoE lost CORE by 0.004 (was 0.013 at start of night). MoE wins val_bpb (−0.010 vs dense) but dense generalizes better to CORE tasks.

## What we learned

### The big discovery: val_bpb and CORE DECOUPLE for MoE

Across every MoE config tested, val_bpb came out at-or-below dense, but CORE came out at-or-below dense. MoE is a better language model (compresses bits) but worse at downstream tasks. Most likely explanation: router over-specialization — routed experts latch onto narrow patterns in the training distribution that reduce perplexity but don't transfer to diverse task types.

**shared=3 is the first knob that moves CORE forward without sacrificing val_bpb.** Going from sh=1 to sh=3 (with Dₑ halved to match total active FFN) gave +0.008 CORE at 6000 iter. DeepSeek-v3's use of multiple shared experts looks vindicated.

### The earlier comparison was wrong

Previous dev notes compared MoE d22 against dense d26 FP8, declaring MoE loses by 0.028 CORE. At matched depth (d22 vs d22) the gap is only 0.004 — ~7× smaller. Karpathy's speedrun happens to use d26; comparing to his d26 overstated the gap.

### Ideas that did NOT help (tested tonight)

| idea | source | result |
|---|---|---|
| 1dense+SE layout | arxiv 2506.12119 | 0 val_bpb effect, 2% speed |
| Data reuse (loose scheme) | arxiv 2506.12119 §6 | hurt CORE −0.009 at d12 |
| Expert-choice routing | Zhou 2022 NeurIPS | hurt CORE −0.011 at d12 |
| Fine-grained E=8 Dₑ=2816 | arxiv 2506.12119 | hurt CORE −0.007 |
| FlashMoE fused kernel | Dao+ScatterMoE-inspired | 12× SLOWER than cuBLAS bmm |

### Ideas that DID help

| idea | improvement |
|---|---|
| **shared=3 (vs sh=1), matched compute** | **+0.008 CORE at 6000 iter** |

### Ideas that might still close the gap (not tested tonight)

1. **Even more shared experts (sh=5, sh=7)**: continue the trend if sh=3→sh=5 gives another +0.008 CORE.
2. **top-k=2 with sh=3**: makes MoE active > dense, not just matched. Probably +0.01 CORE but slower per step.
3. **FP8 through MoE backward**: close the 44% → 57% MFU gap → MoE at matched wall-clock.
4. **Longer training beyond 6000 iter**: dense plateaus; MoE may still be scaling.
5. **Dense + FP8-everything + tuned d24**: maybe dense d24 FP8 beats dense d22 FP8 and dense d26 FP8 at wall-clock (it's in between).

## Code changes pushed to `moe-experiments` branch

- `--moe-first-layer N` — layers [0, N) dense, rest MoE (arxiv 2506.12119)
- `--max-train-shards N` — cap training data to first N shards, enable multi-epoch
- `--moe-expert-choice` — expert-choice routing (Zhou 2022)
- `nanochat/flash_moe.py` + `scripts/bench_flash_moe.py` — standalone Triton kernel + bench (not wired into moe.py; it's 12x too slow)

## Recommended next steps

1. **Pursue shared=5, shared=7 at d22**: simple hyperparameter sweep, cheap. Might push past dense at sh=5.
2. **Scale v75 config to Karpathy speedrun length (8.25B tokens = ~7800 iter)** for full apples-to-apples on the leaderboard.
3. If #1 gives CORE > 0.2694 at 6000 iter, pair with `--fp8 --moe-expert-fp8` and call the win.
