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

## Follow-up session findings (2026-04-20)

### ScatterMoE Triton kernel works, but replicated memory ceiling blocks true fine-grained MoE

Vendored ScatterMoE (github.com/shawntan/scattermoe, Apache-2.0) into
`nanochat/kernels/scattermoe/`. Wired behind `--moe-scattermoe` flag.
Kernel runs correctly and saturates tensor cores (v85 E=16 K=2 Dₑ=1024
bf16: 1.42s/step ≈ v58 bmm baseline).

**Memory ceiling discovered** at d22 replicated: active-intermediate buffers
scale as K × N × Dₑ × 22 layers × 2 bytes × (forward + backward + autograd
save). Practical budget is K × Dₑ ≲ 2048 before OOM. Pareto frontier:

| config | K | Dₑ | active FFN | fits? |
|---|---|---|---|---|
| v85 ScatterMoE | 2 | 1024 | 3072 | ✅ |
| v87 ScatterMoE | 2 | 2048 | 6144 | ❌ OOM |
| v88 ScatterMoE | 4 | 512 | 2560 | ❌ OOM |
| v89 ScatterMoE | 2 | 512 | 1536 | ✅ |

This means at d22 replicated, ScatterMoE cannot match v74's active=8192 setup.
Either sparsity (E≥16) or compute (active≥6144) — not both.

**v89 E=32 K=2 sh=1 Dₑ=512 (true fine-grained, r_a=9%)**: CORE 0.1818,
val_bpb 0.809. Worst so far because active FFN=1536 is way undermatched vs
dense d22's 5632. Sparsity alone does not compensate for compute starvation.

### Other attempts this session

| idea | result |
|---|---|
| `--moe-routed-scaling=2.827` (DeepSeek-V3/K2.6) | in code, only tested in broken v83 |
| `--moe-auxfree-bias` (DeepSeek-V3 aux-loss-free) | v83 ran at 3.8 s/step, MFU 13% — in-place `.add_()` on the bias buffer broke torch.compile |
| V3-mini E=32 K=4 Dₑ=512 bmm (v79) | 3.4 s/step, MFU 10% — tiny-matmul disaster, killed |
| sh=2 Dₑ=2752 (v78) | CORE 0.1844, outlier noise below sh=1 |
| K=2 sh=3 combined (v77) | +0.0008 CORE over K=1 sh=3, marginal |
| sh=5 Dₑ=1408 (v76) | CORE 0.2072 (drops below sh=3 peak 0.2090) |
| E=8 K=1 sh=3 Dₑ=1408 + auxfree + routed_scaling=2.827 (v83) | CORE 0.1939, below dense d22 |

### What would unlock true very-sparse MoE on this hardware

1. **Expert Parallelism + working CORE eval**: shard E experts across 8 GPUs,
   each rank holds E/8 experts. Removes the replicated memory ceiling. Our
   `_forward_ep` exists but CORE eval crashed on variable-shape all_to_all.
   Workaround is `--core-metric-every=999999`; still worth a retry.
2. **Gradient checkpointing on MoE blocks**: trades compute for activation
   memory. Would let v87 (K=2 Dₑ=2048) fit. Modest code change.
3. **FP8 MoE through forward AND backward**: halves memory + improves MFU.
   Our current FP8 grouped saves FP8 tensors for backward but still holds
   bf16 activations for autograd.
4. **Switch to d16/d18 MoE**: smaller activations → more MoE memory budget.
   But prior data showed MoE-vs-dense gap widens at lower depths (MoE only
   led at d12).

### Verdict

**v75 (d22 MoE E=4 K=1 sh=3 Dₑ=2048 bmm, CORE 0.2651 at 142 min) remains
the best MoE config found.** Dense d22 FP8 (CORE 0.2694 at 88 min) still
wins time-to-CORE by ~55 minutes at matched depth.

The core bottleneck is structural: at d22 replicated, we can't access the
configurations (very-sparse + compute-matched) where modern Chinese OSS MoE
actually beats dense. Closing the gap requires either EP or a scale where
VRAM is less constrained (d16 probably; Karpathy speedrun length definitely).

