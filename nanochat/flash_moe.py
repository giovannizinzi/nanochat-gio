"""FlashMoE — fused Triton kernel for MoE expert FFN (WIP attempt).

Motivation (from dev/MOE_WHY_DENSE_WINS.md): our MoE per-step time breakdown shows
~450 ms/step of overhead vs dense, dominated by:
  - 330 ms for expert-bmm forward + backward
  - 140 ms for dispatch buffer memory traffic

A FlashMoE kernel in the style of FlashAttention collapses these into fewer kernel
calls, keeping intermediate tensors in SRAM and avoiding the (E, capacity, H) hidden
buffer that bounces through HBM.

This file is an **initial implementation** — it compiles and runs the expected shapes,
but hasn't been extensively tested for numerical parity with the reference bf16 bmm
path. Before shipping it replace-the-default, verify vs `nanochat.moe.MoE`'s forward
at several configs and compare val_bpb trajectory.

Inspired by / credits:
  - ScatterMoE (Shawn Tan, https://github.com/shawntan/scattermoe) — scatter-to-scatter
    approach that avoids padded dispatch buffers entirely.
  - FlashAttention (Dao et al.) — general template of fused matmul + epilogue in Triton.

Scope of this kernel:
  - Input: tokens already sorted by expert assignment (done in PyTorch before kernel call).
  - Per-expert block: launches a tile over the expert's contiguous token chunk + loads
    the expert's (D, H) and (H, D) weights once per expert block.
  - Output: fully-processed token outputs (after FFN), no intermediate hidden buffer
    materialization in HBM.

What's NOT attempted here (would be needed for a production FlashMoE):
  - Fused scatter of router outputs to pre-sorted token order (still Python).
  - FP8 math inside the kernel (this version is bf16).
  - A corresponding backward kernel (current version relies on PyTorch autograd recompute).
  - Dynamic expert-block boundaries in a single launch (we launch one kernel per expert).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _expert_ffn_kernel(
        # Pointers
        x_ptr,           # (N_e, D)     input tokens for this expert (contiguous)
        w_fc_ptr,        # (D, H)       expert's c_fc weight
        w_proj_ptr,      # (H, D)       expert's c_proj weight
        out_ptr,         # (N_e, D)     output buffer for this expert
        # Shapes
        N_e,             # number of tokens routed to this expert
        D: tl.constexpr, # token / output dim
        H: tl.constexpr, # hidden dim
        # Strides
        stride_xn, stride_xd,
        stride_wfc_d, stride_wfc_h,
        stride_wp_h, stride_wp_d,
        stride_on, stride_od,
        # Tile sizes
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        """Proper tiled fused FFN: never loads the full D-dim of x into SRAM at once.

        Grid: (cdiv(N_e, BLOCK_M), cdiv(D, BLOCK_D))
        Per thread block processes one (BLOCK_M rows) × (BLOCK_D out-columns) output tile.

        For each BLOCK_H slab of the hidden dim:
          1. GEMM #1 tiled over D: compute (BLOCK_M, BLOCK_H) hidden slab, fp32 accum.
             Inner loop over d_in_start in [0, D, BLOCK_D): loads (BLOCK_M, BLOCK_D) x chunk
             and (BLOCK_D, BLOCK_H) w_fc chunk, accumulates `h_slab_acc`.
          2. Elementwise: h_slab = relu(h_slab_acc)²  (in fp32, cast to bf16 for the next dot)
          3. GEMM #2: out_tile += h_slab_bf @ w_proj[h_slab_start:h_slab_end, d_out_start:]
        Store out_tile once H is fully consumed.

        The key win over unfused bmm+relu²+bmm: the (BLOCK_M, BLOCK_H) hidden slab
        lives in registers and never touches HBM. Over a full d22 forward (cap=~20k tokens,
        H=4096), this eliminates ~2.5 GB of HBM round-trips per forward.
        """
        pid_m = tl.program_id(axis=0)
        pid_d = tl.program_id(axis=1)

        m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_off < N_e
        d_out_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_out_mask = d_out_off < D

        out_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        # Stream over H in BLOCK_H-sized slabs.
        for h_start in range(0, H, BLOCK_H):
            h_off = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_off < H

            # GEMM #1: (BLOCK_M, BLOCK_H) = sum_d x[:, d] * w_fc[d, h_slab]
            # Inner-tiled over D so we never load (BLOCK_M, D) at once.
            h_slab_acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
            for d_in_start in range(0, D, BLOCK_D):
                d_in_off = d_in_start + tl.arange(0, BLOCK_D)
                d_in_mask = d_in_off < D
                x_ptrs = x_ptr + m_off[:, None] * stride_xn + d_in_off[None, :] * stride_xd
                x_chunk = tl.load(x_ptrs, mask=m_mask[:, None] & d_in_mask[None, :], other=0.0)
                wfc_ptrs = (w_fc_ptr
                            + d_in_off[:, None] * stride_wfc_d
                            + h_off[None, :] * stride_wfc_h)
                wfc_chunk = tl.load(wfc_ptrs, mask=d_in_mask[:, None] & h_mask[None, :], other=0.0)
                h_slab_acc += tl.dot(x_chunk, wfc_chunk)

            # relu² in fp32, then cast to bf16 for the next dot (tensor cores want bf16 inputs).
            h_slab_acc = tl.where(h_slab_acc > 0, h_slab_acc * h_slab_acc, 0.0)
            h_slab_bf = h_slab_acc.to(tl.bfloat16)

            # GEMM #2 (accumulating): out_tile += h_slab @ w_proj[h_slab, d_out_slab]
            wp_ptrs = (w_proj_ptr
                       + h_off[:, None] * stride_wp_h
                       + d_out_off[None, :] * stride_wp_d)
            wp_chunk = tl.load(wp_ptrs, mask=h_mask[:, None] & d_out_mask[None, :], other=0.0)
            out_acc += tl.dot(h_slab_bf, wp_chunk)

        out_ptrs = out_ptr + m_off[:, None] * stride_on + d_out_off[None, :] * stride_od
        tl.store(out_ptrs, out_acc.to(tl.bfloat16),
                 mask=m_mask[:, None] & d_out_mask[None, :])


def flash_moe_expert_ffn(x_sorted: torch.Tensor,
                         w_fc: torch.Tensor,
                         w_proj: torch.Tensor,
                         expert_offsets: torch.Tensor) -> torch.Tensor:
    """Run the fused FFN kernel per expert over expert-sorted tokens.

    Args:
        x_sorted: (N_total, D), bf16, tokens in expert-sorted order.
        w_fc:     (E, D, H)    bf16
        w_proj:   (E, H, D)    bf16
        expert_offsets: (E+1,) int64 — cumulative boundaries; expert e's tokens are
                                       x_sorted[offsets[e] : offsets[e+1]].

    Returns:
        out: (N_total, D) bf16, same order as x_sorted.

    This is a **reference-only / experimental** entry point. It launches one kernel per
    expert (sequential launches; a production path would fuse all experts into a single
    launch via grouped Triton, matching ScatterMoE's design). Use only behind a flag.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available — install triton to use flash_moe_expert_ffn")

    assert x_sorted.dim() == 2 and x_sorted.is_contiguous(), \
        "x_sorted must be (N, D) contiguous"
    assert w_fc.dim() == 3 and w_proj.dim() == 3, "expert weights must be (E, D, H) / (E, H, D)"
    E, D, H = w_fc.shape
    assert w_proj.shape == (E, H, D), f"w_proj shape mismatch: {w_proj.shape}"
    assert expert_offsets.shape == (E + 1,)

    N_total = x_sorted.shape[0]
    out = torch.empty_like(x_sorted)

    # Tile sizes — all powers of 2 (Triton requirement). These fit H100 SRAM with room
    # for the register accumulators: per-thread-block footprint is roughly
    #   x_chunk (BLOCK_M × BLOCK_D × 2)   +  w_fc_chunk (BLOCK_D × BLOCK_H × 2)
    # + h_slab (BLOCK_M × BLOCK_H × 4fp32) + out_tile (BLOCK_M × BLOCK_D × 4fp32)
    # ≈ 64·128·2 + 128·128·2 + 64·128·4 + 64·128·4  ≈ 112 KB  (well under 256 KB SRAM).
    BLOCK_M = 64
    BLOCK_D = 128
    BLOCK_H = 128

    offs = expert_offsets.tolist()
    for e in range(E):
        start, end = offs[e], offs[e + 1]
        N_e = end - start
        if N_e <= 0:
            continue
        x_e = x_sorted[start:end]       # (N_e, D) view (contiguous since it's a prefix slice)
        w_fc_e = w_fc[e]                 # (D, H)
        w_proj_e = w_proj[e]             # (H, D)
        out_e = out[start:end]

        grid = (triton.cdiv(N_e, BLOCK_M), triton.cdiv(D, BLOCK_D))
        _expert_ffn_kernel[grid](
            x_e, w_fc_e, w_proj_e, out_e,
            N_e, D, H,
            x_e.stride(0), x_e.stride(1),
            w_fc_e.stride(0), w_fc_e.stride(1),
            w_proj_e.stride(0), w_proj_e.stride(1),
            out_e.stride(0), out_e.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H,
        )
    return out
