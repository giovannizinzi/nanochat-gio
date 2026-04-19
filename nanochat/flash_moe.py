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
        """Processes one row-block of tokens for a single expert.

        Per-thread-block work:
          1. Load x_block: (BLOCK_M, D) from x_ptr.
          2. Compute h_block[m, h] = relu²(sum_d x[m, d] * w_fc[d, h]) for h in [0, H).
             We stream over H in BLOCK_H chunks; each chunk computes one slab of hidden.
          3. For each BLOCK_H slab, accumulate out[m, d] += h_block[m, :] @ w_proj[:, d].
          4. Write final out_block back to out_ptr.

        The trick: `h_block` lives in registers/SRAM only. The (N_e, H) hidden tensor is
        never materialized in HBM, saving ~E·capacity·H·2 bytes of memory bandwidth per
        forward vs the unfused bf16 bmm path.
        """
        pid_m = tl.program_id(axis=0)
        row_start = pid_m * BLOCK_M

        m_offsets = row_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < N_e
        d_offsets = tl.arange(0, BLOCK_D)

        # Load x_block: (BLOCK_M, D). Requires D fits in one load. For d22 D=1408, that's
        # BLOCK_D=1408; might exceed SRAM for large D. If so, the kernel needs to tile D
        # too (not done here — this is a first attempt).
        x_ptrs = x_ptr + m_offsets[:, None] * stride_xn + d_offsets[None, :] * stride_xd
        x_block = tl.load(x_ptrs, mask=m_mask[:, None] & (d_offsets[None, :] < D), other=0.0)

        # Accumulator for the final output (BLOCK_M, D).
        out_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        # Stream over H in BLOCK_H-sized slabs.
        for h_start in range(0, H, BLOCK_H):
            h_offsets = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H

            # Compute h_block[:, h_slab] = x_block @ w_fc[:, h_slab], then relu².
            # w_fc slab: (D, BLOCK_H). Load once per slab.
            wfc_ptrs = (w_fc_ptr
                        + d_offsets[:, None] * stride_wfc_d
                        + h_offsets[None, :] * stride_wfc_h)
            wfc_slab = tl.load(wfc_ptrs,
                               mask=(d_offsets[:, None] < D) & h_mask[None, :],
                               other=0.0)
            # (BLOCK_M, BLOCK_H) hidden for this slab
            h_slab = tl.dot(x_block, wfc_slab)
            # relu²
            h_slab = tl.where(h_slab > 0, h_slab * h_slab, 0.0)

            # Accumulate h_slab @ w_proj[h_slab, :] into out_acc.
            wp_ptrs = (w_proj_ptr
                       + h_offsets[:, None] * stride_wp_h
                       + d_offsets[None, :] * stride_wp_d)
            wp_slab = tl.load(wp_ptrs,
                              mask=h_mask[:, None] & (d_offsets[None, :] < D),
                              other=0.0)
            # (BLOCK_M, BLOCK_H) @ (BLOCK_H, BLOCK_D) -> (BLOCK_M, BLOCK_D)
            out_acc += tl.dot(h_slab, wp_slab)

        # Write out_block.
        out_ptrs = out_ptr + m_offsets[:, None] * stride_on + d_offsets[None, :] * stride_od
        tl.store(out_ptrs,
                 out_acc.to(tl.bfloat16),
                 mask=m_mask[:, None] & (d_offsets[None, :] < D))


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

    # Tile sizes. Triton needs BLOCK_D as a power of 2 for `tl.arange`. We round D up
    # and mask off the out-of-bounds lanes inside the kernel. For d22 D=1408 → BLOCK_D=2048.
    BLOCK_M = 64
    BLOCK_D = triton.next_power_of_2(D)
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

        grid = (triton.cdiv(N_e, BLOCK_M),)
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
