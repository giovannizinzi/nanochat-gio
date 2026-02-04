"""
FP8 utilities for Blackwell/Hopper-class GPUs.

Includes a minimal MXFP8-style block scaling implementation in pure PyTorch.
This follows the principle of quantizing both non-transposed and transposed
copies from high-precision inputs to avoid double-quantization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# FP8 format constants
FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0
EPS = 1e-12


def fp8_available() -> bool:
    if not torch.cuda.is_available():
        return False
    # torch._scaled_mm is required for FP8 kernels
    if not hasattr(torch, "_scaled_mm"):
        return False
    # Hopper (SM90) and newer are required for FP8 tensor cores
    major, _minor = torch.cuda.get_device_capability()
    return major >= 9


def linear_dims_supported(in_features: int, out_features: int) -> bool:
    return (in_features % 16 == 0) and (out_features % 16 == 0)


def block_size_supported(block_size: int) -> bool:
    return block_size in {8, 16, 32}


def _enable_fp8_matmul_settings():
    # Best-effort: enable reduced precision accumulation for FP8 on supported builds.
    try:
        matmul_cfg = torch.backends.cuda.matmul
        if hasattr(matmul_cfg, "allow_fp8_reduced_precision_reduction"):
            matmul_cfg.allow_fp8_reduced_precision_reduction = True
    except Exception:
        pass


def _block_scales(x: torch.Tensor, block_size: int) -> torch.Tensor:
    # Compute one scale per block along the last dimension.
    K = x.shape[-1]
    num_blocks = (K + block_size - 1) // block_size
    scales = []
    for i in range(num_blocks):
        k0 = i * block_size
        k1 = min(K, (i + 1) * block_size)
        amax = x[..., k0:k1].abs().max()
        scale = (amax / FP8_E4M3_MAX).clamp(min=EPS).float()
        scales.append(scale)
    return torch.stack(scales)


def _quantize_blockwise(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Returns fp8 tensor and per-block scales (1D tensor length num_blocks).
    scales = _block_scales(x, block_size)
    x_f8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    K = x.shape[-1]
    for i, s in enumerate(scales):
        k0 = i * block_size
        k1 = min(K, (i + 1) * block_size)
        x_f8[..., k0:k1] = (x[..., k0:k1] / s).to(torch.float8_e4m3fn)
    return x_f8, scales


def _scaled_mm_blockwise(
    x_f8: torch.Tensor,
    w_f8_t: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    block_size: int,
    out_dtype: torch.dtype,
    use_fast_accum: bool,
) -> torch.Tensor:
    # x_f8: (N, K), w_f8_t: (K, M), scales per K-block
    N, K = x_f8.shape
    K2, M = w_f8_t.shape
    assert K == K2
    out = None
    for i in range(len(x_scales)):
        k0 = i * block_size
        k1 = min(K, (i + 1) * block_size)
        if k0 >= k1:
            continue
        x_blk = x_f8[:, k0:k1].contiguous()
        w_blk = w_f8_t[k0:k1, :].contiguous()
        part = torch._scaled_mm(
            x_blk,
            w_blk,
            out_dtype=out_dtype,
            scale_a=x_scales[i],
            scale_b=w_scales[i],
            use_fast_accum=use_fast_accum,
        )
        out = part if out is None else out + part
    return out


class _FP8Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, grad_scale: torch.Tensor):
        # x: (N, C) bf16, w: (V, C) bf16
        x = x.contiguous()
        w = w.contiguous()

        # Blockwise quantization along reduction dimension (K)
        x_f8, x_scales = _quantize_blockwise(x, block_size=_FP8Matmul.block_size)
        w_f8, w_scales = _quantize_blockwise(w, block_size=_FP8Matmul.block_size)

        # Create transposed copies from high-precision inputs to avoid FP8 transpose re-quantization
        x_t_f8, x_t_scales = _quantize_blockwise(x.T, block_size=_FP8Matmul.block_size)
        w_t_f8, w_t_scales = _quantize_blockwise(w.T, block_size=_FP8Matmul.block_size)

        out = _scaled_mm_blockwise(
            x_f8, w_t_f8, x_scales, w_t_scales,
            _FP8Matmul.block_size, out_dtype=torch.bfloat16, use_fast_accum=True
        )

        ctx.save_for_backward(x_f8, w_f8, x_scales, w_scales, x_t_f8, w_t_f8, x_t_scales, w_t_scales, grad_scale)
        ctx.w_dtype = w.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_f8, w_f8, x_scales, w_scales, x_t_f8, w_t_f8, x_t_scales, w_t_scales, grad_scale = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_f8 = (grad_out / grad_scale).to(torch.float8_e5m2)

        # grad_x = grad @ W (use non-transposed W fp8)
        grad_scale_blocks = grad_scale.repeat(len(w_scales))
        grad_x = _scaled_mm_blockwise(
            grad_f8, w_f8, grad_scale_blocks, w_scales,
            _FP8Matmul.block_size, out_dtype=torch.bfloat16, use_fast_accum=False
        )

        # grad_w = x.T @ grad (use transposed X fp8)
        grad_scale_blocks_t = grad_scale.repeat(len(x_t_scales))
        grad_w_t = _scaled_mm_blockwise(
            x_t_f8, grad_f8, x_t_scales, grad_scale_blocks_t,
            _FP8Matmul.block_size, out_dtype=torch.float32, use_fast_accum=False
        )
        grad_w = grad_w_t.T.to(ctx.w_dtype)

        return grad_x, grad_w, None


class LinearFP8LMHead(nn.Linear):
    """
    FP8 Linear layer intended for lm_head.

    Assumes input shape (B, T, C) and cross-entropy loss with mean reduction,
    so grad_scale is computed as (1 / (B*T)) / FP8_E5M2_MAX.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        assert bias is False, "LinearFP8LMHead does not support bias"
        super().__init__(in_features, out_features, bias=False)
        _enable_fp8_matmul_settings()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not fp8_available():
            return F.linear(x, self.weight.type_as(x))

        assert x.ndim == 3 and x.shape[2] == self.in_features, (
            f"Expected input shape (B, T, {self.in_features}), got {x.shape}"
        )
        B, T, _ = x.shape
        _x = x.flatten(0, -2)  # (B*T, C)
        grad_scale = torch.tensor(1.0 / (B * T) / FP8_E5M2_MAX, device=x.device, dtype=torch.float32)
        _FP8Matmul.block_size = 16
        out = _FP8Matmul.apply(_x, self.weight, grad_scale)
        return out.reshape(B, T, -1)


class LinearMXFP8(nn.Linear):
    """
    Full-model MXFP8-style Linear with block scaling.

    Uses FP8 for forward and backward matmuls with blockwise scaling along
    the reduction dimension. Creates both normal and transposed FP8 copies
    from high precision to avoid double quantization.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, block_size: int = 16, monitor: bool = False):
        assert bias is False, "LinearMXFP8 does not support bias"
        super().__init__(in_features, out_features, bias=False)
        self.block_size = block_size
        self.monitor = monitor
        self._x_amax = None
        self._w_amax = None
        self._grad_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not fp8_available():
            return F.linear(x, self.weight.type_as(x))

        if x.ndim == 3:
            B, T, _ = x.shape
            _x = x.flatten(0, -2)
            grad_scale = torch.tensor(1.0 / (B * T) / FP8_E5M2_MAX, device=x.device, dtype=torch.float32)
            _FP8Matmul.block_size = self.block_size
            out = _FP8Matmul.apply(_x, self.weight, grad_scale)
            if self.monitor:
                self._x_amax = _x.detach().abs().max().item()
                self._w_amax = self.weight.detach().abs().max().item()
                self._grad_scale = grad_scale.detach().item()
            return out.reshape(B, T, -1)

        _FP8Matmul.block_size = self.block_size
        grad_scale = torch.tensor(1.0 / x.numel() / FP8_E5M2_MAX, device=x.device, dtype=torch.float32)
        out = _FP8Matmul.apply(x, self.weight, grad_scale)
        if self.monitor:
            self._x_amax = x.detach().abs().max().item()
            self._w_amax = self.weight.detach().abs().max().item()
            self._grad_scale = grad_scale.detach().item()
        return out

    def get_fp8_stats(self) -> dict:
        return {
            "x_amax": self._x_amax,
            "w_amax": self._w_amax,
            "grad_scale": self._grad_scale,
            "block_size": self.block_size,
        }
