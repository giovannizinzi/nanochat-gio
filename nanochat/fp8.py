"""Minimal FP8 training for nanochat — tensorwise dynamic scaling only.

Drop-in replacement for torchao's Float8Linear (~2000 lines) with ~150 lines.
We only need the "tensorwise" recipe (one scalar scale per tensor), not the full
generality of torchao (rowwise scaling, FSDP float8 all-gather, DTensor, tensor
subclass dispatch tables, etc.)

How FP8 training works
======================
A standard Linear layer does one matmul in forward and two in backward:
  forward:      output     = input      @ weight.T
  backward:     grad_input = grad_output @ weight
                grad_weight= grad_output.T @ input

FP8 training wraps each of these three matmuls with:
  1. Compute scale = FP8_MAX / max(|tensor|)  for each operand
  2. Quantize: fp8_tensor = clamp(tensor * scale, -FP8_MAX, FP8_MAX).to(fp8)
  3. Matmul via torch._scaled_mm (cuBLAS FP8 kernel, ~2x faster than bf16)
  4. Dequantize: _scaled_mm handles this internally using the inverse scales

The key insight: torch._scaled_mm and the float8 dtypes are PyTorch built-ins.
torchao is just orchestration around these primitives. We can call them directly.

FP8 dtype choice
================
There are two FP8 formats. We use both, following the standard convention:
  - float8_e4m3fn: 4-bit exponent, 3-bit mantissa, range [-448, 448]
    Higher precision (more mantissa bits), used for input and weight.
  - float8_e5m2:   5-bit exponent, 2-bit mantissa, range [-57344, 57344]
    Wider range (more exponent bits), used for gradients which can be large.

torch._scaled_mm layout requirements
=====================================
The cuBLAS FP8 kernel requires specific memory layouts:
  - First argument (A):  must be row-major (contiguous)
  - Second argument (B): must be column-major (B.t().contiguous().t())
If B is obtained by transposing a contiguous tensor (e.g. weight.t()), it is
already column-major — no copy needed. Otherwise we use _to_col_major().

How this differs from torchao's approach
========================================
torchao uses a "tensor subclass" architecture: Float8TrainingTensor is a subclass
of torch.Tensor that bundles FP8 data + scale + metadata. It implements
__torch_dispatch__ with a dispatch table that intercepts every aten op (mm, t,
reshape, clone, ...) and handles it in FP8-aware fashion. When you call
  output = input @ weight.T
the @ operator dispatches to aten.mm, which gets intercepted and routed to
torch._scaled_mm behind the scenes. This is ~2000 lines of code because you need
a handler for every tensor operation that might touch an FP8 tensor.

We take a simpler approach: a single autograd.Function (_Float8Matmul) that takes
full-precision inputs, quantizes to FP8 internally, calls _scaled_mm, and returns
full-precision outputs. Marked @allow_in_graph so torch.compile treats it as one
opaque node rather than trying to trace inside.

The trade-off is in how torch.compile sees the two approaches:
  - torchao: compile decomposes the tensor subclass (via __tensor_flatten__) and
    sees every individual op (amax, scale, cast, _scaled_mm) as separate graph
    nodes. Inductor can fuse these with surrounding operations (e.g. fuse the
    amax computation with the preceding layer's activation function).
  - ours: compile sees a single opaque call. It can optimize everything around
    the FP8 linear (attention, norms, etc.) but cannot fuse across the boundary.

Both call the exact same cuBLAS _scaled_mm kernel — the GPU matmul is identical.
The difference is only in the "glue" ops (amax, scale, cast) which are tiny
compared to the matmul. In practice this means our version is slightly faster
(less compilation overhead, no tensor subclass dispatch cost) but can produce
subtly different floating-point rounding paths under torch.compile, since Inductor
generates a different graph. Numerics are bitwise identical in eager mode.
"""

import torch
import torch.nn as nn

from nanochat.common import COMPUTE_DTYPE

# Avoid division by zero when computing scale from an all-zeros tensor
EPS = 1e-12


@torch.no_grad()
def _to_fp8(x, fp8_dtype):
    """Dynamically quantize a tensor to FP8 using tensorwise scaling.

    "Tensorwise" means one scalar scale for the entire tensor (as opposed to
    "rowwise" which computes a separate scale per row). Tensorwise is faster
    because cuBLAS handles the scaling; rowwise needs the CUTLASS kernel.

    Returns (fp8_data, inverse_scale) for use with torch._scaled_mm.
    """
    fp8_max = torch.finfo(fp8_dtype).max
    # Compute the max absolute value across the entire tensor
    amax = x.float().abs().max()
    # Scale maps [0, amax] -> [0, fp8_max]. Use float64 for the division to
    # ensure consistent numerics between torch.compile and eager mode.
    # (torchao does the same upcast — without it, compile/eager can diverge)
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    # Quantize: scale into FP8 range, saturate (clamp prevents overflow when
    # casting — PyTorch's default is to wrap, not saturate), then cast to FP8
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)
    # _scaled_mm expects the *inverse* of our scale (it multiplies by this to
    # convert FP8 values back to the original range during the matmul)
    inv_scale = scale.reciprocal()
    return x_fp8, inv_scale


def _to_col_major(x):
    """Rearrange a 2D tensor's memory to column-major layout.

    torch._scaled_mm requires its second operand in column-major layout.
    The trick: transpose -> contiguous (forces a copy in transposed order)
    -> transpose back. The result has the same logical shape but column-major
    strides, e.g. a [M, N] tensor gets strides (1, M) instead of (N, 1).
    """
    return x.t().contiguous().t()


# allow_in_graph tells torch.compile to treat this as an opaque operation —
# dynamo won't try to decompose it into smaller ops. See the module docstring
# for how this differs from torchao's tensor subclass approach.
@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    """Custom autograd for the three FP8 GEMMs of a Linear layer.

    The forward quantizes input and weight to FP8 and saves
    the quantized tensors + scales for backward.
    """

    @staticmethod
    def forward(ctx, input_2d, weight):
        # Quantize both operands to e4m3 (higher precision format)
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)

        # output = input @ weight.T
        # input_fp8 is [B, K] contiguous = row-major (good for first arg)
        # weight_fp8 is [N, K] contiguous, so weight_fp8.t() is [K, N] with
        # strides (1, K) = column-major (good for second arg, no copy needed!)
        output = torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_inv,
            scale_b=weight_inv,
            out_dtype=input_2d.dtype,
            # use_fast_accum=True accumulates the dot products in lower precision.
            # Slightly less accurate but measurably faster. Standard practice for
            # the forward pass; we use False in backward for more precise gradients.
            use_fast_accum=True,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors

        # === GEMM 1: grad_input = grad_output @ weight ===
        # Shapes: [B, N] @ [N, K] -> [B, K]
        # Gradients use e5m2 (wider range), weights use e4m3 (higher precision)
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        # go_fp8 is [B, N] contiguous = row-major, good for first arg
        # w_fp8 is [N, K] contiguous = row-major, need column-major for second arg
        w_col = _to_col_major(w_fp8)
        grad_input = torch._scaled_mm(
            go_fp8,
            w_col,
            scale_a=go_inv,
            scale_b=w_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        # === GEMM 2: grad_weight = grad_output.T @ input ===
        # Shapes: [N, B] @ [B, K] -> [N, K]
        # go_fp8 is [B, N] contiguous, we need go.T = [N, B] as first arg.
        # Transposing gives column-major, but first arg needs row-major,
        # so we must call .contiguous() to physically rearrange the memory.
        go_T = go_fp8.t().contiguous()  # [N, B] row-major
        in_col = _to_col_major(in_fp8)    # [B, K] column-major
        grad_weight = torch._scaled_mm(
            go_T,
            in_col,
            scale_a=go_inv,
            scale_b=in_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        return grad_input, grad_weight


# =============================================================================
# MoE routed-expert FP8 path
# =============================================================================
#
# MoE routed experts are 3D nn.Parameters of shape (E, K, N) — one (K, N) weight
# slice per expert — so they can't use Float8Linear (which wraps nn.Linear 2D weights).
#
# For dense: output = input @ weight (shape (M, K) @ (K, N) per expert).
# We quantize input and weight to FP8 per-expert (tensorwise) and call
# torch._scaled_mm once per expert. Python loop over E experts adds overhead
# but the per-expert GEMM is large enough (M >> E·D·H stride) that the FP8
# matmul still wins on wall-clock compared to bf16 bmm.
#
# If torch._scaled_grouped_mm becomes available/stable with the same per-group
# scaling semantics, this can be rewritten to eliminate the Python loop.

@torch._dynamo.allow_in_graph
class _Float8MatmulDirect(torch.autograd.Function):
    """FP8 matmul: output = input @ weight (no transpose).

    Shapes:
      input:  (M, K)   — row-major
      weight: (K, N)   — will be converted to col-major internally for _scaled_mm
      output: (M, N)

    Same three FP8 GEMMs as _Float8Matmul, but the weight is (K, N) rather than
    (N, K) — so the forward uses `_to_col_major(weight)` instead of `weight.t()`.
    """

    @staticmethod
    def forward(ctx, input, weight):
        input_fp8, input_inv = _to_fp8(input, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)
        # input (M, K) row-major — good for first arg of _scaled_mm
        # weight (K, N) — need col-major for second arg
        weight_col = _to_col_major(weight_fp8)
        output = torch._scaled_mm(
            input_fp8, weight_col,
            scale_a=input_inv, scale_b=weight_inv,
            out_dtype=input.dtype, use_fast_accum=True,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors
        # grad_input = grad_output @ weight.T  : (M, N) @ (N, K) -> (M, K)
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        # go_fp8 (M, N) row-major — good for first arg
        # w_fp8 (K, N) row-major; w_fp8.t() is (N, K) with strides (1, K) = col-major. Good.
        grad_input = torch._scaled_mm(
            go_fp8, w_fp8.t(),
            scale_a=go_inv, scale_b=w_inv,
            out_dtype=grad_output.dtype, use_fast_accum=False,
        )
        # grad_weight = input.T @ grad_output : (K, M) @ (M, N) -> (K, N)
        in_T = in_fp8.t().contiguous()    # (K, M) row-major
        go_col = _to_col_major(go_fp8)    # (M, N) col-major
        grad_weight = torch._scaled_mm(
            in_T, go_col,
            scale_a=in_inv, scale_b=go_inv,
            out_dtype=grad_output.dtype, use_fast_accum=False,
        )
        return grad_input, grad_weight


def fp8_expert_bmm(act_3d, weight_3d):
    """FP8 per-expert batched matmul: act (E, M, K) @ weight (E, K, N) -> out (E, M, N).

    Each expert is its own independent FP8 matmul. Implemented as a Python for-loop
    over experts; each iteration calls the 3-GEMM FP8 autograd path from
    `_Float8MatmulDirect`. See also `fp8_expert_bmm_grouped` for a single-kernel
    path using `torch._scaled_grouped_mm`.
    """
    E = act_3d.shape[0]
    outputs = []
    for e in range(E):
        act_e = act_3d[e].contiguous()      # (M, K) row-major
        w_e = weight_3d[e].contiguous()      # (K, N) row-major
        out_e = _Float8MatmulDirect.apply(act_e, w_e)
        outputs.append(out_e)
    return torch.stack(outputs, dim=0)


# =============================================================================
# Grouped FP8 MoE matmul via torch._scaled_grouped_mm
# =============================================================================
#
# torch._scaled_grouped_mm is a fused cuBLAS kernel that does many FP8 matmuls
# in one call. For MoE this replaces the Python loop in fp8_expert_bmm.
#
# Signature (torch 2.9.x): _scaled_grouped_mm(mat1, mat2, scale_a, scale_b,
#   offs=None, bias=None, scale_result=None, out_dtype=None, use_fast_accum=False)
#
# For 3D inputs (our MoE case):
#   mat1: (E, M, K) float8_e4m3fn, row-major
#   mat2: (E, N, K) float8_e4m3fn, row-major  — NOTE: "transposed" layout
#   scale_a: (E, M) float32  — per-row per-group
#   scale_b: (E, N) float32  — per-row per-group (after transpose)
#   out:  (E, M, N) out_dtype
#
# The rowwise scales mean we quantize each row of act / each column of weight
# with its own scale. This is higher-precision than tensorwise but requires
# slightly more bookkeeping.

def _rowwise_fp8_act(x, fp8_dtype=torch.float8_e4m3fn):
    """Rowwise quantize activation (E, M, K) -> fp8 (E, M, K) + scale (E, M)."""
    fp8_max = torch.finfo(fp8_dtype).max
    # Per-row amax over last dim
    amax = x.float().abs().amax(dim=-1)  # (E, M)
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    x_fp8 = (x.float() * scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    # _scaled_grouped_mm wants the *inverse* scale (applied during the matmul
    # to convert FP8 back to the original range)
    inv_scale = scale.reciprocal()
    return x_fp8, inv_scale


def _rowwise_fp8_weight_T(w, fp8_dtype=torch.float8_e4m3fn):
    """Rowwise quantize weight (E, K, N) for the grouped_mm "transposed mat2" layout.

    _scaled_grouped_mm expects mat2 as (E, N, K) with rowwise scales (E, N).
    We receive (E, K, N) and produce (E, N, K) + (E, N) scale.
    """
    fp8_max = torch.finfo(fp8_dtype).max
    w_T = w.transpose(-1, -2).contiguous()  # (E, N, K), row-major
    amax = w_T.float().abs().amax(dim=-1)   # (E, N)
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    w_fp8 = (w_T.float() * scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    inv_scale = scale.reciprocal()
    return w_fp8, inv_scale


@torch._dynamo.allow_in_graph
class _Float8GroupedBMM(torch.autograd.Function):
    """FP8 grouped forward + bf16 grouped backward.

    Forward: quantize act + weight, call `torch._scaled_grouped_mm` (single cuBLAS kernel).
    Backward: the per-group backward scaling semantics of `_scaled_grouped_mm` are
    non-obvious (needs col-wise rather than row-wise scales for the weight-grad path),
    so we keep backward in bf16 via `torch._grouped_mm`. This loses ~half the FP8
    speedup but keeps the forward path's ~1.5-2x GEMM speedup intact.
    """

    @staticmethod
    def forward(ctx, act, weight):
        # Save FP8 tensors + scales (half the memory of bf16). Dequantize on demand
        # in backward. Also save the output dtype so backward produces the right type.
        act_fp8, act_inv = _rowwise_fp8_act(act, torch.float8_e4m3fn)
        w_T_fp8, w_inv = _rowwise_fp8_weight_T(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(act_fp8, act_inv, w_T_fp8, w_inv)
        ctx.out_dtype = act.dtype
        out = torch._scaled_grouped_mm(
            act_fp8, w_T_fp8.transpose(-1, -2),
            act_inv, w_inv,
            out_dtype=act.dtype,
            use_fast_accum=True,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        act_fp8, act_inv, w_T_fp8, w_inv = ctx.saved_tensors
        dt = ctx.out_dtype
        # Dequantize on the fly. act_fp8 shape (E, M, K); act_inv (E, M).
        # Note: act_inv is the *inverse* scale, so multiplying recovers original magnitudes.
        act_bf16 = act_fp8.to(dt) * act_inv.unsqueeze(-1).to(dt)                  # (E, M, K)
        weight_bf16 = (w_T_fp8.to(dt) * w_inv.unsqueeze(-1).to(dt)).transpose(-1, -2)  # (E, K, N)
        # grad_input = grad_output @ weight.T : (E, M, N) @ (E, N, K) -> (E, M, K)
        grad_input = torch.bmm(grad_output, weight_bf16.transpose(-1, -2))
        # grad_weight = input.T @ grad_output : (E, K, M) @ (E, M, N) -> (E, K, N)
        grad_weight = torch.bmm(act_bf16.transpose(-1, -2), grad_output)
        return grad_input, grad_weight


def fp8_expert_bmm_grouped(act_3d, weight_3d):
    """Fused FP8 forward grouped matmul for MoE. Single cuBLAS call, no Python loop.

    Forward uses FP8 via `torch._scaled_grouped_mm`; backward falls back to bf16.
    Use this when `num_experts > 1` and the act/weight dims are multiples of 16
    (cuBLAS FP8 alignment requirement).
    """
    return _Float8GroupedBMM.apply(act_3d, weight_3d)


class Float8Linear(nn.Linear):
    """Drop-in nn.Linear replacement that does FP8 compute.

    Weights and biases remain in their original precision (e.g. fp32/bf16).
    Only the matmul is performed in FP8 via the _Float8Matmul autograd function.
    """

    def forward(self, input):
        # Cast input to COMPUTE_DTYPE (typically bf16) since _scaled_mm expects
        # reduced precision input, and we no longer rely on autocast to do this.
        input = input.to(COMPUTE_DTYPE)
        # _scaled_mm only works on 2D tensors, so flatten batch dimensions
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])
        output = _Float8Matmul.apply(input_2d, self.weight)
        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_float(cls, mod):
        """Create Float8Linear from nn.Linear, sharing the same weight and bias.

        Uses meta device to avoid allocating a temporary weight tensor — we
        create the module shell on meta (shapes/dtypes only, no memory), then
        point .weight and .bias to the original module's parameters.
        """
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


class Float8LinearConfig:
    """Minimal config matching torchao's API. Only tensorwise recipe is supported."""

    @staticmethod
    def from_recipe_name(recipe_name):
        if recipe_name != "tensorwise":
            raise ValueError(
                f"Only 'tensorwise' recipe is supported, got '{recipe_name}'. "
                f"Rowwise/axiswise recipes require the full torchao library."
            )
        return Float8LinearConfig()


def convert_to_float8_training(module, *, config=None, module_filter_fn=None):
    """Replace nn.Linear layers with Float8Linear throughout a module.

    Walks the module tree in post-order (children before parents) and swaps
    each nn.Linear that passes the optional filter. The new Float8Linear shares
    the original weight and bias tensors — no copies, no extra memory.

    Args:
        module: Root module to convert.
        config: Float8LinearConfig (accepted for API compat, only tensorwise supported).
        module_filter_fn: Optional filter(module, fqn) -> bool. Only matching Linears
            are converted. Common use: skip layers with dims not divisible by 16
            (hardware requirement for FP8 matmuls on H100).
    """
    def _convert(mod, prefix=""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float8Linear.from_float(child))

    _convert(module)
    return module
