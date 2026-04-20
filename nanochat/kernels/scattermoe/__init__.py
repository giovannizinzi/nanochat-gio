"""Vendored ScatterMoE kernels (Tan et al. 2024).

Source: https://github.com/shawntan/scattermoe (Apache-2.0).
Vendored to avoid pip dependency and to keep the install Triton-only.

We use this to replace the tiny-matmul bmm bottleneck at E>=16, where cuBLAS
grouped-bmm gives <15% MFU. ScatterMoE's scatter-to-scatter Triton kernels
report 50%+ MFU on H100 at fine-grained MoE.

Import lazily via `from .parallel_experts import ...` — importing this module
directly (without triton installed) should not error, so CPU dev machines can
still build models with moe_scattermoe=False.
"""
