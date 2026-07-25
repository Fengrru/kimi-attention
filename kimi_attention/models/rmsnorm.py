"""
RMSNorm (Root Mean Square Layer Normalization)
===============================================
Standardized implementation following the Kimi / LLaMA official specification.

RMSNorm normalizes inputs by their root-mean-square, omitting the mean-centering
step found in traditional LayerNorm. This reduces computation while maintaining
training stability.

Reference:
    Zhang, B. & Sennrich, R. (2019). Root Mean Square Layer Normalization.
    NeurIPS 2019.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Computes::
        output = x / sqrt(mean(x^2) + eps) * gamma

    where ``gamma`` is a learned per-channel scale parameter.

    Args:
        dim: Number of features (channels) in the input.
        eps: Small constant for numerical stability. Default: 1e-6.

    Shape:
        - Input: ``(*, dim)`` where ``*`` denotes any number of leading dimensions.
        - Output: same shape as input.

    Example::
        >>> norm = RMSNorm(512)
        >>> x = torch.randn(2, 16, 512)
        >>> out = norm(x)
        >>> rms = out.pow(2).mean(-1)
        >>> assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"
