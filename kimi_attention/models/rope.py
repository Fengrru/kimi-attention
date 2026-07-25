"""
Rotary Position Embedding (RoPE)
================================
Implements the rotary position encoding from Su et al. (2021) used in
LLaMA, GPT-NeoX, Mistral, and Kimi-Linear.

RoPE encodes position information by rotating query and key vectors
in pairs of dimensions, making the dot product depend only on the
relative distance between tokens.

Reference:
    Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by swapping halves and negating the first."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to tensor x.

    Args:
        x: Input ``[..., T, D]`` or ``[B, H, T, D]``.
        cos: Cosine values broadcastable with x.
        sin: Sine values broadcastable with x.

    Returns:
        Rotated tensor of same shape as x.
    """
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """Precomputed rotary position embeddings.

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum sequence length to precompute.
        theta: Base frequency for the geometric progression (default 10000.0).

    Buffers:
        cos_cached: ``[max_seq_len, dim]``.
        sin_cached: ``[max_seq_len, dim]``.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)  # [max_seq_len, dim // 2]

        # Interleave to full dim: [f0, f0, f1, f1, ...]
        # Standard RoPE requires adjacent dimension pairs to share frequency.
        emb = freqs.repeat_interleave(2, dim=-1)  # [max_seq_len, dim]

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RoPE to a tensor.

        Args:
            x: Input ``[B, H, T, D]`` where D is head_dim.
            position_ids: Position indices ``[B, T]`` (e.g. arange).

        Returns:
            Rotated tensor of same shape as x.
        """
        cos = self.cos_cached[position_ids]  # [B, T, D]
        sin = self.sin_cached[position_ids]  # [B, T, D]

        # Insert head dim: [B, 1, T, D]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        return apply_rotary_emb(x, cos, sin)
