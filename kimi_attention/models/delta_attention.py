"""
Kimi Delta Attention (KDA)
==========================
Production implementation of Kimi's revolutionary linear attention mechanism
with fine-grained channel-wise gating.

KDA improves upon traditional linear attention (e.g., Linear Transformer,
RWKV) by introducing per-dimension independent forget gates, enabling
dynamic memory control at the finest granularity.

Core Formulation
----------------
Memory update (fine-grained delta rule)::

    m_t = beta_t * m_{t-1} + (1 - beta_t) * (k_t * v_t)

Output computation::

    o_t = g_t * (q_t * m_t)

where:
    - ``beta_t`` in [0, 1]^d is the channel-wise forget rate (learned)
    - ``g_t`` is the output gate (typically SiLU-activated)
    - ``*`` denotes element-wise multiplication
    - ``m_t`` is the incremental memory state

Advantages over Standard Attention
----------------------------------
    - **O(T) complexity**: Linear in sequence length (vs. O(T^2) for
softmax)
    - **Constant KV cache**: Memory does not grow with sequence length
    - **6x faster decoding** on 1M+ context lengths
    - **75% KV cache reduction** compared to standard MLA

Reference:
    Moonshot AI (2025). Kimi Linear: Scaling Linear Attention to 48
Billion
    Parameters. arXiv:2510.26692.
    
    Flash Linear Attention (FLA): https://github.com/fla-org/flash-
linear-attention
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kimi_attention.models.rmsnorm import RMSNorm


class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention core computation.

    This module implements the KDA forward pass with automatic fallback:
        1. If ``fla-core`` is installed and inputs are on CUDA, it uses
           the optimized ``chunk_kda`` CUDA kernel.
        2. Otherwise, it falls back to a pure PyTorch sequential
           implementation suitable for CPU inference or debugging.

    Args:
        chunk_size: Chunk size for the FLA kernel. Only used when FLA is
            available. Default: 64 (official optimum).

    Example::
        >>> kda = KimiDeltaAttention(chunk_size=64)
        >>> B, H, T, D = 2, 8, 256, 64
        >>> q = torch.randn(B, H, T, D)
        >>> k = torch.randn(B, H, T, D)
        >>> v = torch.randn(B, H, T, D)
        >>> beta = torch.randn(B, H, T, D)
        >>> g = torch.sigmoid(torch.randn(B, H, T, D))
        >>> out = kda(q, k, v, beta, g)
        >>> assert out.shape == (B, H, T, D)
    """

    def __init__(self, chunk_size: int = 64) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self._has_fla: Optional[bool] = None
        self._chunk_kda = None
        self._recurrent_state: Optional[torch.Tensor] = None

    def _try_import_fla(self) -> bool:
        """Lazily import FLA to avoid hard dependency."""
        if self._has_fla is not None:
            return self._has_fla
        try:
            from fla.ops.kda import chunk_kda  # type: ignore[import-untyped]

            self._chunk_kda = chunk_kda
            self._has_fla = True
        except Exception:
            self._has_fla = False
        return self._has_fla

    @staticmethod
    def _parallel_recurrence(
        a: torch.Tensor,
        b: torch.Tensor,
        init: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel prefix scan: m_t = a_t * m_{t-1} + b_t for all t.

        Hillis-Steele inclusive scan in O(log C) steps.  Used inside
        chunks; suitable for C ≤ 256 where double-buffer overhead is
        negligible.

        Args:
            a: Multiplicative coefficients ``[B, H, C, D]``.
            b: Additive terms ``[B, H, C, D]``.
            init: Initial state ``[B, H, D]``.

        Returns:
            All recurrent states ``[B, H, C, D]``.
        """
        B, H, C, D = a.shape

        if C == 1:
            return a * init.unsqueeze(2) + b

        C_orig = C
        C_pow2 = 1
        while C_pow2 < C:
            C_pow2 <<= 1
        if C_pow2 > C:
            pad = C_pow2 - C
            a = F.pad(a, (0, 0, 0, pad), value=1.0)
            b = F.pad(b, (0, 0, 0, pad), value=0.0)

        C = C_pow2

        # Clone to avoid in‑place modification of caller's tensor
        b = b.clone()
        b[:, :, 0, :] = a[:, :, 0, :] * init + b[:, :, 0, :]

        stride = 1
        while stride < C:
            a_prev = torch.cat(
                [
                    torch.ones(B, H, stride, D, device=a.device, dtype=a.dtype),
                    a[:, :, : C - stride, :],
                ],
                dim=2,
            )
            b_prev = torch.cat(
                [
                    torch.zeros(B, H, stride, D, device=a.device, dtype=a.dtype),
                    b[:, :, : C - stride, :],
                ],
                dim=2,
            )
            a_next = a * a_prev
            b_next = a * b_prev + b
            a, b = a_next, b_next
            stride <<= 1

        return b[:, :, :C_orig, :]

    def _forward_pytorch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        """Chunked parallel PyTorch fallback — no FLA kernel needed.

        Splits the sequence into chunks of ``chunk_size``.  Within each
        chunk a parallel prefix scan computes all recurrent states in
        O(log chunk_size) steps; between chunks the state is propagated
        sequentially.  This is far faster than fully-sequential looping
        while staying in pure PyTorch.

        If ``self._recurrent_state`` is not None, the forward pass starts
        from that stored state (used for single‑token incremental decoding).
        After completion, the final memory state is saved for the next step.

        Complexity:
            Time:  O(T · B · H · D)  (parallelised over the chunk dim)
            Space: O(B · H · chunk_size · D)
        """
        B, H, T, D = q.shape

        beta_sig = torch.sigmoid(beta)
        a_coeff = beta_sig
        b_coeff = (1.0 - beta_sig) * (k * v)

        chunk_size = max(self.chunk_size, 1)
        outputs = []

        # Use stored recurrent state only when NOT training and shapes match.
        # During training every forward is independent (fresh zero-state).
        if not self.training and self._recurrent_state is not None:
            rs = self._recurrent_state
            if rs.shape[0] == B and rs.shape[1] == H and rs.shape[2] == D:
                m_state = rs
            else:
                self._recurrent_state = None
                m_state = torch.zeros(B, H, D, device=q.device, dtype=q.dtype)
        else:
            m_state = torch.zeros(B, H, D, device=q.device, dtype=q.dtype)

        for c in range(0, T, chunk_size):
            end = min(c + chunk_size, T)

            a_c = a_coeff[:, :, c:end, :]
            b_c = b_coeff[:, :, c:end, :]
            q_c = q[:, :, c:end, :]
            g_c = g[:, :, c:end, :]

            m_c = self._parallel_recurrence(a_c, b_c, m_state)

            o_c = g_c * (q_c * m_c)
            outputs.append(o_c)

            m_state = m_c[:, :, -1, :]

        # Store final memory state for incremental decoding (eval mode only)
        if not self.training:
            self._recurrent_state = m_state

        return torch.cat(outputs, dim=2)  # [B, H, T, D]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        """Execute KDA forward pass.

        Args:
            q: Query tensor ``[B, H, T, D]``.
            k: Key tensor ``[B, H, T, D]``.
            v: Value tensor ``[B, H, T, D]``.
            beta: Raw forget gate logits ``[B, H, T, D]``. Will be
                sigmoid-activated internally.
            g: Output gate ``[B, H, T, D]``. Should be pre-activated
(typically
                with SiLU) before passing to this function.

        Returns:
            Attention output ``[B, H, T, D]``.
        """
        if self._try_import_fla() and q.is_cuda:
            beta_sig = torch.sigmoid(beta)
            return self._chunk_kda(
                q, k, v, beta_sig, g, chunk_size=self.chunk_size
            )
        else:
            if not q.is_cuda and self._try_import_fla():
                warnings.warn(
                    "FLA kernel requires CUDA. Falling back to PyTorch CPU "
                    "implementation. For production GPU inference, install "
                    "fla-core and ensure inputs are on CUDA device.",
                    stacklevel=2,
                )
            return self._forward_pytorch(q, k, v, beta, g)


class KimiDeltaAttentionLayer(nn.Module):
    """Complete KDA attention layer (drop-in replacement for standard
 attention).

    This module combines all necessary projections, normalization, and the
    KDA computation into a single layer that can directly replace
    ``nn.MultiheadAttention`` in any Transformer architecture.

    Args:
        dim: Model dimension (d_model). Must be divisible by num_heads.
        num_heads: Number of attention heads. Default: 8.
        chunk_size: Chunk size for FLA kernel. Default: 64.
        eps: RMSNorm stability constant. Default: 1e-6.
        rope: Optional RotaryEmbedding module for positional encoding.

    Attributes:
        q_proj, k_proj, v_proj: Input projections for Q, K, V.
        beta_proj: Projects input to forget gate logits.
        g_proj: Projects input to output gate (followed by SiLU).
        o_proj: Output projection.

    Example::
        >>> layer = KimiDeltaAttentionLayer(dim=512, num_heads=8)
        >>> x = torch.randn(2, 128, 512)
        >>> out = layer(x)
        >>> assert out.shape == (2, 128, 512)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        chunk_size: int = 64,
        eps: float = 1e-6,
        rope: Optional[object] = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope = rope

        # Projections for Q, K, V, forget gate, and output gate
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.beta_proj = nn.Linear(dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.kda = KimiDeltaAttention(chunk_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters for stable training.

        Q/K/V/O projections use Xavier uniform initialization.
        Beta projection uses small normal init (sigmoid -> ~0.5 forget
 rate).
        All projections are marked with ``_custom_init`` to prevent
        ``KimiTransformer._init_weights`` from overwriting them.
        """
        for m in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.xavier_uniform_(m.weight)
            setattr(m, "_custom_init", True)
        nn.init.normal_(self.beta_proj.weight, mean=0.0, std=0.02)
        setattr(self.beta_proj, "_custom_init", True)
        nn.init.xavier_uniform_(self.g_proj.weight)
        setattr(self.g_proj, "_custom_init", True)

    def clear_recurrent_state(self) -> None:
        """Reset the KDA recurrent memory for fresh generation."""
        self.kda._recurrent_state = None

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``[B, T, D]``.
            positions: Position ids ``[B, T]`` for RoPE.

        Returns:
            Output tensor ``[B, T, D]``.
        """
        B, T, D = x.shape

        # Project and reshape to multi-head format
        q = (
            self.q_proj(x)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply RoPE to Q and K
        if self.rope is not None and positions is not None:
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        beta = (
            self.beta_proj(x)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        # Output gate with SiLU activation
        g = (
            F.silu(self.g_proj(x))
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # KDA computation
        attn_out = self.kda(q, k, v, beta, g)

        # Merge heads and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(attn_out)
