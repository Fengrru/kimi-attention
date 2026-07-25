"""
Attention Residuals (AttnRes)
=============================
Industrial-strength implementation of the Attention Residuals mechanism
introduced by Moonshot AI in the Kimi model series.

Attention Residuals replace traditional residual connections with a
depth-wise attention mechanism that dynamically aggregates information
from previous layers. This addresses the dilution problem in deep
Transformers where early-layer signals are progressively attenuated.

Core Idea
---------
Traditional residual::

    x_{l+1} = x_l + F(x_l)

All preceding layers contribute with fixed equal weights, causing deep-layer
information to be diluted by shallow-layer signals.

Attention Residuals::

    h_l = sum_i alpha_{i->l} * v_i

where ``alpha_{i->l} = softmax(w_l^T * RMSNorm(v_i))``. Each layer uses a
learned pseudo-query vector ``w_l`` to dynamically select and weight
information from all previous layers.

Block-wise Strategy
-------------------
To control memory complexity, layers are grouped into blocks:
    - **Intra-block**: Standard residual connections (compute-efficient)
    - **Inter-block**: Attention-based aggregation (preserves deep signals)

Memory complexity: O(N * d) where N = num_layers / layers_per_block.

Reference:
    Moonshot AI (2025). Attention Residuals for Deep Transformer Networks.
    arXiv:2603.15031.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kimi_attention.models.rmsnorm import RMSNorm


class BlockAttentionResiduals(nn.Module):
    """Block Attention Residuals (AttnRes) for deep Transformers.

        This module implements depth-wise attention aggregation that can be
    drop-in
        inserted into any existing Transformer architecture with zero modifications
    to
        the underlying attention or FFN sublayers.

        Args:
            dim: Model hidden dimension (``d_model``).
            layers_per_block: Number of Transformer layers per block. Within a
                block, standard residuals are used. Between blocks, attention
                residuals aggregate all preceding block outputs. Default: 4.
            eps: Numerical stability constant for RMSNorm. Default: 1e-6.

        Attributes:
            attn_res_proj: Pseudo-query projection for attention residual path.
            mlp_res_proj: Pseudo-query projection for FFN residual path.

        Example::
            >>> attn_res = BlockAttentionResiduals(dim=512, layers_per_block=4)
            >>> blocks, hidden = [], torch.randn(2, 16, 512)
            >>> for idx in range(8):
            ...     blocks, hidden = attn_res(
            ...         blocks=blocks,
            ...         hidden_states=hidden,
            ...         layer_number=idx,
            ...         attn_fn=lambda x: x * 0.1,
            ...         mlp_fn=lambda x: x * 0.1,
            ...         attn_norm=RMSNorm(512),
            ...         mlp_norm=RMSNorm(512),
            ...     )
    """

    def __init__(
        self,
        dim: int,
        layers_per_block: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.layers_per_block = layers_per_block

        # Independent pseudo-query projections for attention and FFN paths
        self.attn_res_proj = nn.Linear(dim, 1, bias=False)
        self.attn_res_norm = RMSNorm(dim, eps)
        self.mlp_res_proj = nn.Linear(dim, 1, bias=False)
        self.mlp_res_norm = RMSNorm(dim, eps)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize pseudo-query projections with small values.

        Small initializations produce near-uniform attention distributions
        at the start of training, ensuring stability during warm-up.
        """
        nn.init.normal_(self.attn_res_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.mlp_res_proj.weight, mean=0.0, std=0.02)

    def _attn_res_aggregate(
        self,
        blocks: List[torch.Tensor],
        current: torch.Tensor,
        proj: nn.Linear,
        norm: RMSNorm,
    ) -> torch.Tensor:
        """Core attention residual aggregation.

               Computes a weighted sum of all previous block outputs plus the
        current
               layer's input, where weights are determined by a pseudo-query
        attention
               mechanism over the depth dimension.

               Args:
                   blocks: List of completed block outputs, each ``[B, T, D]``.
                   current: Current layer's input, ``[B, T, D]``.
                   proj: Pseudo-query projection layer.
                   norm: RMSNorm applied before computing attention scores.

               Returns:
                   Aggregated hidden state of shape ``[B, T, D]``.

               Note:
                   When ``blocks`` is empty, this degenerates to the identity
        function
                   (returns ``current`` unchanged), ensuring correct behavior at
        layer 0.
        """
        if not blocks:
            # Degenerate case: no previous blocks → identity mapping
            return current

        # Stack all previous blocks + current state: [N+1, B, T, D]
        V = torch.stack(blocks + [current], dim=0)
        K = norm(V)

        # Pseudo-query dot-product attention over depth dimension
        # logits: [N+1, B, T]
        logits = torch.einsum("d, n b t d -> n b t", proj.weight.squeeze(), K)
        attn_weights = F.softmax(logits, dim=0)

        # Weighted aggregation: [B, T, D]
        h = torch.einsum("n b t, n b t d -> b t d", attn_weights, V)
        return h

    def _attn_res_aggregate_step(
        self,
        blocks: List[torch.Tensor],
        current: torch.Tensor,
        proj: nn.Linear,
        norm: RMSNorm,
    ) -> torch.Tensor:
        """Single‑token aggregation for incremental generation.

        Blocks have shape ``[B, T_full, D]`` while current is
        ``[B, 1, D]``.  Each block contributes through its last
        position (the most recent context).

        Args:
            blocks: Previously completed block outputs ``[B, T, D]``.
            current: Single‑token input ``[B, 1, D]``.
            proj: Pseudo-query projection.
            norm: RMSNorm.

        Returns:
            Aggregated hidden state ``[B, 1, D]``.
        """
        if not blocks:
            return current

        # Represent each block by its last position
        block_repr = torch.stack([b[:, -1:, :] for b in blocks], dim=0)
        V = torch.cat([block_repr, current.unsqueeze(0)], dim=0)
        K = norm(V)

        logits = torch.einsum("d, n b t d -> n b t", proj.weight.squeeze(), K)
        attn_weights = F.softmax(logits, dim=0)

        return torch.einsum("n b t, n b t d -> b t d", attn_weights, V)

    def step(
        self,
        blocks: List[torch.Tensor],
        hidden_states: torch.Tensor,
        layer_number: int,
        attn_fn: Callable[[torch.Tensor], torch.Tensor],
        mlp_fn: Callable[[torch.Tensor], torch.Tensor],
        attn_norm: RMSNorm,
        mlp_norm: RMSNorm,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Single‑step forward for incremental generation.

        Unlike :meth:`forward`, this method uses ``_aggregate_step``
        (last‑position block aggregation) so that blocks stored from
        the prefill can be combined with a single‑token hidden state.

        No new blocks are appended during generation.
        """
        layer_in_block = layer_number % self.layers_per_block
        is_block_start = layer_in_block == 0

        if is_block_start and len(blocks) > 0:
            h_attn = self._attn_res_aggregate_step(
                blocks, hidden_states, self.attn_res_proj, self.attn_res_norm
            )
        else:
            h_attn = hidden_states

        attn_out = attn_fn(attn_norm(h_attn))
        h = h_attn + attn_out

        if is_block_start and len(blocks) > 0:
            h_mlp = self._attn_res_aggregate_step(blocks, h, self.mlp_res_proj, self.mlp_res_norm)
        else:
            h_mlp = h

        mlp_out = mlp_fn(mlp_norm(h_mlp))
        return blocks, h + mlp_out

    def forward(
        self,
        blocks: List[torch.Tensor],
        hidden_states: torch.Tensor,
        layer_number: int,
        attn_fn: Callable[[torch.Tensor], torch.Tensor],
        mlp_fn: Callable[[torch.Tensor], torch.Tensor],
        attn_norm: RMSNorm,
        mlp_norm: RMSNorm,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Execute one Transformer layer with integrated Attention Residuals.

                This method wraps both the attention and FFN sublayers, applying
                attention residuals at block boundaries while using standard
        residuals
                within blocks.

                Args:
                    blocks: Global list of completed block outputs.
                    hidden_states: Current layer input ``[B, T, D]``.
                    layer_number: 0-based layer index.
                    attn_fn: Attention computation callable. Accepts normalized
                        input ``[B, T, D]`` and returns attention output ``[B, T,
        D]``.
                    mlp_fn: FFN computation callable. Accepts normalized input
                        ``[B, T, D]`` and returns FFN output ``[B, T, D]``.
                    attn_norm: Pre-attention RMSNorm layer.
                    mlp_norm: Pre-FFN RMSNorm layer.

                Returns:
                    Tuple of (updated_blocks, layer_output) where ``layer_output``
                    is ``[B, T, D]``.

                Note:
                    On block boundaries (every ``layers_per_block`` layers), this
                    module appends the current output to ``blocks`` after
        computation.
                    These stored block outputs are detached during training to
        prevent
                    gradient flow through historical layers.
        """
        layer_in_block = layer_number % self.layers_per_block
        is_block_start = layer_in_block == 0

        # ---- 1. Attention sublayer ----
        if is_block_start and len(blocks) > 0:
            # At block start: aggregate all previous blocks via AttnRes
            h_attn = self._attn_res_aggregate(
                blocks, hidden_states, self.attn_res_proj, self.attn_res_norm
            )
        else:
            # Within block: standard residual path
            h_attn = hidden_states

        attn_out = attn_fn(attn_norm(h_attn))
        h = h_attn + attn_out  # Standard residual

        # ---- 2. FFN sublayer ----
        if is_block_start and len(blocks) > 0:
            h_mlp = self._attn_res_aggregate(blocks, h, self.mlp_res_proj, self.mlp_res_norm)
        else:
            h_mlp = h

        mlp_out = mlp_fn(mlp_norm(h_mlp))
        output = h + mlp_out  # Standard residual

        # ---- 3. Block boundary: store completed block ----
        if layer_in_block == self.layers_per_block - 1:
            # Detach to prevent backprop through history (per paper)
            blocks.append(output.detach() if self.training else output)

        return blocks, output
