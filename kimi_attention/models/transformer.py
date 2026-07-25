"""
Kimi Transformer
================
End-to-end Transformer architecture integrating Attention Residuals
and Kimi Delta Attention in a hybrid configuration.

Architecture Overview
---------------------
The default configuration follows Moonshot AI's official 3:1 hybrid
ratio:
    - 75% of layers use KDA (efficient linear attention)
    - 25% of layers use standard Multi-Head Attention (full precision)

This design achieves:
    - **3-6x speedup** on long-context decoding (1M+ tokens)
    - **75% KV cache reduction** vs. full standard attention
    - **Comparable or better accuracy** on standard benchmarks

Configuration Presets
---------------------
    - 1B params:  24 layers, d_model=2048,  8 heads
    - 7B params:  32 layers, d_model=4096,  32 heads
    - 48B params: 64 layers, d_model=8192,  64 heads (official Kimi-Linear)

Example::
    >>> config = KimiConfig(dim=512, num_layers=8, num_heads=8)
    >>> model = KimiTransformer(config)
    >>> input_ids = torch.randint(0, config.vocab_size, (2, 32))
    >>> logits = model(input_ids)
    >>> print(logits.shape)  # [2, 32, vocab_size]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kimi_attention.models.attention_residuals import BlockAttentionResiduals
from kimi_attention.models.delta_attention import KimiDeltaAttentionLayer
from kimi_attention.models.rmsnorm import RMSNorm
from kimi_attention.models.rope import RotaryEmbedding


@dataclass
class KimiConfig:
    """Configuration dataclass for Kimi Transformer models.

    This follows the Hugging Face configuration pattern for easy
    integration with their ecosystem.

    Attributes:
        dim: Model hidden dimension (d_model).
        num_layers: Total number of Transformer layers.
        num_heads: Number of attention heads.
        mlp_ratio: FFN hidden dim multiplier. Default: 4.0.
        vocab_size: Size of the token vocabulary. Default: 32000.
        max_seq_len: Maximum sequence length for position embeddings.
            Default: 4096.
        layers_per_block: Layers per AttnRes block. Default: 4.
        kda_every: Insert standard attention every N layers (0 = always
            KDA). Default: 4 (3:1 KDA:MHA ratio).
        chunk_size: FLA kernel chunk size. Default: 64.
        eps: RMSNorm numerical stability constant. Default: 1e-6.
        dropout: Dropout rate. Default: 0.0.
        tie_weights: Whether to tie embedding and LM head weights.
            Default: True.
    """

    dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    num_kv_heads: int = 0
    mlp_ratio: float = 4.0
    vocab_size: int = 32000
    max_seq_len: int = 4096
    layers_per_block: int = 4
    kda_every: int = 4
    chunk_size: int = 64
    eps: float = 1e-6
    dropout: float = 0.0
    tie_weights: bool = True
    rope_theta: float = 10000.0
    num_experts: int = 0
    num_experts_per_tok: int = 2

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.dim // self.num_heads

    @property
    def kv_heads(self) -> int:
        """Number of KV heads (GQA).  Falls back to num_heads when 0."""
        return self.num_kv_heads if self.num_kv_heads > 0 else self.num_heads

    @property
    def kv_head_dim(self) -> int:
        """Dimension per KV head."""
        return self.dim // self.kv_heads

    @property
    def mlp_dim(self) -> int:
        """FFN intermediate dimension (per expert when MoE is active)."""
        return int(self.dim * self.mlp_ratio)

    @classmethod
    def from_size(cls, size: str) -> "KimiConfig":
        """Create a config from a named model size.

        Args:
            size: One of "1B", "7B", "48B".

        Returns:
            Pre-configured KimiConfig instance.

        Note:
            These presets match ``kimi_attention.configs.model_configs``.
        """
        presets = {
            "1B": dict(
                dim=2048, num_layers=24, num_heads=8,
                vocab_size=32000, max_seq_len=32768,
            ),
            "7B": dict(
                dim=4096, num_layers=32, num_heads=32,
                num_kv_heads=8, vocab_size=64000, max_seq_len=131072,
                rope_theta=500000.0,
            ),
            "48B": dict(
                dim=8192, num_layers=64, num_heads=64,
                num_kv_heads=8, vocab_size=128000, max_seq_len=1048576,
                rope_theta=1000000.0, num_experts=8, num_experts_per_tok=2,
            ),
        }
        if size not in presets:
            raise ValueError(f"Unknown size: {size}. Choose from {list(presets.keys())}")
        return cls(**presets[size])


class TransformerBlock(nn.Module):
    """Single Transformer layer with RoPE, GQA, MoE, and KDA.

    Args:
        config: Model configuration.
        layer_idx: Layer index (0-based).
        use_kda: If True, use KimiDeltaAttentionLayer; otherwise GQA+MHA.
        rope: Shared rotary embedding module (applied to Q and K).
    """

    def __init__(
        self,
        config: KimiConfig,
        layer_idx: int,
        use_kda: bool = True,
        rope: Optional["RotaryEmbedding"] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.use_kda = use_kda
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.kv_heads
        self.head_dim = config.head_dim
        self.kv_head_dim = config.kv_head_dim
        self.rope = rope

        # Attention sublayer
        if use_kda:
            self.attn = KimiDeltaAttentionLayer(
                dim=config.dim,
                num_heads=config.num_heads,
                chunk_size=config.chunk_size,
                eps=config.eps,
                rope=rope,
            )
        else:
            self.q_proj = nn.Linear(config.dim, config.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(config.dim, config.kv_heads * self.kv_head_dim, bias=False)
            self.v_proj = nn.Linear(config.dim, config.kv_heads * self.kv_head_dim, bias=False)
            self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.dim, bias=False)

        # FFN sublayer
        if config.num_experts > 0:
            from kimi_attention.models.moe import MoEFeedForward

            self.mlp = MoEFeedForward(
                dim=config.dim,
                hidden_dim=config.mlp_dim,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
            )
            self._has_moe = True
        else:
            mlp_hidden = config.mlp_dim
            self.fc1 = nn.Linear(config.dim, mlp_hidden, bias=False)
            self.fc2 = nn.Linear(config.dim, mlp_hidden, bias=False)
            self.fc3 = nn.Linear(mlp_hidden, config.dim, bias=False)
            self._has_moe = False

        # Normalization
        self.attn_norm = RMSNorm(config.dim, config.eps)
        self.mlp_norm = RMSNorm(config.dim, config.eps)

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Flash Attention (lazy import)
        self._flash_attn_fn = None
        self._flash_checked = False

        self._init_weights()

    def _try_flash_attn(self) -> bool:
        if self._flash_checked:
            return self._flash_attn_fn is not None
        self._flash_checked = True
        try:
            from flash_attn import flash_attn_func  # type: ignore

            self._flash_attn_fn = flash_attn_func
        except Exception:
            pass
        return self._flash_attn_fn is not None

    def _attn_sdpa(self, q, k, v, is_causal: bool = True):
        """Attention dispatch: FlashAttn on CUDA else PyTorch SDPA."""
        if q.is_cuda and self._try_flash_attn():
            return self._flash_attn_fn(q, k, v, causal=is_causal)
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    def _init_weights(self) -> None:
        if self.use_kda:
            return
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.xavier_uniform_(proj.weight)
            setattr(proj, "_custom_init", True)

    def attn_forward(
        self,
        x: torch.Tensor,
        normed: bool = False,
        positions: Optional[torch.Tensor] = None,
        store_cache: bool = False,
    ) -> torch.Tensor:
        """Attention sublayer (GQA with RoPE).

        Args:
            x: Input ``[B, T, D]``.
            normed: If True, ``x`` is already RMSNorm'd.  Set when called
                from ``attn_res`` which applies its own normalization.
            positions: Position ids for RoPE ``[B, T]``.
            store_cache: Store K/V tensors for incremental decoding.
        """
        if not normed:
            x = self.attn_norm(x)
        B, T, D_in = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)

        if positions is not None:
            q, k = self._apply_rope(q, k, positions)

        if store_cache:
            self._kv_cache = {"k": k, "v": v}

        attn_out = self._attn_sdpa(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_out)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        """Apply RoPE if the module is configured."""
        if self.rope is None:
            return q, k
        q = self.rope(q, positions)
        k = self.rope(k, positions)
        return q, k

    def mlp_forward(self, x: torch.Tensor, normed: bool = False) -> torch.Tensor:
        """FFN sublayer (SwiGLU or MoE).

        Args:
            x: Input ``[B, T, D]``.
            normed: If True, ``x`` is already RMSNorm'd.
        """
        if not normed:
            x = self.mlp_norm(x)
        if self._has_moe:
            result = self.mlp(x, return_balance_loss=self.training)
            if self.training:
                mlp_out, self._last_balance_loss = result
            else:
                mlp_out = result
                self._last_balance_loss = 0.0
            return mlp_out
        gate = F.silu(self.fc1(x))
        hidden = gate * self.fc2(x)
        return self.fc3(hidden)

    def attn_incremental(
        self,
        x: torch.Tensor,
        normed: bool = False,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single‑token attention forward with KV‑cache for incremental decoding.

        Args:
            x: Input ``[B, 1, D]``.
            normed: If True, ``x`` is already RMSNorm'd.
            positions: Position ids ``[B, 1]`` for RoPE.

        Returns:
            Output ``[B, 1, D]``.
        """
        if not normed:
            x = self.attn_norm(x)
        B, T, D_in = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)

        if positions is not None:
            q, k = self._apply_rope(q, k, positions)

        had_cache = hasattr(self, "_kv_cache") and self._kv_cache is not None
        if had_cache:
            k = torch.cat([self._kv_cache["k"], k], dim=2)
            v = torch.cat([self._kv_cache["v"], v], dim=2)

        self._kv_cache = {"k": k, "v": v}

        attn_out = self._attn_sdpa(q, k, v, is_causal=not had_cache)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_out)

    def clear_cache(self) -> None:
        """Reset KV cache and KDA recurrent state."""
        if hasattr(self, "_kv_cache"):
            self._kv_cache = None
        if self.use_kda:
            self.attn.clear_recurrent_state()

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full block: attention residual + FFN residual.

        Args:
            x: Input ``[B, T, D]``.
            use_cache: Store KV cache / recurrent state for decode.
            positions: Position ids ``[B, T]`` for RoPE (auto‑generated
                if None and T > 1).

        Returns:
            Output ``[B, T, D]``.
        """
        if self.use_kda:
            h = x + self.dropout(self.attn(self.attn_norm(x), positions=positions))
        else:
            normed = self.attn_norm(x)
            B, T, D = normed.shape

            if positions is None:
                positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

            q = (
                self.q_proj(normed)
                .view(B, T, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = (
                self.k_proj(normed)
                .view(B, T, self.num_kv_heads, self.kv_head_dim)
                .transpose(1, 2)
            )
            v = (
                self.v_proj(normed)
                .view(B, T, self.num_kv_heads, self.kv_head_dim)
                .transpose(1, 2)
            )

            q, k = self._apply_rope(q, k, positions)

            had_cache = use_cache and hasattr(self, "_kv_cache") and self._kv_cache is not None
            if had_cache:
                k = torch.cat([self._kv_cache["k"], k], dim=2)
                v = torch.cat([self._kv_cache["v"], v], dim=2)

            if use_cache:
                self._kv_cache = {"k": k, "v": v}

            attn_out = self._attn_sdpa(q, k, v, is_causal=not had_cache)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
            h = x + self.dropout(self.o_proj(attn_out))

        normed = self.mlp_norm(h)
        self._last_balance_loss = 0.0
        if self._has_moe:
            result = self.mlp(normed, return_balance_loss=self.training)
            if self.training:
                mlp_out, self._last_balance_loss = result
            else:
                mlp_out = result
                self._last_balance_loss = 0.0
        else:
            gate = F.silu(self.fc1(normed))
            hidden = gate * self.fc2(normed)
            mlp_out = self.fc3(hidden)

        return h + self.dropout(mlp_out)

    def get_balance_loss(self) -> "torch.Tensor":
        """Return the MoE load‑balancing loss from the last forward."""
        val = getattr(self, "_last_balance_loss", 0.0)
        if isinstance(val, float):
            val = torch.tensor(val, dtype=torch.float32)
        return val

    def get_kv_cache_state(self) -> dict:
        """Clone MHA KV cache for beam‑search snapshotting."""
        state: dict = {}
        if (
            not self.use_kda
            and hasattr(self, "_kv_cache")
            and self._kv_cache is not None
        ):
            state = {k: v.clone() for k, v in self._kv_cache.items()}
        return state

    def set_kv_cache_state(self, state: dict) -> None:
        """Restore a previously cloned MHA KV cache."""
        if self.use_kda:
            return
        if state:
            self._kv_cache = {k: v.clone() for k, v in state.items()}
        else:
            self._kv_cache = None


class KimiTransformer(nn.Module):
    """Complete Kimi Transformer with Attention Residuals and hybrid KDA.

    This model integrates:
        1. **Attention Residuals**: Dynamic depth-wise aggregation via
           BlockAttentionResiduals
        2. **Hybrid KDA**: 3:1 ratio of KDA to standard attention layers
        3. **SwiGLU FFN**: Gated activation for improved expressiveness

    Args:
        config: KimiConfig instance specifying model architecture.

    Attributes:
        token_emb: Token embedding lookup table.
        pos_emb: Learned positional embeddings.
        attn_res: BlockAttentionResiduals module.
        layers: ModuleList of TransformerBlock layers.
        final_norm: Final RMSNorm before LM head.
        lm_head: Output projection to vocabulary.

    Example::
        >>> config = KimiConfig.from_size("1B")
        >>> model = KimiTransformer(config)
        >>> ids = torch.randint(0, config.vocab_size, (1, 128))
        >>> logits = model(ids)
        >>> print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    """

    def __init__(self, config: KimiConfig) -> None:
        super().__init__()
        self.config = config

        # RoPE
        self.rope = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

        # Attention Residuals
        self.attn_res = BlockAttentionResiduals(
            dim=config.dim,
            layers_per_block=config.layers_per_block,
            eps=config.eps,
        )

        # Transformer layers with hybrid KDA/MHA configuration
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            use_kda = (
                config.kda_every <= 0 or (i % config.kda_every) != config.kda_every - 1
            )
            self.layers.append(
                TransformerBlock(
                    config, layer_idx=i, use_kda=use_kda, rope=self.rope
                )
            )

        # Output
        self.final_norm = RMSNorm(config.dim, config.eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following standard LLM practices.

        Skips linear layers whose parent is ``KimiDeltaAttentionLayer``
        (those layers manage their own initialization via
        ``_reset_parameters``).
        """
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, "_custom_init"):
                    continue
                # Do not re-initialise KDA internal projections
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Embedding) and module is not self.token_emb:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass computing logits for next-token prediction.

        Args:
            input_ids: Integer token indices ``[B, T]``.

        Returns:
            Logits tensor ``[B, T, vocab_size]``.
        """
        B, T = input_ids.shape
        if T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len "
                f"{self.config.max_seq_len}"
            )

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_emb(input_ids)
        x = self.dropout(x)

        # Reset KDA recurrent states for a fresh training forward
        if self.training:
            for layer in self.layers:
                if layer.use_kda:
                    layer.attn.clear_recurrent_state()

        blocks: List[torch.Tensor] = []
        hidden = x

        for layer_idx, layer in enumerate(self.layers):
            def make_attn_fn(lyr: TransformerBlock):
                if lyr.use_kda:
                    # h is already normalized by attn_res.forward()
                    return lambda h: lyr.attn(h, positions=positions)
                else:
                    # h is already normalized; pass positions for RoPE
                    return lambda h: lyr.attn_forward(h, normed=True, positions=positions)

            def make_mlp_fn(lyr: TransformerBlock):
                # h is already normalized by attn_res.forward()
                return lambda h: lyr.mlp_forward(h, normed=True)

            blocks, hidden = self.attn_res(
                blocks=blocks,
                hidden_states=hidden,
                layer_number=layer_idx,
                attn_fn=make_attn_fn(layer),
                mlp_fn=make_mlp_fn(layer),
                attn_norm=layer.attn_norm,
                mlp_norm=layer.mlp_norm,
            )

        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        return logits

    def clear_caches(self) -> None:
        """Clear all KV caches across layers."""
        for layer in self.layers:
            layer.clear_cache()

    def _snapshot_layer_caches(self) -> list:
        """Return per‑layer KV‑cache clones (one dict per layer)."""
        return [lyr.get_kv_cache_state() for lyr in self.layers]

    def _restore_layer_caches(self, caches: list) -> None:
        """Restore per‑layer KV‑cache clones."""
        for lyr, cache in zip(self.layers, caches):
            lyr.set_kv_cache_state(cache)

    def get_total_balance_loss(self) -> torch.Tensor:
        """Sum MoE load‑balancing losses across all layers."""
        total = torch.tensor(0.0, device=self.lm_head.weight.device, dtype=torch.float32)
        for layer in self.layers:
            total = total + layer.get_balance_loss().to(total.device)
        return total

    @torch.no_grad()
    def generate_beam(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Beam‑search generation with per‑beam KV‑cache (incremental).

        .. warning::
            KDA layers re‑process the full sequence each step (they do
            not maintain recurrent state across calls).  MHA layers use
            per‑beam KV caches for incremental O(T) decoding.

        Args:
            input_ids: Prompt ``[B, T]`` (batch=1 currently required).
            max_new_tokens: Maximum tokens to generate.
            num_beams: Number of beams.
            length_penalty: α for score = log_prob / (len^α).  Applied
                **only at final selection** (not accumulated per step).
            early_stopping: Stop when all beams hit EOS.
            eos_token_id: End‑of‑sentence token.

        Returns:
            Best sequence ``[1, T + generated]``.
        """
        if input_ids.size(0) != 1:
            raise ValueError("Beam search currently supports batch_size=1")

        self.eval()
        self.clear_caches()
        device = input_ids.device
        B, T = input_ids.shape
        Vocab = self.config.vocab_size

        # ── Prefill: process prompt once ──────────────────────────
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_emb(input_ids)
        x = self.dropout(x)
        hidden = x
        for layer in self.layers:
            hidden = layer.forward(hidden, use_cache=True, positions=positions)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        first_logits = F.log_softmax(logits[:, -1, :], dim=-1)  # [1, V]

        # Snapshot the prompt KV‑cache before any beam‑specific changes
        prompt_cache = self._snapshot_layer_caches()

        # ── Expand to num_beams ───────────────────────────────────
        top_scores, top_tokens = torch.topk(first_logits[0], num_beams)
        beam_scores = top_scores.clone()            # [B] raw log‑prob sum
        beam_tokens = top_tokens                    # [B]

        # Each beam holds its own list-of-tensors sequence (not padded)
        beam_seqs: List[torch.Tensor] = [
            torch.cat([input_ids[0], beam_tokens[i : i + 1]]) for i in range(num_beams)
        ]
        beam_done = [False] * num_beams

        # ── Build per‑beam KV caches ──────────────────────────────
        # Each beam needs its own cache that includes the prompt AND
        # that beam's first generated token.
        beam_caches: List[List[dict]] = []
        for i in range(num_beams):
            self.clear_caches()
            self._restore_layer_caches(prompt_cache)
            beam_caches.append(self._snapshot_layer_caches())
            # Incremental: forward only the beam's first new token
            tok = beam_tokens[i : i + 1].unsqueeze(0)          # [1, 1]
            pos = torch.tensor([[T]], device=device)           # position = prompt length
            h = self.token_emb(tok)
            h = self.dropout(h)
            for layer in self.layers:
                h = layer.forward(h, use_cache=True, positions=pos)

        # ── Generation loop ──────────────────────────────────────
        for step in range(1, max_new_tokens):
            if early_stopping and all(beam_done):
                break

            all_logits: List[torch.Tensor] = []
            for i in range(num_beams):
                if beam_done[i]:
                    all_logits.append(
                        torch.full((Vocab,), float("-inf"), device=device)
                    )
                    continue

                self.clear_caches()
                self._restore_layer_caches(beam_caches[i])

                # Single‑token incremental forward
                last_tok = beam_seqs[i][-1:].unsqueeze(0)          # [1, 1]
                pos = torch.tensor(
                    [[beam_seqs[i].size(0) - 1]], device=device,
                )
                h = self.token_emb(last_tok)
                h = self.dropout(h)
                for layer in self.layers:
                    h = layer.forward(h, use_cache=True, positions=pos)
                h = self.final_norm(h)
                nxt = F.log_softmax(self.lm_head(h[:, -1, :]), dim=-1)
                all_logits.append(nxt[0])  # [V]

                # Update cache snapshot (now includes the last token)
                beam_caches[i] = self._snapshot_layer_caches()

            next_logits = torch.stack(all_logits)  # [B, V]

            # Candidate scores (raw sum — NO length penalty here)
            candidate_scores = beam_scores.unsqueeze(1) + next_logits  # [B, V]

            # Mask finished beams
            for i in range(num_beams):
                if beam_done[i]:
                    candidate_scores[i] = float("-inf")

            flat = candidate_scores.view(-1)
            top_scores, top_indices = torch.topk(flat, num_beams)

            parents = top_indices // Vocab
            tokens = top_indices % Vocab

            # ── Update beams ─────────────────────────────────────
            new_seqs: List[torch.Tensor] = []
            new_scores: List[torch.Tensor] = []
            new_done: List[bool] = []
            new_caches: List[List[dict]] = []

            for rank in range(num_beams):
                p = parents[rank].item()
                t = tokens[rank].item()

                if p == rank:
                    # Same parent — reuse already‑updated cache
                    c = beam_caches[p]
                else:
                    # Different parent — forward new token on parent's cache
                    self.clear_caches()
                    self._restore_layer_caches(beam_caches[p])
                    tok = torch.tensor([[t]], device=device)
                    pos = torch.tensor(
                        [[beam_seqs[p].size(0)]], device=device,
                    )
                    h = self.token_emb(tok)
                    h = self.dropout(h)
                    for layer in self.layers:
                        h = layer.forward(h, use_cache=True, positions=pos)
                    c = self._snapshot_layer_caches()

                new_seqs.append(
                    torch.cat([beam_seqs[p], torch.tensor([t], device=device)])
                )
                new_scores.append(top_scores[rank])
                new_done.append(
                    beam_done[p]
                    or (eos_token_id is not None and t == eos_token_id)
                )
                new_caches.append(c)

            beam_seqs = new_seqs
            beam_scores = torch.stack(new_scores)
            beam_done = new_done
            beam_caches = new_caches

        # ── Final selection (length penalty applied ONCE) ─────────
        if length_penalty != 1.0:
            lengths = torch.tensor(
                [s.size(0) for s in beam_seqs], device=device, dtype=torch.float32,
            )
            final_scores = beam_scores / (lengths ** length_penalty)
        else:
            final_scores = beam_scores

        if any(beam_done):
            mask = final_scores.clone()
            for i in range(num_beams):
                if not beam_done[i]:
                    mask[i] = float("-inf")
            best = mask.argmax().item()
        else:
            best = final_scores.argmax().item()

        return beam_seqs[best].unsqueeze(0)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with AttnRes + KV‑cache + KDA recurrent state.

        Prefill processes the full prompt through
        :meth:`BlockAttentionResiduals.forward`; each subsequent token runs
        through :meth:`BlockAttentionResiduals.step` which uses last‑position
        block aggregation for efficient incremental decoding.

        Args:
            input_ids: Initial token indices ``[B, T]``.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.  Default: 1.0.
            top_k: If set, only sample from the top‑k logits.
            top_p: If set, use nucleus sampling.
            eos_token_id: Token that signals end of generation.

        Returns:
            Generated token IDs ``[B, T + generated]``.
        """
        self.eval()
        self.clear_caches()

        generated = input_ids.clone()
        B, T = input_ids.shape
        device = input_ids.device

        def _sample(logits):
            nxt = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(nxt, min(top_k, nxt.size(-1)))
                nxt[nxt < v[:, [-1]]] = float("-inf")
            if top_p is not None:
                s_logits, s_idx = torch.sort(nxt, descending=True, dim=-1)
                cum = torch.cumsum(F.softmax(s_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[:, 0] = False
                for b in range(nxt.size(0)):
                    nxt[b, s_idx[b][remove[b]]] = float("-inf")
            probs = F.softmax(nxt, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        # ── Prefill through AttnRes ──────────────────────────────
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids)
        x = self.dropout(x)

        blocks: List[torch.Tensor] = []
        hidden = x

        for layer_idx, layer in enumerate(self.layers):
            def make_attn_fn(lyr: TransformerBlock):
                if lyr.use_kda:
                    # KDA handles its own RoPE via the ``rope`` module
                    return lambda h: lyr.attn(h, positions=positions)
                else:
                    return lambda h: lyr.attn_forward(
                        h, normed=True, positions=positions, store_cache=True
                    )

            def make_mlp_fn(lyr: TransformerBlock):
                return lambda h: lyr.mlp_forward(h, normed=True)

            blocks, hidden = self.attn_res(
                blocks=blocks,
                hidden_states=hidden,
                layer_number=layer_idx,
                attn_fn=make_attn_fn(layer),
                mlp_fn=make_mlp_fn(layer),
                attn_norm=layer.attn_norm,
                mlp_norm=layer.mlp_norm,
            )

        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)

        # ── Decode loop ──────────────────────────────────────────
        pos = T
        for _ in range(max_new_tokens):
            if pos >= self.config.max_seq_len:
                break

            next_token = _sample(logits)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            pos_tensor = torch.tensor([[pos]], device=device).expand(B, -1)
            x = self.token_emb(next_token)
            x = self.dropout(x)
            hidden = x

            for layer_idx, layer in enumerate(self.layers):
                def make_attn_fn_step(lyr: TransformerBlock):
                    if lyr.use_kda:
                        return lambda h: lyr.attn(h, positions=pos_tensor)
                    else:
                        return lambda h: lyr.attn_incremental(
                            h, normed=True, positions=pos_tensor
                        )

                def make_mlp_fn_step(lyr: TransformerBlock):
                    return lambda h: lyr.mlp_forward(h, normed=True)

                blocks, hidden = self.attn_res.step(
                    blocks=blocks,
                    hidden_states=hidden,
                    layer_number=layer_idx,
                    attn_fn=make_attn_fn_step(layer),
                    mlp_fn=make_mlp_fn_step(layer),
                    attn_norm=layer.attn_norm,
                    mlp_norm=layer.mlp_norm,
                )

            hidden = self.final_norm(hidden)
            logits = self.lm_head(hidden)
            pos += 1

        return generated

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_memory(self, batch_size: int = 1, seq_len: int = 2048) -> dict:
        """Estimate memory usage for a given input configuration.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Dictionary with memory estimates in MB.
        """
        param_bytes = self.count_parameters() * 4  # FP32
        num_mha_layers = sum(1 for ly in self.layers if not ly.use_kda)
        kv_cache_bytes = (
            2  # K and V
            * batch_size
            * seq_len
            * self.config.kv_heads
            * self.config.kv_head_dim
            * num_mha_layers
            * 4
        )
        activation_bytes = (
            batch_size
            * seq_len
            * self.config.dim
            * self.config.num_layers
            * 4
        )

        return {
            "parameters_mb": param_bytes / (1024 ** 2),
            "kv_cache_mb": kv_cache_bytes / (1024 ** 2),
            "activations_mb": activation_bytes / (1024 ** 2),
            "total_mb": (param_bytes + kv_cache_bytes + activation_bytes)
            / (1024 ** 2),
        }
