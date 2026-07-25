"""
kimi-attention: Production-grade PyTorch implementation of
Kimi's revolutionary attention mechanisms.

Modules:
    models: Core neural network architectures
    utils: Training utilities, logging, and helpers
    configs: Model and training configurations
"""

__version__ = "1.0.0"

from kimi_attention.models.attention_residuals import BlockAttentionResiduals
from kimi_attention.models.delta_attention import (
    KimiDeltaAttention,
    KimiDeltaAttentionLayer,
)
from kimi_attention.models.moe import MoEFeedForward
from kimi_attention.models.rmsnorm import RMSNorm
from kimi_attention.models.rope import RotaryEmbedding
from kimi_attention.models.transformer import (
    KimiConfig,
    KimiTransformer,
    TransformerBlock,
)

__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "BlockAttentionResiduals",
    "KimiDeltaAttention",
    "KimiDeltaAttentionLayer",
    "MoEFeedForward",
    "TransformerBlock",
    "KimiTransformer",
    "KimiConfig",
]
