"""
Model Configuration Presets
============================
Official configuration presets for Kimi-Linear models at various scales.

These configurations are derived from Moonshot AI's published architecture
specifications and can be used directly with ``KimiTransformer``.

Usage::
    from kimi_attention.configs import KIMI_LINEAR_7B_CONFIG
    from kimi_attention.models import KimiTransformer

    model = KimiTransformer(KIMI_LINEAR_7B_CONFIG)
"""

from kimi_attention.models.transformer import KimiConfig

# ---------------------------------------------------------------------------
# Official Kimi-Linear Configurations
# ---------------------------------------------------------------------------

KIMI_LINEAR_1B_CONFIG = KimiConfig(
    dim=2048,
    num_layers=24,
    num_heads=8,
    mlp_ratio=4.0,
    vocab_size=32000,
    max_seq_len=32768,
    layers_per_block=4,
    kda_every=4,
    chunk_size=64,
    eps=1e-6,
    dropout=0.0,
    tie_weights=True,
)
"""Kimi-Linear 1B parameter configuration.

Suited for edge deployment, research experiments, and fine-tuning
applications where resource constraints are significant.
"""

KIMI_LINEAR_7B_CONFIG = KimiConfig(
    dim=4096,
    num_layers=32,
    num_heads=32,
    num_kv_heads=8,
    mlp_ratio=4.0,
    vocab_size=64000,
    max_seq_len=131072,
    layers_per_block=4,
    kda_every=4,
    chunk_size=64,
    eps=1e-6,
    dropout=0.0,
    tie_weights=True,
    rope_theta=500000.0,
)
"""Kimi-Linear 7B parameter configuration.

Balanced performance for production inference and fine-tuning.
Supports up to 128K context length with efficient KDA layers.
"""

KIMI_LINEAR_48B_CONFIG = KimiConfig(
    dim=8192,
    num_layers=64,
    num_heads=64,
    num_kv_heads=8,
    mlp_ratio=4.0,
    vocab_size=128000,
    max_seq_len=1048576,  # 1M context
    layers_per_block=4,
    kda_every=4,
    chunk_size=64,
    eps=1e-6,
    dropout=0.0,
    tie_weights=True,
    rope_theta=1000000.0,
    num_experts=8,
    num_experts_per_tok=2,
)
"""Kimi-Linear 48B parameter configuration (official).

Full-scale model as described in the Kimi Linear paper.
Supports 1M context length with 3.98x throughput improvement.
"""
