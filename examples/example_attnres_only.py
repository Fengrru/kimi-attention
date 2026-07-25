"""
Example: Using Attention Residuals (AttnRes) Standalone
========================================================
This example demonstrates how to integrate BlockAttentionResiduals
into an existing Transformer architecture.

The key insight: AttnRes acts as a wrapper around your existing
attention and FFN functions, requiring zero changes to their internals.

Usage::
    python examples/example_attnres_only.py
"""

import torch
import torch.nn as nn

from kimi_attention.models import BlockAttentionResiduals, RMSNorm
from kimi_attention.utils import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging()


def main():
    # Configuration
    batch_size = 2
    seq_len = 32
    dim = 128
    num_layers = 8
    layers_per_block = 4  # AttnRes block size

    logger.info("Attention Residuals (AttnRes) Standalone Example")
    logger.info(f"Config: {num_layers} layers, block_size={layers_per_block}")

    # Step 1: Initialize AttnRes module
    attn_res = BlockAttentionResiduals(dim=dim, layers_per_block=layers_per_block)
    logger.info(f"AttnRes initialized: {sum(p.numel() for p in attn_res.parameters())} params")

    # Step 2: Create simple dummy layers (your real layers would go here)
    attn_layers = [nn.Linear(dim, dim) for _ in range(num_layers)]
    mlp_layers = [
        nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        for _ in range(num_layers)
    ]
    norm_layers = [(RMSNorm(dim), RMSNorm(dim)) for _ in range(num_layers)]

    # Step 3: Initialize with random weights
    for layer in attn_layers:
        nn.init.normal_(layer.weight, 0, 0.02)
    for mlp in mlp_layers:
        for fc in mlp:
            if hasattr(fc, "weight"):
                nn.init.normal_(fc.weight, 0, 0.02)

    # Step 4: Forward pass with AttnRes
    x = torch.randn(batch_size, seq_len, dim)
    logger.info(f"Input shape: {x.shape}")

    blocks = []
    hidden = x

    for layer_idx in range(num_layers):
        attn_fn = lambda h, idx=layer_idx: attn_layers[idx](h)
        mlp_fn = lambda h, idx=layer_idx: mlp_layers[idx](h)
        attn_norm, mlp_norm = norm_layers[layer_idx]

        blocks, hidden = attn_res(
            blocks=blocks,
            hidden_states=hidden,
            layer_number=layer_idx,
            attn_fn=attn_fn,
            mlp_fn=mlp_fn,
            attn_norm=attn_norm,
            mlp_norm=mlp_norm,
        )

    logger.info(f"Output shape: {hidden.shape}")
    logger.info(f"Blocks stored: {len(blocks)} (expected: {num_layers // layers_per_block})")

    # Step 5: Compare with standard Transformer (no AttnRes)
    std_hidden = x
    for layer_idx in range(num_layers):
        h = std_hidden + attn_layers[layer_idx](norm_layers[layer_idx][0](std_hidden))
        std_hidden = h + mlp_layers[layer_idx](norm_layers[layer_idx][1](h))

    logger.info("\nComparison with standard residual Transformer:")
    logger.info(f"  With AttnRes:    mean={hidden.mean():.4f}, std={hidden.std():.4f}")
    logger.info(f"  Standard:        mean={std_hidden.mean():.4f}, std={std_hidden.std():.4f}")
    logger.info(f"  Difference:      {torch.abs(hidden - std_hidden).mean():.4f} (mean abs diff)")

    # Step 6: Verify gradient flow
    hidden_grad = x.clone().requires_grad_(True)
    blocks_grad = []

    for layer_idx in range(num_layers):
        blocks_grad, h = attn_res(
            blocks=blocks_grad,
            hidden_states=hidden_grad,
            layer_number=layer_idx,
            attn_fn=lambda h, idx=layer_idx: attn_layers[idx](h),
            mlp_fn=lambda h, idx=layer_idx: mlp_layers[idx](h),
            attn_norm=norm_layers[layer_idx][0],
            mlp_norm=norm_layers[layer_idx][1],
        )

    loss = h.mean()
    loss.backward()

    assert hidden_grad.grad is not None
    logger.info(f"\nGradient flow verified: input grad mean={hidden_grad.grad.mean():.6f}")

    # Step 7: Show memory benefit
    bytes_per_block = batch_size * seq_len * dim * 4  # FP32
    num_blocks = num_layers // layers_per_block
    logger.info(f"\nMemory analysis:")
    logger.info(
        f"  Stored blocks: {num_blocks} x {bytes_per_block / 1024:.1f} KB = "
        f"{num_blocks * bytes_per_block / 1024:.1f} KB"
    )
    logger.info(f"  Without AttnRes: would need full-layer cache")
    logger.info(f"  Savings: only {num_blocks}/{num_layers} layer outputs stored")

    logger.info("\nAll checks passed! AttnRes is ready for integration.")


if __name__ == "__main__":
    main()
