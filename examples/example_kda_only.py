"""
Example: Using Kimi Delta Attention (KDA) Standalone
======================================================
This example shows how to use KDA as a drop-in replacement for
standard multi-head attention in your own architectures.

Usage::
    python examples/example_kda_only.py
"""

import torch
import torch.nn as nn

from kimi_attention.models import KimiDeltaAttentionLayer, RMSNorm
from kimi_attention.utils import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging()


def main():
    # Configuration
    batch_size = 2
    seq_len = 64
    dim = 256
    num_heads = 8

    logger.info("Kimi Delta Attention Standalone Example")
    logger.info(f"Config: batch={batch_size}, seq_len={seq_len}, dim={dim}, heads={num_heads}")

    # Step 1: Initialize KDA layer
    kda_layer = KimiDeltaAttentionLayer(dim=dim, num_heads=num_heads, chunk_size=32)
    norm = RMSNorm(dim)

    logger.info("KDA layer initialized (PyTorch fallback mode on CPU)")

    # Step 2: Create random input
    x = torch.randn(batch_size, seq_len, dim)
    logger.info(f"Input shape: {x.shape}")

    # Step 3: Forward pass
    with torch.no_grad():
        output = kda_layer(norm(x))

    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Output stats: mean={output.mean():.4f}, std={output.std():.4f}")
    assert output.shape == x.shape, "Shape mismatch!"
    assert not torch.isnan(output).any(), "NaN detected in output!"
    logger.info("Forward pass successful!")

    # Step 4: Compare with standard attention
    std_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
    with torch.no_grad():
        std_output, _ = std_attn(x, x, x)

    logger.info("\nComparison with standard MultiheadAttention:")
    logger.info(f"  KDA output:    mean={output.mean():.4f}, std={output.std():.4f}")
    logger.info(f"  Std output:    mean={std_output.mean():.4f}, std={std_output.std():.4f}")
    logger.info("  KDA provides linear complexity vs quadratic for standard attention")

    # Step 5: Demonstrate gradient flow
    x_grad = x.clone().requires_grad_(True)
    output_grad = kda_layer(norm(x_grad))
    loss = output_grad.mean()
    loss.backward()

    assert x_grad.grad is not None, "No gradient computed!"
    logger.info(
        f"\nGradient flow verified: input grad shape={x_grad.grad.shape}, "
        f"mean={x_grad.grad.mean():.6f}"
    )

    logger.info("\nAll checks passed! KDA layer is ready for integration.")


if __name__ == "__main__":
    main()
