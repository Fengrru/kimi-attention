"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch

from kimi_attention.models import KimiConfig, KimiTransformer


@pytest.fixture
def device():
    """Default compute device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def small_config():
    """Small configuration for fast unit tests."""
    return KimiConfig(
        dim=128,
        num_layers=4,
        num_heads=4,
        mlp_ratio=4.0,
        vocab_size=1000,
        max_seq_len=128,
        layers_per_block=2,
        kda_every=2,
        eps=1e-6,
        dropout=0.0,
        tie_weights=True,
    )


@pytest.fixture
def tiny_config():
    """Tiny configuration for very fast tests."""
    return KimiConfig(
        dim=64,
        num_layers=2,
        num_heads=4,
        mlp_ratio=4.0,
        vocab_size=100,
        max_seq_len=32,
        layers_per_block=2,
        kda_every=0,
        eps=1e-6,
        dropout=0.0,
        tie_weights=True,
    )


@pytest.fixture
def small_model(small_config, device):
    """Pre-built small model on the default device."""
    model = KimiTransformer(small_config).to(device)
    return model
