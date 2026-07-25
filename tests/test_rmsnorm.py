"""
Tests for RMSNorm module.
"""

import pytest
import torch

from kimi_attention.models import RMSNorm


class TestRMSNorm:
    """Comprehensive test suite for RMSNorm."""

    def test_output_shape_preserved(self):
        """RMSNorm should preserve input shape."""
        dim = 512
        norm = RMSNorm(dim)
        x = torch.randn(2, 16, dim)
        out = norm(x)
        assert out.shape == x.shape

    def test_rms_is_unity(self):
        """After RMSNorm, the RMS of each vector should be approximately 1."""
        dim = 256
        norm = RMSNorm(dim)
        x = torch.randn(4, 8, dim)
        out = norm(x)
        rms = out.float().pow(2).mean(-1)
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_learnable_weight(self):
        """The weight parameter should be learnable."""
        norm = RMSNorm(128)
        assert norm.weight.requires_grad
        assert norm.weight.shape == (128,)

    def test_weight_initialization(self):
        """Weight should be initialized to ones."""
        norm = RMSNorm(64)
        assert torch.allclose(norm.weight, torch.ones(64))

    def test_eps_effect(self):
        """Smaller eps should give slightly different results."""
        x = torch.randn(2, 4, 32)
        norm1 = RMSNorm(32, eps=1e-3)
        norm2 = RMSNorm(32, eps=1e-6)
        out1 = norm1(x)
        out2 = norm2(x)
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_dtype_preservation(self):
        """Output dtype should match input dtype (float32 always correct; f16/bf16 on GPU)."""
        # Float32 always works
        norm = RMSNorm(32)
        x = torch.randn(2, 4, 32)
        out = norm(x)
        assert out.dtype == torch.float32

        # Float16 and BFloat16 require CUDA
        if torch.cuda.is_available():
            for dtype in [torch.float16, torch.bfloat16]:
                norm = RMSNorm(32).cuda()
                x = torch.randn(2, 4, 32, device="cuda", dtype=dtype)
                out = norm(x)
                assert out.dtype == dtype

    def test_backward_gradient(self):
        """Gradients should flow through RMSNorm correctly."""
        norm = RMSNorm(16)
        x = torch.randn(2, 4, 16, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_single_token(self):
        """RMSNorm should work with single-token sequences."""
        norm = RMSNorm(64)
        x = torch.randn(1, 1, 64)
        out = norm(x)
        assert out.shape == (1, 1, 64)

    def test_repr(self):
        """String representation should contain key info."""
        norm = RMSNorm(512, eps=1e-5)
        repr_str = repr(norm)
        assert "RMSNorm" in repr_str
        assert "512" in repr_str

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    @pytest.mark.parametrize("seq_len", [1, 16, 128])
    @pytest.mark.parametrize("dim", [32, 128, 512])
    def test_various_shapes(self, batch_size, seq_len, dim):
        """RMSNorm should handle various input shapes."""
        norm = RMSNorm(dim)
        x = torch.randn(batch_size, seq_len, dim)
        out = norm(x)
        assert out.shape == (batch_size, seq_len, dim)
