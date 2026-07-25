"""
Tests for KimiDeltaAttention and KimiDeltaAttentionLayer modules.
"""

import pytest
import torch

from kimi_attention.models import KimiDeltaAttention, KimiDeltaAttentionLayer, RMSNorm


class TestKimiDeltaAttention:
    """Test suite for the core KDA computation module."""

    @pytest.fixture
    def default_tensors(self):
        """Create default test tensors."""
        B, H, T, D = 2, 4, 16, 32
        return {
            "q": torch.randn(B, H, T, D),
            "k": torch.randn(B, H, T, D),
            "v": torch.randn(B, H, T, D),
            "beta": torch.randn(B, H, T, D),
            "g": torch.sigmoid(torch.randn(B, H, T, D)),
            "shape": (B, H, T, D),
        }

    def test_output_shape(self, default_tensors):
        """Output shape should match input Q shape."""
        kda = KimiDeltaAttention(chunk_size=8)
        out = kda(**{k: v for k, v in default_tensors.items() if k != "shape"})
        assert out.shape == default_tensors["shape"]

    def test_no_nan_output(self, default_tensors):
        """Output should not contain NaN values."""
        kda = KimiDeltaAttention()
        out = kda(**{k: v for k, v in default_tensors.items() if k != "shape"})
        assert not torch.isnan(out).any()

    def test_no_inf_output(self, default_tensors):
        """Output should not contain Inf values."""
        kda = KimiDeltaAttention()
        out = kda(**{k: v for k, v in default_tensors.items() if k != "shape"})
        assert not torch.isinf(out).any()

    def test_pytorch_fallback_path(self, default_tensors):
        """PyTorch fallback should produce valid output on CPU."""
        kda = KimiDeltaAttention()
        tensors = {k: v for k, v in default_tensors.items() if k != "shape"}
        # Force CPU execution
        tensors = {k: v.cpu() for k, v in tensors.items()}
        out = kda(**tensors)
        assert out.shape == default_tensors["shape"]
        assert not torch.isnan(out).any()

    def test_beta_sigmoid_clamping(self):
        """Beta should be clamped to (0, 1) via sigmoid."""
        kda = KimiDeltaAttention()
        B, H, T, D = 1, 2, 4, 8

        # Extreme beta values
        beta = torch.tensor([[-10.0, 10.0]]).repeat(B, H, T, D // 2)
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        g = torch.ones(B, H, T, D)

        out = kda(q, k, v, beta, g)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_flow(self, default_tensors):
        """Gradients should flow through KDA computation."""
        kda = KimiDeltaAttention()
        q = default_tensors["q"].clone().requires_grad_(True)
        k = default_tensors["k"].clone().requires_grad_(True)
        v = default_tensors["v"].clone().requires_grad_(True)
        beta = default_tensors["beta"].clone().requires_grad_(True)
        g = default_tensors["g"].clone().requires_grad_(True)

        out = kda(q, k, v, beta, g)
        loss = out.mean()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert beta.grad is not None
        assert g.grad is not None

        for grad in [q.grad, k.grad, v.grad, beta.grad, g.grad]:
            assert not torch.isnan(grad).any()

    def test_single_timestep(self):
        """KDA should work with single-timestep sequences."""
        kda = KimiDeltaAttention()
        B, H, T, D = 2, 4, 1, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        beta = torch.randn(B, H, T, D)
        g = torch.sigmoid(torch.randn(B, H, T, D))

        out = kda(q, k, v, beta, g)
        assert out.shape == (B, H, T, D)

    def test_single_head(self):
        """KDA should work with single-head attention."""
        kda = KimiDeltaAttention()
        B, H, T, D = 2, 1, 8, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        beta = torch.randn(B, H, T, D)
        g = torch.sigmoid(torch.randn(B, H, T, D))

        out = kda(q, k, v, beta, g)
        assert out.shape == (B, H, T, D)

    def test_different_chunk_sizes(self, default_tensors):
        """KDA should work with different chunk sizes."""
        tensors = {k: v for k, v in default_tensors.items() if k != "shape"}
        for chunk_size in [1, 4, 8, 16]:
            kda = KimiDeltaAttention(chunk_size=chunk_size)
            out = kda(**tensors)
            assert out.shape == default_tensors["shape"]

    def test_zero_beta_full_forget(self):
        """When beta ≈ 1 (after sigmoid), memory should be mostly forgotten."""
        kda = KimiDeltaAttention()
        B, H, T, D = 1, 1, 8, 4

        q = torch.ones(B, H, T, D)
        k = torch.ones(B, H, T, D)
        v = torch.ones(B, H, T, D)
        # Large positive beta → sigmoid ≈ 1 → strong forget
        beta = torch.ones(B, H, T, D) * 5.0
        g = torch.ones(B, H, T, D)

        out = kda(q, k, v, beta, g)
        assert out.shape == (B, H, T, D)
        assert not torch.isnan(out).any()

    def test_zero_beta_full_retain(self):
        """When beta ≈ 0 (after sigmoid), memory should be mostly retained."""
        kda = KimiDeltaAttention()
        B, H, T, D = 1, 1, 8, 4

        q = torch.ones(B, H, T, D)
        k = torch.ones(B, H, T, D)
        v = torch.ones(B, H, T, D)
        # Large negative beta → sigmoid ≈ 0 → strong retain
        beta = torch.ones(B, H, T, D) * (-5.0)
        g = torch.ones(B, H, T, D)

        out = kda(q, k, v, beta, g)
        assert out.shape == (B, H, T, D)
        assert not torch.isnan(out).any()

    def test_chunked_matches_serial(self):
        """Chunked parallel forward must match ground-truth serial recurrence."""
        kda = KimiDeltaAttention(chunk_size=3)

        B, H, T, D = 1, 2, 10, 4
        torch.manual_seed(42)
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        beta_raw = torch.randn(B, H, T, D)
        g = torch.sigmoid(torch.randn(B, H, T, D))

        # Chunked forward (multiple chunks since C=3 < T=10)
        out_chunked = kda(q, k, v, beta_raw, g)

        # Serial ground truth
        beta_sig = torch.sigmoid(beta_raw)
        m = torch.zeros(B, H, D, dtype=q.dtype)
        serial_outputs = []
        for t in range(T):
            q_t = q[:, :, t, :]
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            b_t = beta_sig[:, :, t, :]
            g_t = g[:, :, t, :]
            m = b_t * m + (1 - b_t) * (k_t * v_t)
            serial_outputs.append(g_t * (q_t * m))
        out_serial = torch.stack(serial_outputs, dim=2)

        assert torch.allclose(out_chunked, out_serial, atol=1e-5)

    def test_chunked_matches_serial_non_power_of_two(self):
        """Works when chunk_count is not a power of two."""
        kda = KimiDeltaAttention(chunk_size=4)

        B, H, T, D = 2, 2, 7, 4  # 7 is odd, not a power of two
        torch.manual_seed(17)
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        beta_raw = torch.randn(B, H, T, D)
        g = torch.sigmoid(torch.randn(B, H, T, D))

        out_chunked = kda(q, k, v, beta_raw, g)

        beta_sig = torch.sigmoid(beta_raw)
        m = torch.zeros(B, H, D, dtype=q.dtype)
        serial_outputs = []
        for t in range(T):
            q_t = q[:, :, t, :]
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            b_t = beta_sig[:, :, t, :]
            g_t = g[:, :, t, :]
            m = b_t * m + (1 - b_t) * (k_t * v_t)
            serial_outputs.append(g_t * (q_t * m))
        out_serial = torch.stack(serial_outputs, dim=2)

        assert torch.allclose(out_chunked, out_serial, atol=1e-5)

    def test_chunked_matches_serial_larger(self):
        """Larger test: T=64 with chunk_size=8 (8 chunks of 8)."""
        kda = KimiDeltaAttention(chunk_size=8)

        B, H, T, D = 1, 4, 64, 8
        torch.manual_seed(99)
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        beta_raw = torch.randn(B, H, T, D)
        g = torch.sigmoid(torch.randn(B, H, T, D))

        out_chunked = kda(q, k, v, beta_raw, g)

        beta_sig = torch.sigmoid(beta_raw)
        m = torch.zeros(B, H, D, dtype=q.dtype)
        serial_outputs = []
        for t in range(T):
            q_t = q[:, :, t, :]
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            b_t = beta_sig[:, :, t, :]
            g_t = g[:, :, t, :]
            m = b_t * m + (1 - b_t) * (k_t * v_t)
            serial_outputs.append(g_t * (q_t * m))
        out_serial = torch.stack(serial_outputs, dim=2)

        assert torch.allclose(out_chunked, out_serial, atol=1e-5)


class TestKimiDeltaAttentionLayer:
    """Test suite for the complete KDA attention layer."""

    def test_output_shape(self):
        """Layer output shape should match input shape."""
        layer = KimiDeltaAttentionLayer(dim=128, num_heads=4)
        x = torch.randn(2, 16, 128)
        out = layer(x)
        assert out.shape == x.shape

    def test_no_nan(self):
        """Layer output should not contain NaN."""
        layer = KimiDeltaAttentionLayer(dim=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        """Gradients should flow through the entire layer."""
        layer = KimiDeltaAttentionLayer(dim=64, num_heads=4)
        x = torch.randn(1, 8, 64, requires_grad=True)
        out = layer(x)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_learnable_params(self):
        """Layer should have learnable parameters."""
        layer = KimiDeltaAttentionLayer(dim=128, num_heads=8)
        params = list(layer.parameters())
        assert len(params) > 0
        total = sum(p.numel() for p in params)
        assert total > 0

    def test_dim_not_divisible_by_heads(self):
        """Should raise error when dim is not divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible"):
            KimiDeltaAttentionLayer(dim=100, num_heads=8)

    def test_inference_mode(self):
        """Layer should work in eval mode."""
        layer = KimiDeltaAttentionLayer(dim=64, num_heads=4)
        layer.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            out = layer(x)
        assert out.shape == x.shape

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [1, 8, 32, 128])
    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_various_shapes(self, batch_size, seq_len, dim):
        """Layer should handle various input shapes."""
        layer = KimiDeltaAttentionLayer(dim=dim, num_heads=4)
        x = torch.randn(batch_size, seq_len, dim)
        out = layer(x)
        assert out.shape == (batch_size, seq_len, dim)

    def test_with_rmsnorm(self):
        """KDA layer should work correctly after RMSNorm."""
        dim = 64
        layer = KimiDeltaAttentionLayer(dim=dim, num_heads=4)
        norm = RMSNorm(dim)
        x = torch.randn(1, 8, dim)
        out = layer(norm(x))
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_parameter_count_consistency(self):
        """Parameter count should match expected formula."""
        dim = 128
        num_heads = 8
        layer = KimiDeltaAttentionLayer(dim=dim, num_heads=num_heads)

        # 5 projections (q, k, v, beta, g) + 1 output projection
        # Each: dim * dim parameters
        expected = 6 * dim * dim
        actual = sum(p.numel() for p in layer.parameters())
        assert actual == expected
