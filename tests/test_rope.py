"""
Tests for Rotary Position Embedding (RoPE).
"""

import pytest
import torch

from kimi_attention.models.rope import RotaryEmbedding, apply_rotary_emb


class TestRotaryEmbedding:
    def test_output_shape(self):
        rope = RotaryEmbedding(dim=64, max_seq_len=128)
        x = torch.randn(2, 4, 16, 64)
        pos = torch.arange(16).unsqueeze(0).expand(2, -1)
        out = rope(x, pos)
        assert out.shape == x.shape

    def test_relative_distance_property(self):
        """RoPE should make q·k depend only on relative position."""
        rope = RotaryEmbedding(dim=16, max_seq_len=64)
        q = torch.randn(1, 2, 2, 16)
        k = torch.randn(1, 2, 2, 16)

        # Position 0, 1
        pos = torch.tensor([[0, 1]])
        q_rot = rope(q, pos)
        k_rot = rope(k, pos)

        # Dot product q[0]·k[0] ≠ q[1]·k[1] due to different positions
        dot_0 = (q_rot[:, :, 0, :] * k_rot[:, :, 0, :]).sum()
        dot_1 = (q_rot[:, :, 1, :] * k_rot[:, :, 1, :]).sum()
        assert not torch.allclose(dot_0, dot_1)

    def test_identity_at_position_zero(self):
        """At position 0, cos=1 and sin=0 → no rotation."""
        rope = RotaryEmbedding(dim=16, max_seq_len=64)
        x = torch.randn(1, 2, 1, 16)
        pos = torch.tensor([[0]])
        out = rope(x, pos)
        assert torch.allclose(out, x, atol=1e-6)

    def test_deterministic(self):
        rope = RotaryEmbedding(dim=32, max_seq_len=64)
        x = torch.randn(2, 4, 8, 32)
        pos = torch.arange(8).unsqueeze(0).expand(2, -1)
        out1 = rope(x, pos)
        out2 = rope(x, pos)
        assert torch.equal(out1, out2)

    def test_gradient_flow(self):
        rope = RotaryEmbedding(dim=16, max_seq_len=32)
        x = torch.randn(1, 2, 4, 16, requires_grad=True)
        pos = torch.arange(4).unsqueeze(0)
        out = rope(x, pos)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_various_dims(self):
        for dim in [8, 16, 32, 64, 128]:
            rope = RotaryEmbedding(dim=dim, max_seq_len=32)
            x = torch.randn(1, 2, 4, dim)
            pos = torch.arange(4).unsqueeze(0)
            out = rope(x, pos)
            assert out.shape == x.shape

    def test_single_token(self):
        """RoPE must work for single-token generation."""
        rope = RotaryEmbedding(dim=32, max_seq_len=64)
        x = torch.randn(1, 4, 1, 32)
        pos = torch.tensor([[42]])  # arbitrary position
        out = rope(x, pos)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_even_dim_enforced(self):
        with pytest.raises(ValueError):
            RotaryEmbedding(dim=15, max_seq_len=32)

    def test_rotate_half(self):
        from kimi_attention.models.rope import rotate_half

        x = torch.randn(2, 4, 8, 16)
        rh = rotate_half(x)
        assert rh.shape == x.shape
        # rotate_half(x) should be orthogonal to x in each 2D pair
        x1, x2 = x[..., :8], x[..., 8:]
        rh1, rh2 = rh[..., :8], rh[..., 8:]
        assert torch.allclose(rh1, -x2)
        assert torch.allclose(rh2, x1)

    def test_apply_rotary_emb_direct(self):
        cos = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
        sin = torch.tensor([[[[0.0, 1.0, 0.0, 1.0]]]])
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        out = apply_rotary_emb(x, cos, sin)
        # rotate_half swaps halves: [1,2,3,4] → [-3,-4,1,2]
        # x*cos = [1,0,3,0]
        # rh*sin = [0,-4,0,2]
        # result = [1,-4,3,2]
        expected = torch.tensor([[[[1.0, -4.0, 3.0, 2.0]]]])
        assert torch.allclose(out, expected, atol=1e-6)
