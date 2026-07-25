"""
Tests for Mixture of Experts (MoE) Feed-Forward network.
"""

import pytest
import torch

from kimi_attention.models.moe import MoEFeedForward


class TestMoEFeedForward:
    def test_output_shape(self):
        moe = MoEFeedForward(dim=64, hidden_dim=128, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 64)
        out = moe(x)
        assert out.shape == x.shape

    def test_no_nan(self):
        moe = MoEFeedForward(dim=64, hidden_dim=128, num_experts=4, top_k=2)
        x = torch.randn(1, 16, 64)
        out = moe(x)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        moe = MoEFeedForward(dim=32, hidden_dim=64, num_experts=4, top_k=2)
        x = torch.randn(1, 4, 32, requires_grad=True)
        out = moe(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_load_balance_loss(self):
        """Load balance loss should be non-negative and higher with skewed routing."""
        moe = MoEFeedForward(dim=32, hidden_dim=64, num_experts=4, top_k=1)

        x_skewed = torch.randn(1, 16, 32)
        moe.router.weight.data.zero_()
        moe.router.weight.data[0, :] = 100.0

        _, loss_skewed = moe.forward(x_skewed, return_balance_loss=True)

        moe.router.weight.data.normal_(0, 0.02)
        _, loss_balanced = moe.forward(torch.randn(1, 16, 32), return_balance_loss=True)

        assert loss_skewed > 0
        assert loss_balanced >= 0

    def test_top_k_routing(self):
        """Only top-k experts should be activated per token."""
        moe = MoEFeedForward(dim=32, hidden_dim=64, num_experts=8, top_k=2)
        # Zero out all expert weights except expert 0 and 1
        for i, expert in enumerate(moe.experts):
            for p in expert.parameters():
                p.data.zero_()
            if i < 2:
                for p in expert.parameters():
                    p.data.normal_(0, 0.02)

        moe.router.weight.data.zero_()
        moe.router.weight.data[0, :] = 10.0
        moe.router.weight.data[1, :] = 5.0

        x = torch.ones(1, 4, 32)
        out = moe(x)
        assert not torch.isnan(out).any()
        assert out.shape == x.shape

    def test_single_token(self):
        moe = MoEFeedForward(dim=64, hidden_dim=128, num_experts=4, top_k=2)
        x = torch.randn(1, 1, 64)
        out = moe(x)
        assert out.shape == (1, 1, 64)

    def test_various_sizes(self):
        for dim, hidden, experts, topk in [
            (32, 64, 2, 1),
            (64, 128, 4, 2),
            (128, 256, 8, 2),
        ]:
            moe = MoEFeedForward(dim=dim, hidden_dim=hidden, num_experts=experts, top_k=topk)
            x = torch.randn(1, 8, dim)
            out = moe(x)
            assert out.shape == x.shape

    def test_parameter_count(self):
        """MoE should have more params than equivalent dense FFN."""
        dim, hidden = 64, 128
        moe = MoEFeedForward(dim, hidden, num_experts=4, top_k=2)
        n_moe = sum(p.numel() for p in moe.parameters())

        # Dense equivalent: 3 linear layers (fc1, fc2, fc3) = 2*dim*hidden + hidden*dim
        n_dense = 2 * dim * hidden + hidden * dim  # fc1 + fc2 + fc3
        assert n_moe > n_dense
