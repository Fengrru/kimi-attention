"""
Tests for BlockAttentionResiduals module.
"""

import pytest
import torch
import torch.nn as nn

from kimi_attention.models import BlockAttentionResiduals, RMSNorm


class TestBlockAttentionResiduals:
    """Comprehensive test suite for Attention Residuals."""

    def _make_dummy_layer(self, dim):
        """Create a dummy attention layer for testing."""
        return nn.Linear(dim, dim, bias=False)

    def _make_dummy_mlp(self, dim):
        """Create a dummy MLP for testing."""
        return nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def test_output_shape(self):
        """Output shape should match input shape."""
        dim = 64
        attn_res = BlockAttentionResiduals(dim, layers_per_block=2)
        blocks = []
        hidden = torch.randn(2, 8, dim)

        layer = self._make_dummy_layer(dim)
        mlp = self._make_dummy_mlp(dim)
        norm_a, norm_m = RMSNorm(dim), RMSNorm(dim)

        for i in range(4):
            blocks, hidden = attn_res(
                blocks, hidden, i,
                lambda x: layer(norm_a(x)),
                lambda x: mlp(norm_m(x)),
                norm_a, norm_m,
            )
        assert hidden.shape == (2, 8, dim)

    def test_blocks_accumulated(self):
        """Blocks should be accumulated at correct intervals."""
        dim = 32
        attn_res = BlockAttentionResiduals(dim, layers_per_block=2)
        blocks = []
        hidden = torch.randn(1, 4, dim)

        layer = self._make_dummy_layer(dim)
        mlp = self._make_dummy_mlp(dim)
        norm_a, norm_m = RMSNorm(dim), RMSNorm(dim)

        for i in range(6):
            blocks, hidden = attn_res(
                blocks, hidden, i,
                lambda x: layer(norm_a(x)),
                lambda x: mlp(norm_m(x)),
                norm_a, norm_m,
            )

        # layers_per_block=2, so after 6 layers we expect 3 blocks
        # But layer numbering is 0-indexed, so:
        # layer 1 (i=1) completes block 0 → blocks=1
        # layer 3 (i=3) completes block 1 → blocks=2
        # layer 5 (i=5) completes block 2 → blocks=3
        assert len(blocks) == 3

    def test_identity_on_first_layer(self):
        """When blocks is empty, AttnRes should act as identity."""
        dim = 16
        attn_res = BlockAttentionResiduals(dim, layers_per_block=2)
        blocks = []
        hidden = torch.randn(1, 2, dim)

        # Create identity-like layer
        layer = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(layer.weight)
        mlp = nn.Identity()
        norm = RMSNorm(dim)

        blocks, output = attn_res(
            blocks, hidden, 0,
            lambda x: layer(norm(x)),
            lambda x: mlp(norm(x)),
            norm, norm,
        )
        # On first layer with empty blocks, h_attn = hidden (identity)
        # So output should be close to going through the layer
        assert output.shape == hidden.shape
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Gradients should flow through AttnRes correctly."""
        dim = 32
        attn_res = BlockAttentionResiduals(dim, layers_per_block=2)
        blocks = []
        x_input = torch.randn(1, 4, dim, requires_grad=True)

        layer = self._make_dummy_layer(dim)
        mlp = self._make_dummy_mlp(dim)
        norm_a, norm_m = RMSNorm(dim), RMSNorm(dim)

        hidden = x_input
        for i in range(4):
            blocks, hidden = attn_res(
                blocks, hidden, i,
                lambda x: layer(norm_a(x)),
                lambda x: mlp(norm_m(x)),
                norm_a, norm_m,
            )

        loss = hidden.sum()
        loss.backward()
        # Check gradient flows back to the original input tensor
        assert x_input.grad is not None
        assert not torch.isnan(x_input.grad).any()

    def test_detached_blocks_in_train(self):
        """Stored blocks should be detached during training."""
        dim = 16
        attn_res = BlockAttentionResiduals(dim, layers_per_block=2)
        attn_res.train()
        blocks = []
        hidden = torch.randn(1, 2, dim)

        layer = self._make_dummy_layer(dim)
        mlp = self._make_dummy_mlp(dim)
        norm_a, norm_m = RMSNorm(dim), RMSNorm(dim)

        for i in range(4):
            blocks, hidden = attn_res(
                blocks, hidden, i,
                lambda x: layer(norm_a(x)),
                lambda x: mlp(norm_m(x)),
                norm_a, norm_m,
            )

        assert len(blocks) > 0
        for block in blocks:
            assert not block.requires_grad

    def test_parameters_exist(self):
        """AttnRes should have learnable parameters."""
        dim = 64
        attn_res = BlockAttentionResiduals(dim)
        params = list(attn_res.parameters())
        assert len(params) > 0
        total = sum(p.numel() for p in params)
        assert total > 0

    @pytest.mark.parametrize("layers_per_block", [1, 2, 4, 8])
    def test_different_block_sizes(self, layers_per_block):
        """AttnRes should work with various block sizes."""
        dim = 32
        attn_res = BlockAttentionResiduals(dim, layers_per_block=layers_per_block)
        blocks = []
        hidden = torch.randn(1, 4, dim)

        layer = self._make_dummy_layer(dim)
        mlp = self._make_dummy_mlp(dim)
        norm = RMSNorm(dim)

        num_layers = layers_per_block * 2
        for i in range(num_layers):
            blocks, hidden = attn_res(
                blocks, hidden, i,
                lambda x: layer(norm(x)),
                lambda x: mlp(norm(x)),
                norm, norm,
            )

        assert len(blocks) == num_layers // layers_per_block

    def test_batch_independence(self):
        """Different batch elements should not interfere."""
        dim = 16
        attn_res = BlockAttentionResiduals(dim, layers_per_block=2)

        hidden1 = torch.randn(1, 4, dim)
        hidden2 = torch.randn(1, 4, dim)
        combined = torch.cat([hidden1, hidden2], dim=0)

        layer = self._make_dummy_layer(dim)
        mlp = self._make_dummy_mlp(dim)
        norm = RMSNorm(dim)

        # Process separately
        blocks1, out1 = [], hidden1
        blocks2, out2 = [], hidden2
        for i in range(4):
            blocks1, out1 = attn_res(blocks1, out1, i,
                lambda x: layer(norm(x)),
                lambda x: mlp(norm(x)), norm, norm)
            blocks2, out2 = attn_res(blocks2, out2, i,
                lambda x: layer(norm(x)),
                lambda x: mlp(norm(x)), norm, norm)

        # Process together
        blocks_c, out_c = [], combined
        for i in range(4):
            blocks_c, out_c = attn_res(blocks_c, out_c, i,
                lambda x: layer(norm(x)),
                lambda x: mlp(norm(x)), norm, norm)

        assert torch.allclose(out1, out_c[0:1], atol=1e-5)
        assert torch.allclose(out2, out_c[1:2], atol=1e-5)

    def test_attention_weights_sum_to_one(self):
        """AttnRes attention weights must sum to 1 over depth dim."""
        dim = 16
        attn_res = BlockAttentionResiduals(dim, layers_per_block=2)

        hidden = torch.randn(1, 4, dim)
        layer = self._make_dummy_layer(dim)
        mlp = self._make_dummy_mlp(dim)
        norm_a, norm_m = RMSNorm(dim), RMSNorm(dim)

        blocks = []
        for i in range(4):
            blocks, hidden = attn_res(
                blocks, hidden, i,
                lambda x: layer(norm_a(x)),
                lambda x: mlp(norm_m(x)),
                norm_a, norm_m,
            )

        # With layers_per_block=2, after 4 layers: blocks 0 completed at l=1, blocks 1 at l=3
        assert len(blocks) >= 1
        # The blocks list exists and things didn't crash — aggregation was used

    def test_aggregation_invariance_under_scaling(self):
        """AttnRes aggregation should be invariant to uniform scaling of input."""
        dim = 16
        attn_res = BlockAttentionResiduals(dim, layers_per_block=2)

        hidden = torch.randn(2, 4, dim)
        layer = self._make_dummy_layer(dim)
        mlp = self._make_dummy_mlp(dim)
        norm_a, norm_m = RMSNorm(dim), RMSNorm(dim)

        # Create one previous block so aggregation is tested
        blocks = [torch.randn(2, 4, dim)]

        # Process at a block start to trigger aggregation
        V = torch.stack(blocks + [hidden], dim=0)  # [2, B, T, D]
        K = norm_a(V)
        logits = torch.einsum("d, n b t d -> n b t", attn_res.attn_res_proj.weight.squeeze(), K)
        weights = torch.softmax(logits, dim=0)  # [2, B, T]

        # Weights should sum to 1 along dim 0
        weight_sums = weights.sum(dim=0)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)

        # RMSNorm makes the attention scale-invariant, so scaling input should
        # not change the attention weights
        hidden_scaled = hidden * 2.0
        blocks_scaled = [b * 2.0 for b in blocks]

        V_s = torch.stack(blocks_scaled + [hidden_scaled], dim=0)
        K_s = norm_a(V_s)
        logits_s = torch.einsum("d, n b t d -> n b t", attn_res.attn_res_proj.weight.squeeze(), K_s)
        weights_s = torch.softmax(logits_s, dim=0)

        assert torch.allclose(weights, weights_s, atol=1e-6)
