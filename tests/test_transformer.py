"""
Tests for KimiTransformer end-to-end model.
"""

import pytest
import torch

from kimi_attention.models import KimiConfig, KimiTransformer


class TestKimiTransformer:
    """End-to-end test suite for KimiTransformer."""

    def test_forward_output_shape(self, small_model, small_config):
        """Forward pass should produce correct output shape."""
        batch_size = 2
        seq_len = 16
        device = next(small_model.parameters()).device
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len), device=device)
        logits = small_model(input_ids)
        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)

    def test_forward_no_nan(self, small_model, small_config):
        """Forward pass should not produce NaN."""
        device = next(small_model.parameters()).device
        input_ids = torch.randint(0, small_config.vocab_size, (1, 16), device=device)
        logits = small_model(input_ids)
        assert not torch.isnan(logits).any()

    def test_gradient_flow(self, small_config):
        """Gradients should flow to all parameters."""
        model = KimiTransformer(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (1, 8))
        logits = model(input_ids)
        loss = logits.mean()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_parameter_count(self, small_config):
        """Model should report parameter count."""
        model = KimiTransformer(small_config)
        count = model.count_parameters()
        assert count > 0
        # Rough estimate: vocab_emb + pos_emb + layers + head
        min_expected = small_config.vocab_size * small_config.dim
        assert count > min_expected

    def test_weight_tying(self, small_config):
        """When tie_weights=True, embedding and LM head should share weights."""
        config = small_config
        config.tie_weights = True
        model = KimiTransformer(config)
        assert model.lm_head.weight is model.token_emb.weight

    def test_no_weight_tying(self, small_config):
        """When tie_weights=False, embedding and LM head should be separate."""
        config = small_config
        config.tie_weights = False
        model = KimiTransformer(config)
        assert model.lm_head.weight is not model.token_emb.weight

    def test_generate_output_shape(self, small_model, small_config):
        """Generation should produce longer sequences."""
        prompt_len = 8
        max_new = 16
        device = next(small_model.parameters()).device
        input_ids = torch.randint(0, small_config.vocab_size, (1, prompt_len), device=device)

        small_model.eval()
        with torch.no_grad():
            generated = small_model.generate(input_ids, max_new_tokens=max_new)

        assert generated.shape[0] == 1
        assert generated.shape[1] >= prompt_len

    def test_generate_deterministic(self, small_model, small_config):
        """Generation with identical seed should be deterministic."""
        device = next(small_model.parameters()).device
        input_ids = torch.randint(0, small_config.vocab_size, (1, 4), device=device)

        small_model.eval()
        with torch.no_grad():
            # Use explicit seed for deterministic sampling
            torch.manual_seed(42)
            out1 = small_model.generate(input_ids, max_new_tokens=8, temperature=1.0)
            torch.manual_seed(42)
            out2 = small_model.generate(input_ids, max_new_tokens=8, temperature=1.0)

        assert torch.equal(out1, out2)

    def test_different_model_sizes(self):
        """Model should work at different scales."""
        # Build a small config inspired by 1B but sized for test environments
        config = KimiConfig(
            dim=512, num_layers=8, num_heads=8,
            vocab_size=1000, max_seq_len=64,
            layers_per_block=2, kda_every=4,
        )
        model = KimiTransformer(config)

        input_ids = torch.randint(0, config.vocab_size, (1, 16))
        logits = model(input_ids)
        assert logits.shape == (1, 16, config.vocab_size)

    def test_config_from_size_error(self):
        """Invalid size should raise ValueError."""
        with pytest.raises(ValueError):
            KimiConfig.from_size("invalid_size")

    def test_memory_estimate(self, small_model, small_config):
        """Memory estimate should return positive values."""
        mem = small_model.estimate_memory(batch_size=2, seq_len=16)
        assert mem["parameters_mb"] > 0
        assert mem["kv_cache_mb"] >= 0
        assert mem["total_mb"] > 0

    def test_train_eval_mode(self, small_model):
        """Model should support train/eval mode switching."""
        small_model.train()
        assert small_model.training
        small_model.eval()
        assert not small_model.training

    def test_batch_processing(self, small_model, small_config):
        """Model should handle various batch sizes."""
        device = next(small_model.parameters()).device
        for batch_size in [1, 2, 4]:
            input_ids = torch.randint(0, small_config.vocab_size, (batch_size, 16), device=device)
            logits = small_model(input_ids)
            assert logits.shape == (batch_size, 16, small_config.vocab_size)

    def test_long_sequence(self, small_config):
        """Model should handle sequences near max_seq_len."""
        config = small_config
        config.max_seq_len = 64
        model = KimiTransformer(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 64))
        logits = model(input_ids)
        assert logits.shape == (1, 64, config.vocab_size)

    def test_seq_len_exceeds_max(self, small_config):
        """Should raise error when sequence exceeds max_seq_len."""
        config = small_config
        config.max_seq_len = 16
        model = KimiTransformer(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 32))
        with pytest.raises(ValueError, match="exceeds"):
            model(input_ids)

    def test_save_load(self, small_config, tmp_path):
        """Model should be saveable and loadable."""
        model1 = KimiTransformer(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (1, 8))

        # Get output before save
        model1.eval()
        with torch.no_grad():
            out1 = model1(input_ids)

        # Save
        ckpt_path = tmp_path / "model.pt"
        torch.save(model1.state_dict(), ckpt_path)

        # Load
        model2 = KimiTransformer(small_config)
        model2.load_state_dict(torch.load(ckpt_path, weights_only=True))

        # Compare outputs
        model2.eval()
        with torch.no_grad():
            out2 = model2(input_ids)

        assert torch.allclose(out1, out2)

    def test_kda_mha_ratio(self, small_config):
        """Verify correct KDA to MHA layer ratio."""
        config = small_config
        config.num_layers = 8
        config.kda_every = 4
        model = KimiTransformer(config)

        kda_count = sum(1 for ly in model.layers if ly.use_kda)
        mha_count = sum(1 for ly in model.layers if not ly.use_kda)

        # kda_every=4 → layers 0,1,2 = KDA, layer 3 = MHA → 3:1 ratio
        assert kda_count == 6
        assert mha_count == 2

    def test_all_kda(self, small_config):
        """With kda_every=0, all layers should use KDA."""
        config = small_config
        config.kda_every = 0
        model = KimiTransformer(config)

        for layer in model.layers:
            assert layer.use_kda

    def test_all_mha(self, small_config):
        """With kda_every=1, alternate layers should use KDA."""
        config = small_config
        config.kda_every = 1
        model = KimiTransformer(config)

        # kda_every=1 → every layer gets standard attention (pattern: MHA)
        for i, layer in enumerate(model.layers):
            # With the formula: (i % 1) != 0 → False for all i → all MHA
            assert not layer.use_kda

    def test_gqa_kv_heads(self):
        """GQA: KV heads fewer than Q heads should work correctly."""
        config = KimiConfig(
            dim=256, num_layers=4, num_heads=8, num_kv_heads=2,
            vocab_size=1000, max_seq_len=64, kda_every=0,
            layers_per_block=2,
        )
        model = KimiTransformer(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        logits = model(input_ids)
        assert logits.shape == (2, 16, config.vocab_size)
        assert not torch.isnan(logits).any()

    def test_moe_model(self):
        """Transformer with MoE FFN should work end‑to‑end."""
        config = KimiConfig(
            dim=128, num_layers=2, num_heads=4,
            vocab_size=500, max_seq_len=32,
            kda_every=0, layers_per_block=2,
            num_experts=4, num_experts_per_tok=2,
        )
        model = KimiTransformer(config)
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        logits = model(input_ids)
        assert logits.shape == (1, 8, config.vocab_size)

        # Check that balance loss is positive during training
        total_loss = model.get_total_balance_loss()
        assert total_loss > 0

        # Check inference mode (no balance loss)
        model.eval()
        logits = model(input_ids)
        total_loss_eval = model.get_total_balance_loss()
        assert total_loss_eval == 0.0

    def test_rope_model(self):
        """Model with RoPE should produce valid outputs."""
        config = KimiConfig(
            dim=128, num_layers=2, num_heads=4,
            vocab_size=500, max_seq_len=64,
            rope_theta=10000.0,
            kda_every=0, layers_per_block=2,
        )
        model = KimiTransformer(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 16))
        logits = model(input_ids)
        assert not torch.isnan(logits).any()

    def test_beam_search(self):
        """Beam search should produce a longer sequence."""
        config = KimiConfig(
            dim=64, num_layers=2, num_heads=4,
            vocab_size=200, max_seq_len=32,
            kda_every=0, layers_per_block=2,
        )
        model = KimiTransformer(config)
        prompt = torch.randint(0, config.vocab_size, (1, 4))

        generated = model.generate_beam(
            prompt, max_new_tokens=8, num_beams=3, eos_token_id=0,
        )
        assert generated.shape[0] == 1
        assert generated.shape[1] > 4
