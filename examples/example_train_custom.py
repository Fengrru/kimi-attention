"""
Example: Training a Custom Kimi Transformer
============================================
This example demonstrates how to:
    1. Create a custom model configuration
    2. Initialize the KimiTransformer
    3. Set up a training loop with gradient clipping and LR scheduling
    4. Save checkpoints

Usage::
    python examples/example_train_custom.py
"""

import torch
from tqdm import tqdm

from kimi_attention.models import KimiConfig, KimiTransformer
from kimi_attention.utils import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging()


def main():
    # Step 1: Define a small custom configuration
    config = KimiConfig(
        dim=256,  # Small dimension for demo
        num_layers=4,  # Only 4 layers
        num_heads=4,  # 4 attention heads
        mlp_ratio=4.0,
        vocab_size=1000,  # Tiny vocabulary
        max_seq_len=128,  # Short sequences
        layers_per_block=2,  # AttnRes block size
        kda_every=2,  # 1:1 KDA:MHA ratio for demo
        eps=1e-6,
        dropout=0.1,
    )
    logger.info(f"Configuration: {config}")

    # Step 2: Initialize model
    model = KimiTransformer(config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized: {total_params:,} parameters ({total_params/1e6:.2f}M)")

    # Step 3: Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Step 4: Training loop with synthetic data
    num_steps = 100
    batch_size = 4
    seq_len = 32

    model.train()
    logger.info(f"Starting training for {num_steps} steps...")

    losses = []
    for step in tqdm(range(num_steps), desc="Training"):
        # Synthetic batch
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        # Forward
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            labels.view(-1),
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if step % 20 == 0:
            avg_loss = sum(losses[-20:]) / len(losses[-20:])
            logger.info(f"Step {step:3d} | Loss: {avg_loss:.4f}")

    final_avg = sum(losses[-20:]) / len(losses[-20:])
    logger.info(f"Training complete! Final avg loss: {final_avg:.4f}")

    # Step 5: Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "step": num_steps,
        "loss": final_avg,
    }
    torch.save(checkpoint, "custom_model.pt")
    logger.info("Saved checkpoint to custom_model.pt")

    # Step 6: Test generation
    model.eval()
    with torch.no_grad():
        prompt = torch.randint(0, config.vocab_size, (1, 8))
        generated = model.generate(prompt, max_new_tokens=16, temperature=1.0)
        logger.info(f"Prompt length: {prompt.size(1)}, Generated length: {generated.size(1)}")


if __name__ == "__main__":
    main()
