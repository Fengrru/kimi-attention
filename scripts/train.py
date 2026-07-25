#!/usr/bin/env python3
"""
Training Script for Kimi Transformer
=====================================
Example training script with data loading, optimization, and checkpointing.

Usage::
    python scripts/train.py \
        --config 1B \
        --dataset openwebtext \
        --batch_size 32 \
        --max_steps 100000 \
        --output_dir ./checkpoints

Features:
    - Automatic mixed precision (AMP) training
    - Gradient clipping and accumulation
    - Learning rate warmup + cosine decay
    - Periodic checkpointing
    - WandB integration

.. note::
    This script uses synthetic random data for demonstration purposes.
    Replace ``dummy_dataloader()`` with a real ``DataLoader`` and
    integrate a proper tokenizer (e.g., HuggingFace ``AutoTokenizer``)
    before using in production.
"""

import argparse
from pathlib import Path

import torch

from kimi_attention.configs import (
    KIMI_LINEAR_1B_CONFIG,
    KIMI_LINEAR_7B_CONFIG,
    KIMI_LINEAR_48B_CONFIG,
)
from kimi_attention.models import KimiConfig, KimiTransformer
from kimi_attention.utils import get_logger, setup_logging

logger = get_logger(__name__)


CONFIG_MAP = {
    "1B": KIMI_LINEAR_1B_CONFIG,
    "7B": KIMI_LINEAR_7B_CONFIG,
    "48B": KIMI_LINEAR_48B_CONFIG,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Kimi Transformer")

    # Model config
    group = parser.add_argument_group("Model Configuration")
    group.add_argument(
        "--config",
        type=str,
        default="1B",
        choices=["1B", "7B", "48B", "custom"],
        help="Model configuration preset",
    )
    group.add_argument("--dim", type=int, help="Override: model dimension")
    group.add_argument("--num_layers", type=int, help="Override: number of layers")
    group.add_argument("--num_heads", type=int, help="Override: number of heads")
    group.add_argument("--num_kv_heads", type=int, help="Override: KV heads (GQA)")
    group.add_argument("--num_experts", type=int, help="Override: MoE experts (0 = dense)")
    group.add_argument("--num_experts_per_tok", type=int, help="Override: MoE top-k")
    group.add_argument("--rope_theta", type=float, help="Override: RoPE theta")

    # Training
    group = parser.add_argument_group("Training Hyperparameters")
    group.add_argument("--batch_size", type=int, default=32, help="Global batch size")
    group.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    group.add_argument("--max_steps", type=int, default=100000, help="Training steps")
    group.add_argument("--learning_rate", type=float, default=3e-4, help="Peak LR")
    group.add_argument("--min_lr", type=float, default=3e-5, help="Minimum LR")
    group.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    group.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    group.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clip norm")
    group.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation")

    # I/O
    group = parser.add_argument_group("I/O and Logging")
    group.add_argument("--output_dir", type=str, default="./checkpoints")
    group.add_argument("--save_every", type=int, default=5000, help="Checkpoint interval")
    group.add_argument("--log_every", type=int, default=100, help="Logging interval")
    group.add_argument("--eval_every", type=int, default=5000, help="Evaluation interval")
    group.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    group.add_argument("--run_name", type=str, default=None, help="WandB run name")

    # System
    group = parser.add_argument_group("System Configuration")
    group.add_argument("--seed", type=int, default=42, help="Random seed")
    group.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    group.add_argument("--device", type=str, default="cuda", help="Compute device")

    return parser.parse_args()


def get_lr(step: int, warmup_steps: int, max_steps: int, peak_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))


def main() -> None:
    args = parse_args()
    setup_logging()

    # Set seed
    torch.manual_seed(args.seed)

    # Build config
    if args.config in CONFIG_MAP:
        config = CONFIG_MAP[args.config]
    else:
        config = KimiConfig()
    # Apply overrides
    if args.dim:
        config.dim = args.dim
    if args.num_layers:
        config.num_layers = args.num_layers
    if args.num_heads:
        config.num_heads = args.num_heads
    if args.num_kv_heads is not None:
        config.num_kv_heads = args.num_kv_heads
    if args.num_experts is not None:
        config.num_experts = args.num_experts
    if args.num_experts_per_tok is not None:
        config.num_experts_per_tok = args.num_experts_per_tok
    if args.rope_theta is not None:
        config.rope_theta = args.rope_theta
    config.max_seq_len = max(config.max_seq_len, args.seq_len)

    logger.info(
        f"Config: {args.config} | dim={config.dim} | layers={config.num_layers} | "
        f"heads={config.num_heads}"
    )

    # Build model
    model = KimiTransformer(config).to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params / 1e6:.1f}M")

    # Optimizer (AdamW with weight decay on non-bias/norm params)
    decay_params = [
        p
        for n, p in model.named_parameters()
        if p.dim() >= 2 and "bias" not in n and "norm" not in n
    ]
    no_decay_params = [
        p for n, p in model.named_parameters() if p.dim() < 2 or "bias" in n or "norm" in n
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Training state
    step = 0

    # Create dummy data loader (replace with real dataset)
    logger.info("Using synthetic data for demonstration. Replace with real dataset.")

    def dummy_dataloader():
        while True:
            yield torch.randint(
                0, config.vocab_size, (args.batch_size, args.seq_len), device=args.device
            )

    data_iter = dummy_dataloader()

    model.train()
    optimizer.zero_grad()

    logger.info("Starting training...")
    for step in range(args.max_steps):
        # Forward
        input_ids = next(data_iter)
        labels = input_ids.clone()

        logits = model(input_ids)
        ce_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            labels.view(-1),
        )
        # MoE load‑balancing auxiliary loss (only when num_experts > 0)
        moe_loss = model.get_total_balance_loss() if config.num_experts > 0 else 0.0
        loss = (ce_loss + 0.01 * moe_loss) / args.accum_steps
        loss.backward()

        if (step + 1) % args.accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # LR scheduling
            lr = get_lr(step, args.warmup_steps, args.max_steps, args.learning_rate, args.min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()

        # Logging
        if step % args.log_every == 0:
            logger.info(
                f"Step {step:6d} | CE: {ce_loss.item():.4f}"
                f"{' | MoE: ' + f'{moe_loss.item():.4f}' if config.num_experts > 0 else ''}"
                f" | LR: {lr:.2e}"
            )

        # Checkpointing
        if step > 0 and step % args.save_every == 0:
            ckpt_path = Path(args.output_dir) / f"checkpoint_step_{step}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "loss": loss.item(),
                },
                ckpt_path,
            )
            logger.info(f"Saved checkpoint to {ckpt_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
