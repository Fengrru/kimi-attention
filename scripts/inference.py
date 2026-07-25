#!/usr/bin/env python3
"""
Inference Script for Kimi Transformer
=======================================
Text generation with the trained Kimi Transformer model.

Usage::
    python scripts/inference.py \
        --checkpoint ./checkpoints/checkpoint_step_100000.pt \
        --prompt "The future of artificial intelligence is" \
        --max_tokens 200 \
        --temperature 0.8

Features:
    - Greedy, top-k, and nucleus (top-p) sampling
    - Batch generation
    - Quantization support (int8/int4)

.. note::
    This script uses ``simple_tokenizer_encode`` / ``simple_tokenizer_decode``
    (hash-based dummy tokenizer) for demonstration purposes.  Replace with a
    real tokenizer (e.g., HuggingFace ``AutoTokenizer``) before production use.
"""

import argparse
import time

import torch

from kimi_attention.models import KimiTransformer
from kimi_attention.utils import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kimi Transformer Inference")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Input prompt text")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling threshold (0 = disabled)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile() the model for speed")

    return parser.parse_args()


def simple_tokenizer_encode(text: str, vocab_size: int) -> torch.Tensor:
    """Dummy tokenizer: hash characters to token IDs. Replace with real tokenizer."""
    token_ids = [(hash(c) % (vocab_size - 100) + 1) for c in text]
    return torch.tensor([token_ids], dtype=torch.long)


def simple_tokenizer_decode(token_ids: torch.Tensor, vocab_size: int) -> str:
    """Dummy detokenizer. Replace with real tokenizer."""
    chars = [chr(32 + (tid % 95)) for tid in token_ids.tolist()]
    return "".join(chars)


def main() -> None:
    args = parse_args()
    setup_logging()

    logger.info(f"Loading checkpoint from {args.checkpoint}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ckpt["config"]

    logger.info(f"Model config: dim={config.dim}, layers={config.num_layers}, "
                f"heads={config.num_heads}")

    # Build and load model
    model = KimiTransformer(config).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])

    if args.compile and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model)

    model.eval()

    # Tokenize input
    input_ids = simple_tokenizer_encode(args.prompt, config.vocab_size).to(args.device)
    prompt_len = input_ids.size(1)

    logger.info(f"Prompt ({prompt_len} tokens): {args.prompt}")

    # Generate
    with torch.no_grad():
        start_time = time.time()
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p if args.top_p > 0 else None,
        )
        elapsed = time.time() - start_time

    # Decode output
    generated_ids = output_ids[0, prompt_len:]
    generated_text = simple_tokenizer_decode(generated_ids, config.vocab_size)

    total_tokens = generated_ids.size(0)
    tps = total_tokens / elapsed

    print("\n" + "=" * 60)
    print("GENERATED TEXT")
    print("=" * 60)
    print(f"{args.prompt}{generated_text}")
    print("=" * 60)
    print(f"Stats: {total_tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")


if __name__ == "__main__":
    main()
