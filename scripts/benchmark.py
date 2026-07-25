#!/usr/bin/env python3
"""
Performance Benchmark for Kimi Transformer
==========================================
Measures prefill and generation throughput for various model sizes.

Usage::
    python scripts/benchmark.py --config 1B --seq_len 2048 --gen_tokens 256
"""

import argparse
import time

import torch

from kimi_attention.configs import (
    KIMI_LINEAR_1B_CONFIG,
    KIMI_LINEAR_7B_CONFIG,
    KIMI_LINEAR_48B_CONFIG,
)
from kimi_attention.models import KimiConfig, KimiTransformer

CONFIG_MAP = {
    "1B": KIMI_LINEAR_1B_CONFIG,
    "7B": KIMI_LINEAR_7B_CONFIG,
    "48B": KIMI_LINEAR_48B_CONFIG,
    "custom": KimiConfig(dim=512, num_layers=8, num_heads=8),
}


def time_prefill(model, input_ids, warmup=2, repeats=5):
    model.clear_caches()
    for _ in range(warmup):
        _ = model(input_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.clear_caches()
        _ = model(input_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / repeats


def time_generate(model, input_ids, max_tokens, warmup=1, repeats=3):
    model.clear_caches()
    for _ in range(warmup):
        _ = model.generate(input_ids, max_new_tokens=max_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.clear_caches()
        _ = model.generate(input_ids, max_new_tokens=max_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / repeats


def main():
    parser = argparse.ArgumentParser(description="Benchmark Kimi Transformer")
    parser.add_argument(
        "--config", type=str, default="custom", choices=["1B", "7B", "48B", "custom"]
    )
    parser.add_argument("--dim", type=int, default=0)
    parser.add_argument("--num_layers", type=int, default=0)
    parser.add_argument("--num_heads", type=int, default=0)
    parser.add_argument("--num_kv_heads", type=int, default=0)
    parser.add_argument("--num_experts", type=int, default=0)
    parser.add_argument("--kda_every", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--gen_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    config = CONFIG_MAP[args.config]
    if args.dim:
        config.dim = args.dim
    if args.num_layers:
        config.num_layers = args.num_layers
    if args.num_heads:
        config.num_heads = args.num_heads
    if args.num_kv_heads:
        config.num_kv_heads = args.num_kv_heads
    if args.num_experts:
        config.num_experts = args.num_experts
    config.kda_every = args.kda_every

    print("=== Kimi Transformer Benchmark ===")
    print(f"  Config:   {args.config}")
    print(f"  Dim:      {config.dim}")
    print(f"  Layers:   {config.num_layers}")
    print(f"  Heads:    {config.num_heads} (KV: {config.kv_heads})")
    print(f"  MoE:      {config.num_experts} experts x top-{config.num_experts_per_tok}")
    print(f"  KDA every: {config.kda_every}")
    print(f"  Device:   {args.device}")
    print()

    model = KimiTransformer(config).to(args.device)
    params = model.count_parameters()
    print(f"  Parameters: {params / 1e6:.1f}M")

    input_ids = torch.randint(
        0, config.vocab_size, (args.batch_size, args.seq_len), device=args.device
    )

    # Prefill
    model.eval()
    prefill_time = time_prefill(model, input_ids)
    prefill_tok_s = (args.seq_len * args.batch_size) / prefill_time
    print(f"\n  Prefill  ({args.seq_len} tokens x {args.batch_size} batch):")
    print(f"    Time:       {prefill_time:.3f} s")
    print(f"    Throughput: {prefill_tok_s:.0f} tok/s")

    # Generation
    gen_input = torch.randint(0, config.vocab_size, (args.batch_size, 16), device=args.device)
    gen_time = time_generate(model, gen_input, args.gen_tokens)
    gen_tok_s = args.gen_tokens / gen_time
    print(f"\n  Generate ({args.gen_tokens} tokens):")
    print(f"    Time:       {gen_time:.3f} s")
    print(f"    Throughput: {gen_tok_s:.0f} tok/s")

    # Memory
    mem = model.estimate_memory(args.batch_size, args.seq_len)
    print(f"\n  Memory (FP32 estimate):")
    print(f"    Params:    {mem['parameters_mb']:.0f} MB")
    print(f"    KV Cache:  {mem['kv_cache_mb']:.0f} MB")
    print(f"    Total:     {mem['total_mb']:.0f} MB")


if __name__ == "__main__":
    main()
