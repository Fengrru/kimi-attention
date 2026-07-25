#!/usr/bin/env python3
"""
Lightweight Experiments for Kimi Attention
==========================================
Three experiments that run entirely on CPU with small models.

Usage::
    python experiments/run_all.py          # run all experiments
    python experiments/run_all.py --exp 1  # KDA vs MHA speed only
    python experiments/run_all.py --exp 2  # AttnRes ablation only
    python experiments/run_all.py --exp 3  # GQA compression only
"""

import argparse
import time
from collections import defaultdict

import torch

from kimi_attention.models import KimiConfig, KimiTransformer


def hr(num: float) -> str:
    """Human-readable number."""
    if abs(num) < 0.01:
        return f"{num:.5f}"
    if abs(num) < 10:
        return f"{num:.3f}"
    if abs(num) < 1000:
        return f"{num:.1f}"
    if abs(num) < 1_000_000:
        return f"{num/1e3:.1f}K"
    return f"{num/1e6:.1f}M"


def sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Experiment 1 — KDA vs MHA speed comparison
# ---------------------------------------------------------------------------
def exp1_kda_vs_mha():
    sep("Experiment 1: KDA vs MHA Speed vs Sequence Length")

    configs = {
        "KDA (all)": dict(kda_every=None),
        "3:1 Hybrid": dict(kda_every=4),
        "MHA (all)": dict(kda_every=1),
    }
    seq_lens = [128, 256, 512, 1024]
    gen_tokens = 32

    results = {name: {"prefill": [], "generate": []} for name in configs}

    for name, overrides in configs.items():
        cfg = KimiConfig(
            dim=128,
            num_layers=4,
            num_heads=4,
            vocab_size=500,
            max_seq_len=max(seq_lens) + gen_tokens,
            layers_per_block=2,
            **overrides,
        )
        model = KimiTransformer(cfg)
        model.eval()
        params = model.count_parameters()
        print(f"\n  [{name}]  params={hr(params)}")

        for sl in seq_lens:
            ids = torch.randint(0, cfg.vocab_size, (1, sl))

            # Prefill
            t0 = time.perf_counter()
            model.clear_caches()
            _ = model(ids)
            t1 = time.perf_counter()
            results[name]["prefill"].append(t1 - t0)

            # Generate
            prompt = torch.randint(0, cfg.vocab_size, (1, 8))
            t0 = time.perf_counter()
            model.clear_caches()
            _ = model.generate(prompt, max_new_tokens=gen_tokens)
            t1 = time.perf_counter()
            results[name]["generate"].append(t1 - t0)

            print(
                f"    seq={sl:5d}  prefill={results[name]['prefill'][-1]:.3f}s  "
                f"gen({gen_tokens})={results[name]['generate'][-1]:.3f}s"
            )

    # Summary table
    print(f"\n  {'Seq Len':>8s}", end="")
    for name in configs:
        print(f"  {name:>18s}", end="")
    print()

    for i, sl in enumerate(seq_lens):
        print(f"  {sl:>8d}", end="")
        for name in configs:
            pref = results[name]["prefill"][i]
            gen = results[name]["generate"][i]
            total = pref + gen
            print(f"  {total:.3f}s", end="")
        print()

    # Speedup vs MHA
    print("\n  Speedup vs all-MHA:")
    print(f"  {'Seq Len':>8s}", end="")
    for name in ["KDA (all)", "3:1 Hybrid"]:
        print(f"  {name:>18s}", end="")
    print()
    for i, sl in enumerate(seq_lens):
        mha_total = results["MHA (all)"]["prefill"][i] + results["MHA (all)"]["generate"][i]
        print(f"  {sl:>8d}", end="")
        for name in ["KDA (all)", "3:1 Hybrid"]:
            total = results[name]["prefill"][i] + results[name]["generate"][i]
            speedup = mha_total / total if total > 0 else 0
            print(f"  {speedup:>17.2f}x", end="")
        print()


# ---------------------------------------------------------------------------
# Experiment 2 — AttnRes ablation (loss curves)
# ---------------------------------------------------------------------------
def exp2_attnres_ablation():
    sep("Experiment 2: AttnRes Ablation — Training Loss Curves")

    torch.manual_seed(42)
    dim = 64
    num_layers = 4
    vocab_size = 64
    seq_len = 32
    batch_size = 2
    steps = 200
    lr = 1e-3

    # Simple data: random sequences, predict next char
    def data_batch():
        return torch.randint(0, vocab_size, (batch_size, seq_len))

    variants = {
        "With AttnRes": dict(layers_per_block=2),
        "No AttnRes": dict(layers_per_block=1),
    }

    losses = {name: [] for name in variants}

    for name, overrides in variants.items():
        config = KimiConfig(
            dim=dim,
            num_layers=num_layers,
            num_heads=4,
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            kda_every=1,  # all MHA for clean comparison
            **overrides,
        )
        model = KimiTransformer(config)
        params = model.count_parameters()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()

        print(
            f"\n  [{name}]  params={hr(params)}  layers_per_block={overrides['layers_per_block']}"
        )

        for step in range(steps):
            ids = data_batch()
            labels = ids.clone()

            logits = model(ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
            loss.backward()
            opt.step()
            opt.zero_grad()

            losses[name].append(loss.item())

            if step % 100 == 0 or step == steps - 1:
                print(f"    step {step:4d}  loss={loss.item():.4f}")

    # Summary
    print(f"\n  {'Step':>6s}  {'With AttnRes':>14s}  {'No AttnRes':>14s}  {'Diff':>10s}")
    for i in range(0, steps, 100):
        with_loss = losses["With AttnRes"][i]
        without_loss = losses["No AttnRes"][i]
        diff = without_loss - with_loss
        sign = "+" if diff > 0 else ""
        marker = " ★" if abs(diff) > 0.01 else ""
        print(f"  {i:>6d}  {with_loss:>14.4f}  {without_loss:>14.4f}  {sign}{diff:>9.4f}{marker}")

    final_with = losses["With AttnRes"][-1]
    final_without = losses["No AttnRes"][-1]
    print(f"\n  Final loss:  with={final_with:.4f}  without={final_without:.4f}")
    if final_with < final_without:
        print(
            f"  AttnRes wins by {final_without - final_with:.4f} "
            f"({final_without/final_with:.1f}x lower loss)"
        )
    else:
        print(f"  No significant difference ({final_with - final_without:.4f})")

    # ASCII chart
    width = 50
    max_loss = max(max(losses["With AttnRes"]), max(losses["No AttnRes"]))
    print(f"\n  Loss curves (0 → {steps} steps):")
    print(f"  {'With AttnRes':>14s}  {'No AttnRes':>14s}")
    for i in range(0, steps, max(1, steps // 20)):
        wl = losses["With AttnRes"][i]
        nl = losses["No AttnRes"][i]
        wb = int(wl / max(max_loss, 0.01) * width)
        nb = int(nl / max(max_loss, 0.01) * width)
        print(f"  step {i:4d}  {'#' * wb}{'.' * (width - wb)}  " f"{'#' * nb}{'.' * (width - nb)}")


# ---------------------------------------------------------------------------
# Experiment 3 — GQA compression trade-off
# ---------------------------------------------------------------------------
def exp3_gqa_compression():
    sep("Experiment 3: GQA — KV Cache vs Throughput Trade-off")

    dim = 128
    num_heads = 8
    num_layers = 2
    seq_len = 512
    gen_tokens = 32

    kv_head_configs = [1, 2, 4, 8]

    results = defaultdict(dict)

    for kv_heads in kv_head_configs:
        config = KimiConfig(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=kv_heads,
            vocab_size=2000,
            max_seq_len=seq_len + gen_tokens,
            kda_every=1,  # all MHA so we can measure pure KV cache effect
            layers_per_block=2,
        )
        model = KimiTransformer(config)
        model.eval()
        params = model.count_parameters()

        # KV cache size
        mem = model.estimate_memory(batch_size=1, seq_len=seq_len)
        kv_mb = mem["kv_cache_mb"]

        # Forward pass (prefill)
        ids = torch.randint(0, config.vocab_size, (1, seq_len))
        t0 = time.perf_counter()
        model.clear_caches()
        _ = model(ids)
        t1 = time.perf_counter()
        prefill_time = t1 - t0

        # Generate
        prompt = torch.randint(0, config.vocab_size, (1, 8))
        t0 = time.perf_counter()
        model.clear_caches()
        _ = model.generate(prompt, max_new_tokens=gen_tokens)
        t1 = time.perf_counter()
        gen_time = t1 - t0

        results[kv_heads] = {
            "params": params,
            "kv_cache_mb": kv_mb,
            "prefill_s": prefill_time,
            "gen_s": gen_time,
            "gen_tok_s": gen_tokens / gen_time if gen_time > 0 else 0,
        }

        print(
            f"  KV heads={kv_heads:2d}  params={hr(params)}  "
            f"KV cache={kv_mb:.1f}MB  prefill={prefill_time:.3f}s  "
            f"gen={gen_time:.3f}s ({gen_tokens/gen_time:.0f} tok/s)"
        )

    # Summary table
    print(
        f"\n  {'KV Heads':>10s}  {'KV Cache':>10s}  {'Prefill':>10s}  "
        f"{'Generate':>10s}  {'Tok/s':>10s}  {'Reduction':>12s}"
    )
    baseline_kv = results[8]["kv_cache_mb"]
    for kv_heads in kv_head_configs:
        r = results[kv_heads]
        reduction = (1 - r["kv_cache_mb"] / baseline_kv) * 100 if baseline_kv > 0 else 0
        print(
            f"  {kv_heads:>10d}  {r['kv_cache_mb']:>8.1f}MB  "
            f"{r['prefill_s']:>8.3f}s  {r['gen_s']:>8.3f}s  "
            f"{r['gen_tok_s']:>8.0f}  {reduction:>10.1f}%"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Lightweight Kimi Attention experiments")
    parser.add_argument(
        "--exp", type=int, choices=[1, 2, 3], default=0, help="Experiment to run (0 = all)"
    )
    args = parser.parse_args()

    print("Kimi Attention — Lightweight Experiments")
    print(f"Device: CPU  |  PyTorch {torch.__version__}")

    if args.exp == 0 or args.exp == 1:
        exp1_kda_vs_mha()
    if args.exp == 0 or args.exp == 2:
        exp2_attnres_ablation()
    if args.exp == 0 or args.exp == 3:
        exp3_gqa_compression()

    print(f"\n{'='*60}")
    print("  All experiments complete.")
    print(f"{'='*60}")
    print("\n  Notes:")
    print("  • KDA speed: pure-PyTorch fallback on CPU is slow; expect 3-6x")
    print("    speedup on GPU with FLA CUDA kernels at >8K context.")
    print("  • MHA appears faster here because F.scaled_dot_product_attention")
    print("    uses optimized CPU kernels; KDA uses a chunked scan in pure Python.")
    print("  • Rerun with --device cuda and pip install fla-core for realistic numbers.")
    print()


if __name__ == "__main__":
    main()
