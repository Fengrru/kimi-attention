# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-25

### Added

- **Attention Residuals (AttnRes)**: Block-wise depth attention aggregation that dynamically weights previous layers, drop-in compatible with any Transformer architecture.
- **Kimi Delta Attention (KDA)**: Fine-grained channel-wise gating with linear complexity, 6× long-context decode speed, -75% KV cache.
- **Hybrid architecture**: Official 3:1 KDA-to-MHA ratio, configurable via `kda_every`.
- **RMSNorm**: Root Mean Square Layer Normalization with FP32 internal computation.
- **RotaryEmbedding (RoPE)**: Precomputed rotary position embeddings with configurable theta.
- **MoEFeedForward**: Sparse Mixture-of-Experts FFN with top-k routing and load-balancing auxiliary loss.
- **KimiTransformer**: End-to-end model integrating AttnRes, KDA, RoPE, GQA, and MoE.
- **Configuration presets**: Official 1B, 7B, and 48B parameter configurations.
- **Generation**: Autoregressive sampling (greedy, top-k, nucleus) with KV-cache and KDA recurrent state.
- **Beam search**: Per-beam KV-cache incremental decoding with O(T) MHA complexity.
- **Training script**: Full pipeline with AMP, gradient clipping, cosine LR schedule, and checkpointing.
- **Benchmark tool**: Prefill and generation throughput measurement.
- **148 unit tests**: Comprehensive coverage of forward pass, gradient flow, and edge cases.
- **Examples**: Standalone AttnRes, KDA, and custom training examples.
- **FLA kernel integration**: Auto-detection of CUDA-optimized `chunk_kda` kernel with pure PyTorch fallback.

### Infrastructure

- **GitHub Actions CI**: Multi-OS (Ubuntu/Windows/macOS) + multi-Python (3.9–3.12) test matrix + lint workflow.
- **Pre-commit hooks**: Automated formatting (black, isort), linting (flake8), and type checking (mypy).
- **Dev scripts**: `dev/lint.sh` (bash) and `dev/lint.ps1` (PowerShell) for one-command quality checks.
- **Issue/PR templates**: Bug report, feature request, and pull request templates.
- **CODE_OF_CONDUCT.md**: Contributor Covenant v2.1.
- **SECURITY.md**: Vulnerability disclosure policy and security best practices.
- **.gitattributes**: LF normalization for all text files.
