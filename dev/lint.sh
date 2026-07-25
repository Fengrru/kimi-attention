#!/usr/bin/env bash
set -euo pipefail

echo "=== black (formatting) ==="
black --check --diff kimi_attention/ tests/ scripts/ examples/

echo ""
echo "=== isort (import ordering) ==="
isort --check-only --diff kimi_attention/ tests/ scripts/ examples/

echo ""
echo "=== flake8 (linting) ==="
flake8 kimi_attention/ tests/ scripts/ examples/ --count --show-source --statistics

echo ""
echo "=== mypy (type checking) ==="
mypy kimi_attention/ --ignore-missing-imports

echo ""
echo "=== pytest (unit tests) ==="
pytest tests/ -v --tb=short

echo ""
echo "All checks passed! ✓"
