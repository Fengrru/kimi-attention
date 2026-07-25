# Contributing to kimi-attention

Thank you for your interest in contributing! This document provides guidelines
for contributing to the project.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Fengrru/kimi-attention.git
cd kimi-attention

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install with development dependencies
pip install -e ".[dev,flash]"

# Run full quality checks
bash dev/lint.sh       # Linux/Mac
powershell dev/lint.ps1  # Windows

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=kimi_attention --cov-report=html
```

## Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting (`black kimi_attention/ tests/`)
- **isort**: Import sorting (`isort kimi_attention/ tests/`)
- **flake8**: Linting (`flake8 kimi_attention/ tests/`)
- **mypy**: Type checking (`mypy kimi_attention/`)

All checks must pass before a PR can be merged.

## Pull Request Process

1. **Fork and branch**: Create a feature branch from `master`
2. **Write tests**: Add tests for new functionality
3. **Pass checks**: Ensure all style and test checks pass
4. **Update docs**: Update README.md if adding new features
5. **Submit PR**: Provide a clear description of changes

## Testing Guidelines

- All new code must include unit tests
- Tests should cover: forward pass, gradient flow, edge cases
- Use `pytest.mark.parametrize` for testing multiple configurations
- Aim for >90% code coverage

## Reporting Issues

When reporting bugs, please include:

- Python and PyTorch versions
- Minimal reproduction code
- Expected vs. actual behavior
- Full error traceback

## Commit Messages

Use conventional commit format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements

Example: `feat: add support for tensor parallelism`

## License

By contributing, you agree that your contributions will be licensed under
the Apache 2.0 License.
