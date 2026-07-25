# Windows PowerShell lint script
$ErrorActionPreference = "Stop"

Write-Host "=== black (formatting) ===" -ForegroundColor Cyan
black --check --diff kimi_attention/ tests/ scripts/ examples/
if ($LASTEXITCODE -ne 0) { Write-Host "black failed!" -ForegroundColor Red }

Write-Host "`n=== isort (import ordering) ===" -ForegroundColor Cyan
isort --check-only --diff kimi_attention/ tests/ scripts/ examples/
if ($LASTEXITCODE -ne 0) { Write-Host "isort failed!" -ForegroundColor Red }

Write-Host "`n=== flake8 (linting) ===" -ForegroundColor Cyan
flake8 kimi_attention/ tests/ scripts/ examples/ --max-line-length=100 --count --show-source --statistics
if ($LASTEXITCODE -ne 0) { Write-Host "flake8 failed!" -ForegroundColor Red }

Write-Host "`n=== mypy (type checking) ===" -ForegroundColor Cyan
mypy kimi_attention/ --ignore-missing-imports
if ($LASTEXITCODE -ne 0) { Write-Host "mypy failed!" -ForegroundColor Red }

Write-Host "`n=== pytest (unit tests) ===" -ForegroundColor Cyan
pytest tests/ -v --tb=short
if ($LASTEXITCODE -ne 0) { Write-Host "pytest failed!" -ForegroundColor Red; exit 1 }

Write-Host "`nAll checks passed! ✓" -ForegroundColor Green
