#!/bin/bash
set -euo pipefail

trap 'echo "❌ Failed at line $LINENO"; exit 1' ERR

# --- move to repo root ---
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "📁 Working in repo: $(pwd)"

# --- venv ---
if [ ! -d ".venv" ]; then
  echo "🐍 Creating virtual environment..."
  python3 -m venv .venv
fi
VENV_PY=".venv/bin/python"

echo "🐍 Installing Python requirements into venv..."
$VENV_PY -m pip install --upgrade pip
$VENV_PY -m pip install -r requirements.txt

# Optional: install package in editable mode if pyproject exists (for clean imports/entry points)
if [ -f "pyproject.toml" ]; then
  echo "📦 Installing package in editable mode (pyproject.toml detected)..."
  $VENV_PY -m pip install -e .
fi

# --- make CLIs runnable ---
if compgen -G "cli/*.py" > /dev/null; then
  chmod +x cli/*.py || true
fi

# --- generate ---
echo "🏗 Running macro and ticker feature generation via CLI wrappers..."
$VENV_PY cli/macro_batch_runner.py
$VENV_PY cli/ticker_batch_runner.py

# --- package locally (optional helper if you run this script by hand) ---
ts=$(date -u +'%Y%m%d_%H%M%S')
pkg="features_${ts}.tar.gz"
echo "📦 Archiving generated features → $pkg"
tar -czf "$pkg" \
  features_data/macro_history \
  features_data/tickers_history \
  features_data/tickers_static \
  features_data/_invalid 2>/dev/null || true

echo "✅ Done."
echo "   If you’re running locally and want to publish this snapshot as a GitHub Release:"
echo "     gh release create data-$ts $pkg -t \"Data snapshot $ts\" -n \"Automated data snapshot.\""
