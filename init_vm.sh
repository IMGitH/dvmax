#!/bin/bash
set -e

echo "📁 Working in repo: $(pwd)"

echo "📦 Ensuring Git LFS is installed..."
if ! command -v git-lfs &> /dev/null; then
  sudo apt-get update
  sudo apt-get install -y git-lfs
fi
git lfs install

# Track parquet if not yet tracked
if ! grep -q '\.parquet' .gitattributes 2>/dev/null; then
  echo "📝 Tracking .parquet with Git LFS..."
  git lfs track "*.parquet"
  echo "*.parquet filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
  git add .gitattributes
  git commit -m "Track .parquet files with LFS" || echo "✅ No changes"
fi

# Create venv if missing
if [ ! -d ".venv" ]; then
  echo "🐍 Creating virtual environment..."
  python3 -m venv .venv
fi

VENV_PY=".venv/bin/python"

echo "🐍 Installing Python requirements into venv..."
$VENV_PY -m pip install --upgrade pip
$VENV_PY -m pip install -r requirements.txt

echo "🏗 Running macro and ticker feature generation..."
$VENV_PY src/dataprep/features/aggregation/macro_batch_runner.py
$VENV_PY src/dataprep/features/aggregation/ticker_batch_runner.py

echo "📤 Committing generated features..."
git config user.name "vm-automation"
git config user.email "automation@startup.vm"

git add features_data/**/*.parquet || true
git commit -m "📊 Auto-generated features" || echo "✅ Nothing to commit"
git push || echo "🚫 Nothing to push"
