#!/usr/bin/env bash
# ============================================================
# Clinical Agentic AI — macOS / Linux one-shot setup
# ============================================================

set -e
cd "$(dirname "$0")"

echo "[1/4] Creating virtual environment (.venv)..."
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate

echo "[2/4] Installing dependencies..."
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo "[3/4] Generating sample dataset and golden table..."
python scripts/generate_sample_data.py

echo "[4/4] Creating .env (if missing)..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example."
    echo
    echo "--------------------------------------------------------"
    echo "  NEXT STEP: open .env and paste your ANTHROPIC_API_KEY"
    echo "  Get one at: https://console.anthropic.com/settings/keys"
    echo "--------------------------------------------------------"
else
    echo ".env already exists — leaving it alone."
fi

echo
echo "Done. To run the system:"
echo "  1) Terminal A:  source .venv/bin/activate && uvicorn backend.main:app --reload --port 8000"
echo "  2) Terminal B:  source .venv/bin/activate && streamlit run frontend/app.py"
