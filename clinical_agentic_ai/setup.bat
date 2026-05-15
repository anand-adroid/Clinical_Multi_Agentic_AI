@echo off
REM ============================================================
REM  Clinical Agentic AI — Windows one-shot setup
REM ============================================================
REM  - Creates a virtualenv
REM  - Installs all dependencies
REM  - Generates the sample dataset and golden file
REM  - Creates a starter .env file (you still need to paste your API key)
REM ============================================================

setlocal
cd /d "%~dp0"

echo.
echo [1/4] Creating virtual environment (.venv)...
if not exist ".venv" (
    python -m venv .venv
)

echo.
echo [2/4] Installing dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo.
echo [3/4] Generating sample dataset and golden table...
python scripts\generate_sample_data.py

echo.
echo [4/4] Creating .env (if missing)...
if not exist ".env" (
    copy .env.example .env >nul
    echo Created .env from .env.example.
    echo.
    echo --------------------------------------------------------
    echo  NEXT STEP: open .env and paste your ANTHROPIC_API_KEY
    echo  Get one at: https://console.anthropic.com/settings/keys
    echo --------------------------------------------------------
) else (
    echo .env already exists — leaving it alone.
)

echo.
echo Done. To run the system:
echo   1) Terminal A:  .venv\Scripts\activate ^&^& uvicorn backend.main:app --reload --port 8000
echo   2) Terminal B:  .venv\Scripts\activate ^&^& streamlit run frontend/app.py
echo.
endlocal
