@echo off
title SKYC AML Detection Platform
echo ============================================================
echo   SKYC ANTI-MONEY LAUNDERING DETECTION PLATFORM
echo ============================================================
cd /d "%~dp0"
if not exist ".venv\" (
    echo [SETUP] Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo [SETUP] Installing dependencies...
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
echo.
echo  Starting application at http://localhost:8501
echo.
streamlit run app.py
pause
