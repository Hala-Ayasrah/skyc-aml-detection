#!/bin/bash
echo "============================================================"
echo "  SKYC ANTI-MONEY LAUNDERING DETECTION PLATFORM"
echo "============================================================"

# Navigate to the directory where the script is located
cd "$(dirname "$0")"

# Check if python3 is installed
if ! command -v python3 &> /dev/null
then
    echo "[ERROR] Python 3 could not be found. Please install Python 3."
    echo "Press Enter to exit..."
    read -r
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "[SETUP] Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

echo "[SETUP] Installing dependencies..."
python3 -m pip install --upgrade pip --quiet
python3 -m pip install -r requirements.txt --quiet

echo ""
echo " Starting application at http://localhost:8501"
echo ""
streamlit run app.py
