#!/bin/bash

set -e

# --- Styles ---
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

clean_project() {
    echo -e "${BLUE}--- Cleaning Project Directory ---${NC}"

    if command -v deactivate &> /dev/null; then
        echo "🔌 Deactivating virtual environment..."
        deactivate
    fi

    echo "🗑️  Removing virtual environment (.venv)..."
    rm -rf .venv

    echo "🗑️  Removing generated files (weights.json, houses.csv)..."
    rm -f weights.json houses.csv

    echo "🗑️  Removing build files (egg-info, build)..."
    rm -rf build dslr.egg-info

    echo "🗑️  Removing Python cache files (__pycache__, *.pyc)..."
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

    echo -e "\n${GREEN}✅ Project cleaned successfully.${NC}"
    echo -e "${BLUE}------------------------------------${NC}"
}

if [ "$1" == "clean" ]; then
    clean_project
    exit 0
fi

echo -e "${BLUE}--- Starting DSLR Project Setup ---${NC}"

echo "🔎 Checking Python version..."
if ! python3 -c 'import sys; assert sys.version_info >= (3, 8)' &>/dev/null; then
    echo -e "${RED}ERROR: Python 3.8 or higher is required.${NC}"
    echo "Please install a compatible Python version and try again."
    exit 1
fi
echo -e "${GREEN}✅ Python version is compatible.${NC}"

VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    echo "♻️  Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
    echo "🐍 Creating virtual environment in '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✅ Virtual environment created.${NC}"
fi


if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    ACTIVATE_SCRIPT="$VENV_DIR/Scripts/activate"
else
    ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
fi

echo "🚀 Activating virtual environment..."
source "$ACTIVATE_SCRIPT"

echo "📦 Installing project dependencies from pyproject.toml..."
pip install --upgrade pip > /dev/null
pip install . > /dev/null
echo -e "${GREEN}✅ Dependencies installed successfully.${NC}"

echo "🔬 Verifying installation..."
if python -c "import pandas, numpy, matplotlib" &>/dev/null; then
    echo -e "${GREEN}✅ Core libraries (pandas, numpy, matplotlib) imported successfully.${NC}"
else
    echo -e "${RED}ERROR: Failed to import one of the core libraries. Installation may have failed.${NC}"
    exit 1
fi

echo "🔗 Checking for dependency conflicts..."
if ! pip check &>/dev/null; then
    echo -e "${YELLOW}⚠️  Warning: Dependency conflicts detected. Run 'pip check' for details.${NC}"
else
    echo -e "${GREEN}✅ No dependency conflicts found.${NC}"
fi

echo -e "\n${GREEN}🎉 Setup complete! The virtual environment is ready and active.${NC}\n"
echo "You can now run the project scripts, for example:"
echo "Application Usage:"
echo "   python3 dslr.py -d train.csv -t test.csv -v truth.csv"
echo ""
echo "DATA ANALYSIS"
echo "   python3 visualize/describe.py data/dataset_train.csv"
echo ""
echo "DATA VISUALIZATION"
echo "   python3 visualize/histogram.py data/dataset_train.csv"
echo "   python3 visualize/scatter_plot.py data/dataset_train.csv"
echo "   python3 visualize/pair_plot.py data/dataset_train.csv"
echo ""
echo "MODEL TRAINING / PREDICTION"
echo "   python logreg_train.py data/dataset_train.csv"
echo "   python logreg_predict.py data/dataset_test.csv weights.json"
echo ""
echo "To deactivate the virtual environment later, simply run:"
echo -e "  ${YELLOW}deactivate${NC}"
echo "To reactivate it, run:"
echo -e "  ${YELLOW}source $ACTIVATE_SCRIPT${NC}"
echo -e "\n${BLUE}------------------------------------${NC}"
