#!/bin/bash
# Setup script for scalability analysis tools
# This script ensures all dependencies are installed

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Random Forest Scalability Analysis Setup"
echo "=========================================="
echo ""

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.8 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Found Python $PYTHON_VERSION"

# Check if packages are already installed
echo ""
echo "Checking Python dependencies..."

MISSING_PACKAGES=()

python3 -c "import numpy" 2>/dev/null || MISSING_PACKAGES+=("numpy")
python3 -c "import matplotlib" 2>/dev/null || MISSING_PACKAGES+=("matplotlib")
python3 -c "import seaborn" 2>/dev/null || MISSING_PACKAGES+=("seaborn")

if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
    echo "✓ All Python packages already installed!"
else
    echo ""
    echo "Missing packages: ${MISSING_PACKAGES[*]}"
    echo ""
    echo "You can install them using one of these methods:"
    echo ""
    echo "Method 1: System packages (Recommended for Debian/Ubuntu):"
    echo "  sudo apt install python3-numpy python3-matplotlib python3-seaborn"
    echo ""
    echo "Method 2: Using pip with virtual environment:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    echo ""
    echo "Method 3: Using pip with --break-system-packages (Not recommended):"
    echo "  pip3 install --break-system-packages -r requirements.txt"
    echo ""
    
    read -p "Would you like to try Method 1 (apt install)? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing via apt..."
        sudo apt update
        sudo apt install -y python3-numpy python3-matplotlib python3-seaborn
        
        # Verify installation
        if python3 -c "import numpy, matplotlib, seaborn" 2>/dev/null; then
            echo "✓ Successfully installed all packages!"
        else
            echo "⚠ Installation completed but packages still not found."
            echo "  Try Method 2 (virtual environment)"
            exit 1
        fi
    else
        echo ""
        echo "Please install packages manually using one of the methods above."
        echo "Then run this script again to verify."
        exit 1
    fi
fi

echo ""
echo "Verifying package versions..."
python3 << 'EOF'
import numpy as np
import matplotlib
import seaborn as sns

print(f"  numpy:      {np.__version__}")
print(f"  matplotlib: {matplotlib.__version__}")
print(f"  seaborn:    {sns.__version__}")
EOF

echo ""
echo "✓ Setup complete!"
echo ""
echo "You can now run:"
echo "  ./run_scalability.sh              - Run experiments"
echo "  python3 example_analysis.py       - Interactive help"
echo "  python3 scalability_analysis.py <report.json> - Analyze results"
echo ""
