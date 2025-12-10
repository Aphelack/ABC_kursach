#!/bin/bash
# Run complete scalability analysis for Random Forest benchmark
# This script runs three series of experiments:
# 1. Thread scaling (fixed estimators, varying threads)
# 2. Estimator scaling (fixed threads, varying estimators)
# 3. Dataset size scaling (varying dataset size)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

BENCHMARK_EXEC="./build/rf_benchmark"
DATASET="../dataset/creditcard.csv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Random Forest Scalability Analysis"
echo "========================================"
echo ""

# Check if benchmark executable exists
if [ ! -f "$BENCHMARK_EXEC" ]; then
    echo -e "${RED}Error: Benchmark executable not found at $BENCHMARK_EXEC${NC}"
    echo "Please run: ./build.sh first"
    exit 1
fi

# Check if dataset exists
if [ ! -f "$DATASET" ]; then
    echo -e "${RED}Error: Dataset not found at $DATASET${NC}"
    exit 1
fi

# Create output directory
OUTPUT_DIR="scalability_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}Output directory: $OUTPUT_DIR${NC}"
echo ""

# Function to run benchmark with a config
run_benchmark() {
    local config_file=$1
    local experiment_name=$2
    
    echo -e "${YELLOW}Running: $experiment_name${NC}"
    echo "Config: $config_file"
    
    # Run benchmark
    "$BENCHMARK_EXEC" "$config_file" "$DATASET"
    
    # Find the most recent JSON report
    LATEST_REPORT=$(ls -t rf_benchmark_*.json | head -1)
    
    if [ -f "$LATEST_REPORT" ]; then
        # Move report to output directory with descriptive name
        mv "$LATEST_REPORT" "$OUTPUT_DIR/${experiment_name}.json"
        echo -e "${GREEN}✓ Report saved: $OUTPUT_DIR/${experiment_name}.json${NC}"
    else
        echo -e "${RED}✗ Warning: Report not found${NC}"
    fi
    
    echo ""
}

# Experiment 1: Thread Scaling
echo "=========================================="
echo "Experiment 1: Thread Scaling"
echo "=========================================="
run_benchmark "config_scalability_threads.json" "exp1_thread_scaling"

# Experiment 2: Estimator Scaling
echo "=========================================="
echo "Experiment 2: Estimator Scaling"
echo "=========================================="
run_benchmark "config_scalability_estimators.json" "exp2_estimator_scaling"

# Note: Dataset size scaling requires code modification to support partial dataset loading
# For now, we'll mention it in the documentation

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "1. Analyze thread scaling:"
echo "   python3 scalability_analysis.py $OUTPUT_DIR/exp1_thread_scaling.json"
echo ""
echo "2. Analyze estimator scaling:"
echo "   python3 scalability_analysis.py $OUTPUT_DIR/exp2_estimator_scaling.json"
echo ""
echo "3. Compare results:"
echo "   python3 compare_scalability.py $OUTPUT_DIR/exp1_thread_scaling.json $OUTPUT_DIR/exp2_estimator_scaling.json"
echo ""

# Auto-generate visualizations if Python is available
if command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Generating visualizations...${NC}"
    echo ""
    
    # Install requirements if needed
    if ! python3 -c "import matplotlib" 2>/dev/null; then
        echo "Installing Python dependencies..."
        pip3 install -r requirements.txt --quiet
    fi
    
    # Generate analysis for each experiment
    for report in "$OUTPUT_DIR"/*.json; do
        if [ -f "$report" ]; then
            echo "Analyzing: $(basename $report)"
            python3 scalability_analysis.py "$report" --output-dir "${report%.json}_plots"
        fi
    done
    
    echo ""
    echo -e "${GREEN}✓ Visualizations generated!${NC}"
    echo ""
    echo "Check the following directories for plots:"
    ls -d "$OUTPUT_DIR"/*_plots 2>/dev/null || echo "No plot directories found"
else
    echo -e "${YELLOW}Python3 not found. Skipping automatic visualization.${NC}"
    echo "Run visualization manually with:"
    echo "  python3 scalability_analysis.py <report.json>"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "Scalability Analysis Complete!"
echo "==========================================${NC}"
