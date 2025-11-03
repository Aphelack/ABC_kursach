#!/bin/bash
# Run benchmark and visualize results

set -e

echo "=================================="
echo "Random Forest Benchmark Runner"
echo "=================================="

# Build if needed
if [ ! -f "build/rf_benchmark" ]; then
    echo "Building project..."
    ./build.sh
fi

# Run benchmark
echo ""
echo "Running benchmark..."
./build/rf_benchmark "$@"

# Find the generated report
REPORT=$(ls -t rf_benchmark_*.json | head -1)

if [ -f "$REPORT" ]; then
    echo ""
    echo "=================================="
    echo "Generating visualization..."
    echo "=================================="
    python3 visualize_single.py "$REPORT"
else
    echo "⚠️  No report file found!"
    exit 1
fi
