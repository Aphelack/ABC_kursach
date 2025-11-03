#!/bin/bash
# Build script for Random Forest Benchmark

set -e

echo "=================================="
echo "Building Random Forest Benchmark"
echo "=================================="

# Create build directory
if [ -d "build" ]; then
    echo "Cleaning existing build directory..."
    rm -rf build
fi

mkdir build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building..."
make -j$(nproc)

echo ""
echo "✅ Build complete!"
echo "Executable: build/rf_benchmark"
echo ""
echo "Usage:"
echo "  ./build/rf_benchmark                    # Run with synthetic data"
echo "  ./build/rf_benchmark data.csv           # Run with CSV file"
echo ""
