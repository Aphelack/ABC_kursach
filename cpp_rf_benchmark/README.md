# Random Forest Benchmark - C++ Edition

A high-performance multithreaded Random Forest implementation in C++ with comprehensive CPU monitoring and cross-processor comparison capabilities.

## Features

- **Multithreaded Random Forest**: Parallel tree training using thread pool with thread-local RNG
- **Real-time CPU Monitoring**: Per-core CPU usage tracking during training (100ms sampling)
- **F1-Score Evaluation**: Uses F1-score metric for imbalanced classification (fraud detection)
- **Stratified Sampling**: Maintains class balance when sampling from imbalanced datasets
- **JSON Reports**: Detailed benchmarking reports with processor name in filename
- **Cross-processor Comparison**: Python scripts for visualizing performance differences
- **Optimized Tree Building**: Smart threshold sampling (max 50 splits/feature) for faster training
- **Credit Card Fraud Detection**: Kaggle dataset (284K samples, 0.17% fraud) for realistic benchmarking

## Quick Start

### Download Dataset

```bash
cd ../dataset
wget https://storage.googleapis.com/kaggle-data-sets/310/684/compressed/creditcard.csv.zip
unzip creditcard.csv.zip
```

### Build and Run

```bash
cd ../cpp_rf_benchmark
./build.sh              # Compile project
./build/rf_benchmark    # Run on full dataset (~40-60s for 100 estimators)
```

### Visualize Results

```bash
python3 visualize_single.py rf_benchmark_<CPU_NAME>_<TIMESTAMP>.json
```

## Benchmark Configuration

Default settings (optimized for credit card fraud dataset):

```cpp
n_estimators: [50, 100, 200]   // Number of trees to test
max_depth: 15                   // Maximum tree depth
num_runs: 3                     // Runs per configuration
test_size: 0.2                  // 80/20 train/test split
n_threads: -1                   // Use all CPU cores
sample_fraction: 1.0            // Full dataset (284K samples)
```

### Performance Expectations

Based on sklearn baseline (100 estimators, full dataset = ~40s):

- **50 estimators**: ~15-20s
- **100 estimators**: ~30-40s  
- **200 estimators**: ~60-80s

Actual time varies by CPU. Our C++ implementation includes:
- Stratified sampling for class balance
- Max 50 thresholds per feature (vs sklearn's all unique values)
- Thread-local RNG for thread safety
- Parallel tree training with optimal core utilization

## Expected Results

**F1-Score**: 70-90% (highly imbalanced dataset, 0.17% fraud)
**CPU Usage**: 85-95% on multi-core systems
**Memory**: ~2-3 GB for full dataset

## Cross-Processor Comparison Workflow

### Step 1: Run on First Processor

```bash
# On Machine A (e.g., Intel i7)
./build/rf_benchmark
# Generates: rf_benchmark_Intel_Core_i7_9700K_<timestamp>.json
```

### Step 2: Run on Second Processor

```bash
# On Machine B (e.g., AMD Ryzen)
./build/rf_benchmark
# Generates: rf_benchmark_AMD_Ryzen_7_5800X_<timestamp>.json
```

### Step 3: Transfer and Compare

Copy both JSON files to one machine:

```bash
python3 compare_processors.py \
    rf_benchmark_Intel_Core_i7_9700K_<time1>.json \
    rf_benchmark_AMD_Ryzen_7_5800X_<time2>.json
```

Output shows:
- Speedup factors per configuration
- Time reduction percentages
- CPU utilization comparison
- Side-by-side bar charts
- Efficiency analysis (F1-score/sec)

## Project Structure

```
cpp_rf_benchmark/
├── include/              # Header files
│   ├── types.h          # Data structures
│   ├── utils.h          # Utility functions
│   ├── decision_tree.h  # Decision tree implementation
│   ├── random_forest.h  # Random forest classifier
│   ├── data_loader.h    # CSV and data generation
│   ├── cpu_monitor.h    # CPU usage monitoring
│   ├── benchmark.h      # Benchmarking framework
│   └── report_generator.h
├── src/                 # Source files
│   ├── main.cpp
│   ├── decision_tree.cpp
│   ├── random_forest.cpp
│   ├── data_loader.cpp
│   ├── cpu_monitor.cpp
│   ├── benchmark.cpp
│   ├── report_generator.cpp
│   └── utils.cpp
├── visualize_single.py   # Single run visualization
├── compare_processors.py # Two-processor comparison
├── build.sh             # Build script
├── run_benchmark.sh     # Run and visualize
└── CMakeLists.txt       # CMake configuration
```

## Requirements

### C++ Build
- CMake 3.15+
- C++17 compatible compiler (GCC 7+, Clang 5+)
- Linux OS (uses /proc for CPU monitoring)

### Python Visualization
- Python 3.6+
- matplotlib
- seaborn
- numpy

## Installation

### 1. Install Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake python3 python3-pip
pip3 install matplotlib seaborn numpy
```

**Fedora/RHEL:**
```bash
sudo dnf install gcc-c++ cmake python3 python3-pip
pip3 install matplotlib seaborn numpy
```

### 2. Build the Project

```bash
cd cpp_rf_benchmark
chmod +x build.sh run_benchmark.sh
./build.sh
```

This will:
- Download nlohmann/json library (header-only)
- Compile the project with optimizations
- Create `build/rf_benchmark` executable

## Usage

### Quick Start

Run benchmark with synthetic data and auto-visualization:
```bash
./run_benchmark.sh
```

### Manual Usage

#### 1. Run Benchmark

**With credit card fraud dataset (default):**
```bash
./build/rf_benchmark
```

This uses the full credit card fraud detection dataset from `../dataset/creditcard.csv`

**With custom CSV file:**
```bash
./build/rf_benchmark path/to/data.csv
```

CSV format: Last column = labels, other columns = features, first row = header

#### 2. Visualize Single Run

```bash
python3 visualize_single.py rf_benchmark_<CPU_NAME>_<TIMESTAMP>.json
```

This generates:
- Training time vs estimators
- CPU usage (average and maximum)
- F1-Score comparison
- Efficiency metrics (F1% / second)
- Per-core CPU heatmap
- Run variation plots
- System information table

#### 3. Compare Two Processors

After running benchmarks on two different machines:

```bash
python3 compare_processors.py report_cpu1.json report_cpu2.json
```

This shows:
- Side-by-side performance comparison
- Speedup factors
- Time reduction percentages
- CPU utilization differences
- Efficiency analysis
- Detailed comparison table

## Configuration

Edit `src/main.cpp` to customize benchmark parameters:

```cpp
BenchmarkConfig config;
config.n_estimators = {50, 100, 200};  // Trees to test
config.max_depth = 15;                  // Max tree depth
config.num_runs = 3;                    // Runs per config
config.random_state = 42;               // Random seed
config.test_size = 0.2;                 // 20% test split
config.n_threads = -1;                  // -1 = use all cores
```

To adjust sample size, edit `src/main.cpp`:
```cpp
// Use full dataset (284K samples)
data = DataLoader::load_creditcard_data(creditcard_path, 1.0);

// Use 10% sample for faster testing
data = DataLoader::load_creditcard_data(creditcard_path, 0.10);
```

## Report File Structure

JSON report includes:

```json
{
  "system_info": {
    "cpu_model": "Intel Core i7-9700K",
    "logical_cores": 8,
    "physical_cores": 8,
    "total_memory": 17179869184,
    ...
  },
  "config": {
    "n_estimators": [50, 100, 200],
    "max_depth": 20,
    ...
  },
  "results": [
    {
      "n_estimators": 50,
      "avg_time": 0.523,
      "avg_accuracy": 0.98,
      "avg_cpu_usage": 753.2,
      "cpu_metrics": {
        "per_core_usage": [95.2, 94.8, ...],
        "timeline": [10.5, 23.1, ...]
      },
      ...
    }
  ],
  "summary": {
    "best_accuracy_estimators": 200,
    "fastest_estimators": 50,
    ...
  }
}
```

**Note:** Report filename automatically includes CPU name for easy identification.

## Example Workflow

### Comparing Two Processors

**On Machine 1 (e.g., Intel i7):**
```bash
./build/rf_benchmark
# Generates: rf_benchmark_Intel_Core_i7-9700K_20251103_150000.json
```

**On Machine 2 (e.g., AMD Ryzen):**
```bash
./build/rf_benchmark
# Generates: rf_benchmark_AMD_Ryzen_7_5800X_20251103_151000.json
```

**Transfer both JSON files to one machine and compare:**
```bash
python3 compare_processors.py \
    rf_benchmark_Intel_Core_i7-9700K_20251103_150000.json \
    rf_benchmark_AMD_Ryzen_7_5800X_20251103_151000.json
```

## Algorithm Details

### Random Forest Implementation
- **Bootstrap Sampling**: Each tree trained on random sample with replacement
- **Feature Randomization**: Each split considers sqrt(n_features) random features
- **Parallel Training**: Trees trained in parallel using thread pool
- **Voting**: Final prediction by majority vote

### CPU Monitoring
- **Per-Core Tracking**: Monitors each CPU core separately
- **Timeline Recording**: 100ms sampling interval
- **Linux /proc/stat**: Reads CPU stats from kernel

### Benchmarking
- **Multiple Runs**: Averages over N runs to reduce variance
- **Metrics**: Training time, accuracy, CPU usage, std deviation
- **Configurations**: Tests multiple estimator counts

## Performance Tips

1. **Use Release Build**: CMake automatically uses `-O3 -march=native`
2. **Thread Count**: Default `-1` uses all cores, adjust if needed
3. **Data Size**: Larger datasets show better parallelization benefits
4. **Tree Depth**: Deeper trees = longer training but potentially better accuracy

## Troubleshooting

### Build Errors

**CMake not found:**
```bash
sudo apt-get install cmake
```

**Compiler too old:**
Requires C++17. Update GCC to 7+ or Clang to 5+

### Runtime Issues

**CPU monitoring returns zeros:**
- Only works on Linux with /proc/stat
- Ensure proper permissions

**Low CPU usage:**
- Dataset might be too small
- Try larger synthetic data or real dataset

### Visualization Issues

**Missing Python packages:**
```bash
pip3 install --upgrade matplotlib seaborn numpy
```

**Display errors (headless server):**
```bash
export MPLBACKEND=Agg  # Use non-interactive backend
```

## License

MIT License - Feel free to use and modify

## Contributing

Contributions welcome! Areas for improvement:
- Windows/macOS CPU monitoring
- More ML algorithms
- GPU acceleration
- Hyperparameter tuning

## Author

Created for ABC_kursach - Random Forest Performance Analysis
