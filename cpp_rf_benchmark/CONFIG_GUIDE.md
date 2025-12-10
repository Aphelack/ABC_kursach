# Benchmark Configuration Guide

## Overview

The benchmark can be configured dynamically using a JSON configuration file. The configuration supports various formats for specifying test pairs of (n_estimators, n_threads).

## Configuration File Format

### Basic Parameters

- `max_depth` (int): Maximum depth of decision trees
- `num_runs` (int): Number of runs per configuration for averaging
- `random_state` (int): Random seed for reproducibility
- `test_size` (double): Fraction of data to use for testing (0.0 to 1.0)

### Test Pairs Configuration

You can specify test pairs in several ways:

#### 1. Direct Pairs (Recommended)

Explicitly define each (n_estimators, n_threads) pair:

```json
{
  "max_depth": 15,
  "num_runs": 3,
  "random_state": 42,
  "test_size": 0.2,
  "test_pairs": [
    {"n_estimators": 1, "n_threads": 1},
    {"n_estimators": 10, "n_threads": 10},
    {"n_estimators": 50, "n_threads": -1},
    {"n_estimators": 100, "n_threads": -1}
  ]
}
```

#### 2. Single n_threads, Multiple n_estimators

Test different estimator counts with a constant thread count:

```json
{
  "max_depth": 15,
  "num_runs": 3,
  "random_state": 42,
  "test_size": 0.2,
  "n_estimators": [10, 50, 100],
  "n_threads": 4
}
```

This creates pairs: (10, 4), (50, 4), (100, 4)

#### 3. Single n_estimators, Multiple n_threads

Test different thread counts with a constant estimator count:

```json
{
  "max_depth": 15,
  "num_runs": 3,
  "random_state": 42,
  "test_size": 0.2,
  "n_estimators": 100,
  "n_threads": [1, 2, 4, 8, -1]
}
```

This creates pairs: (100, 1), (100, 2), (100, 4), (100, 8), (100, -1)

#### 4. Paired by Index

If both arrays have the same length, pairs are created by matching indices:

```json
{
  "max_depth": 15,
  "num_runs": 3,
  "random_state": 42,
  "test_size": 0.2,
  "n_estimators": [10, 50, 100],
  "n_threads": [1, 4, -1]
}
```

This creates pairs: (10, 1), (50, 4), (100, -1)

#### 5. Cartesian Product

If arrays have different lengths (and neither is length 1), all combinations are tested:

```json
{
  "max_depth": 15,
  "num_runs": 3,
  "random_state": 42,
  "test_size": 0.2,
  "n_estimators": [10, 50],
  "n_threads": [1, 2, 4]
}
```

This creates pairs: (10, 1), (10, 2), (10, 4), (50, 1), (50, 2), (50, 4)

## Thread Count Values

- **Positive integer (e.g., 1, 2, 4, 8)**: Use exactly that many threads
- **-1**: Use all available CPU cores (recommended for maximum performance)

## Usage

### Basic Usage

```bash
./rf_benchmark
```

This will look for `benchmark_config.json` in the current directory. If not found, default configuration is used.

### Custom Config File

```bash
./rf_benchmark path/to/config.json
```

### Custom Config and Dataset

```bash
./rf_benchmark path/to/config.json path/to/dataset.csv
```

## Example Configurations

### Example 1: Testing Scalability

Test how performance scales with estimators using all cores:

```json
{
  "max_depth": 15,
  "num_runs": 5,
  "random_state": 42,
  "test_size": 0.2,
  "n_estimators": [10, 25, 50, 100, 200],
  "n_threads": -1
}
```

### Example 2: Testing Parallelization

Test how parallelization affects a fixed number of estimators:

```json
{
  "max_depth": 15,
  "num_runs": 5,
  "random_state": 42,
  "test_size": 0.2,
  "n_estimators": 100,
  "n_threads": [1, 2, 4, 8, 16, -1]
}
```

### Example 3: Finding Optimal Configuration

Test specific combinations to find the best trade-off:

```json
{
  "max_depth": 15,
  "num_runs": 3,
  "random_state": 42,
  "test_size": 0.2,
  "test_pairs": [
    {"n_estimators": 50, "n_threads": 4},
    {"n_estimators": 100, "n_threads": 8},
    {"n_estimators": 200, "n_threads": -1}
  ]
}
```

## Notes

- The configuration file must be valid JSON
- If the configuration file is not found or invalid, default values are used
- Default configuration: test_pairs = [(50, -1), (100, -1), (200, -1)]
- Results are saved to a JSON file with timestamp and CPU information
