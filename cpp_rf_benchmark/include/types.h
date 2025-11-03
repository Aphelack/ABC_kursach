#pragma once

#include <vector>
#include <string>
#include <memory>

namespace rf_benchmark {

// Type aliases
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;
using Labels = std::vector<int>;
using FeatureIndices = std::vector<size_t>;

// Data structures
struct Dataset {
    Matrix features;
    Labels labels;
    size_t n_samples;
    size_t n_features;
    size_t n_classes;
    
    Dataset() : n_samples(0), n_features(0), n_classes(0) {}
};

struct TrainTestSplit {
    Dataset train;
    Dataset test;
};

struct EstimatorThreadPair {
    int n_estimators;
    int n_threads;
    
    EstimatorThreadPair(int est = 50, int threads = -1) 
        : n_estimators(est), n_threads(threads) {}
};

struct BenchmarkConfig {
    std::vector<EstimatorThreadPair> test_pairs;
    int max_depth;
    int num_runs;
    int random_state;
    double test_size;
    
    BenchmarkConfig() 
        : max_depth(20)
        , num_runs(3)
        , random_state(42)
        , test_size(0.2)
    {
        // Default configuration
        test_pairs.push_back(EstimatorThreadPair(50, -1));
        test_pairs.push_back(EstimatorThreadPair(100, -1));
        test_pairs.push_back(EstimatorThreadPair(200, -1));
    }
};

struct CPUMetrics {
    double avg_usage;
    double max_usage;
    std::vector<double> per_core_usage;
    std::vector<double> timeline;
    int n_cores;
    int n_threads;
};

struct BenchmarkResult {
    int n_estimators;
    int n_threads;
    double avg_time;
    double std_time;
    double avg_f1_score;
    double std_f1_score;
    double avg_cpu_usage;
    double max_cpu_usage;
    CPUMetrics cpu_metrics;
    std::vector<double> run_times;
    std::vector<double> run_f1_scores;
};

struct SystemInfo {
    std::string cpu_model;
    std::string architecture;
    int logical_cores;
    int physical_cores;
    long total_memory;
    std::string os_name;
    std::string os_version;
};

} // namespace rf_benchmark
