#include "benchmark.h"
#include "utils.h"
#include <iostream>
#include <chrono>

namespace rf_benchmark {

Benchmark::Benchmark(const BenchmarkConfig& config)
    : config_(config) {
}

std::vector<BenchmarkResult> Benchmark::run(const TrainTestSplit& data) {
    std::vector<BenchmarkResult> results;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "RANDOM FOREST BENCHMARK" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Test pairs (estimators, threads): ";
    for (const auto& pair : config_.test_pairs) {
        std::cout << "(" << pair.n_estimators << ", " << pair.n_threads << ") ";
    }
    std::cout << std::endl;
    std::cout << "  Max depth: " << config_.max_depth << std::endl;
    std::cout << "  Runs per config: " << config_.num_runs << std::endl;
    std::cout << "  Train samples: " << data.train.n_samples << std::endl;
    std::cout << "  Test samples: " << data.test.n_samples << std::endl;
    std::cout << "  Features: " << data.train.n_features << std::endl;
    std::cout << "  Classes: " << data.train.n_classes << "\n" << std::endl;
    
    for (const auto& pair : config_.test_pairs) {
        std::cout << "Testing " << pair.n_estimators << " estimators with " 
                  << pair.n_threads << " threads..." << std::endl;
        BenchmarkResult result = run_single_config(pair.n_estimators, pair.n_threads, data);
        results.push_back(result);
        
        std::cout << "  Avg time: " << result.avg_time << " sec" << std::endl;
        std::cout << "  Avg F1-score: " << (result.avg_f1_score * 100.0) << "%" << std::endl;
        std::cout << "  Avg CPU: " << result.avg_cpu_usage << "%" << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "Benchmark completed!\n" << std::endl;
    
    return results;
}

BenchmarkResult Benchmark::run_single_config(
    int n_estimators,
    int n_threads,
    const TrainTestSplit& data
) {
    BenchmarkResult result;
    result.n_estimators = n_estimators;
    result.n_threads = n_threads;
    result.run_times.resize(config_.num_runs);
    result.run_f1_scores.resize(config_.num_runs);
    
    CPUMonitor monitor(Utils::get_logical_cores());
    
    for (int run = 0; run < config_.num_runs; ++run) {
        Utils::set_seed(config_.random_state + run);
        
        RandomForest rf(
            n_estimators,
            config_.max_depth,
            n_threads,
            config_.random_state + run
        );
        
        // Start monitoring
        monitor.start_monitoring();
        
        // Time training
        auto start = std::chrono::high_resolution_clock::now();
        rf.fit(data.train.features, data.train.labels);
        auto end = std::chrono::high_resolution_clock::now();
        
        // Stop monitoring
        monitor.stop_monitoring();
        
        std::chrono::duration<double> elapsed = end - start;
        double train_time = elapsed.count();
        
        // Calculate F1-score
        double f1 = rf.f1_score(data.test.features, data.test.labels, 1); // positive_class=1 for fraud
        
        result.run_times[run] = train_time;
        result.run_f1_scores[run] = f1;
        
        std::cout << "    Run " << (run + 1) << ": " 
                  << train_time << " sec, "
                  << (f1 * 100.0) << "% F1" << std::endl;
    }
    
    // Calculate statistics
    result.avg_time = Utils::mean(result.run_times);
    result.std_time = Utils::std_dev(result.run_times);
    result.avg_f1_score = Utils::mean(result.run_f1_scores);
    result.std_f1_score = Utils::std_dev(result.run_f1_scores);
    
    // Get CPU metrics from last run
    result.cpu_metrics = monitor.get_metrics();
    result.avg_cpu_usage = result.cpu_metrics.avg_usage;
    result.max_cpu_usage = result.cpu_metrics.max_usage;
    
    return result;
}

} // namespace rf_benchmark
