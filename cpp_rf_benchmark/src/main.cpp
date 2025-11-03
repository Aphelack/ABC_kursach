#include "random_forest.h"
#include "data_loader.h"
#include "benchmark.h"
#include "report_generator.h"
#include "utils.h"
#include <iostream>
#include <string>

using namespace rf_benchmark;

int main(int argc, char* argv[]) {
    std::cout << "\n===========================================\n";
    std::cout << "  Random Forest Benchmark - C++ Edition\n";
    std::cout << "===========================================\n" << std::endl;
    
    // Get system information
    SystemInfo system_info = Utils::get_system_info();
    
    std::cout << "System Information:" << std::endl;
    std::cout << "  CPU: " << system_info.cpu_model << std::endl;
    std::cout << "  Architecture: " << system_info.architecture << std::endl;
    std::cout << "  Logical cores: " << system_info.logical_cores << std::endl;
    std::cout << "  Physical cores: " << system_info.physical_cores << std::endl;
    std::cout << "  Memory: " << (system_info.total_memory / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    std::cout << "  OS: " << system_info.os_name << " " << system_info.os_version << std::endl;
    
    // Configure benchmark
    BenchmarkConfig config;
    config.n_estimators = {50, 100, 200};
    config.max_depth = 15;  // Increased depth for better performance
    config.num_runs = 3;
    config.random_state = 42;
    config.test_size = 0.2;
    config.n_threads = -1; // Use all cores
    
    // Load or generate data
    Dataset data;
    std::string dataset_name = "creditcard";
    
    // Use credit card fraud dataset by default (1% sample)
    std::string creditcard_path = "../dataset/creditcard.csv";
    
    if (argc > 1) {
        creditcard_path = argv[1];
    }
    
    try {
        std::cout << "\nLoading credit card fraud dataset from: " << creditcard_path << std::endl;
        std::cout << "Using full dataset for benchmarking..." << std::endl;
        data = DataLoader::load_creditcard_data(creditcard_path, 1.0);  // Full dataset
        dataset_name = creditcard_path;
        std::cout << "Dataset loaded successfully!" << std::endl;
        std::cout << "  Samples: " << data.n_samples << std::endl;
        std::cout << "  Features: " << data.n_features << " (Time + V1-V28 + Amount)" << std::endl;
        std::cout << "  Classes: " << data.n_classes << " (0=Normal, 1=Fraud)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << std::endl;
        std::cerr << "Using synthetic data instead." << std::endl;
        data = DataLoader::generate_synthetic_data(1000, 20, 2, 42);
        dataset_name = "synthetic";
        std::cout << "  Samples: 1000" << std::endl;
        std::cout << "  Features: 20" << std::endl;
        std::cout << "  Classes: 2" << std::endl;
    }
    
    // Normalize data
    DataLoader::normalize(data);
    
    // Train-test split
    std::cout << "\nSplitting data..." << std::endl;
    TrainTestSplit split = DataLoader::train_test_split(data, config.test_size, config.random_state);
    
    // Run benchmark
    Benchmark benchmark(config);
    std::vector<BenchmarkResult> results = benchmark.run(split);
    
    // Generate report with CPU name in filename
    std::string cpu_name = Utils::get_sanitized_cpu_name();
    std::string report_filename = Utils::get_report_filename(cpu_name);
    
    ReportGenerator::generate_json_report(
        report_filename,
        system_info,
        config,
        results,
        split.train,
        split.test
    );
    
    // Print summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "BENCHMARK SUMMARY" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::cout << "Results:" << std::endl;
    for (const auto& result : results) {
        std::cout << "\n  " << result.n_estimators << " estimators:" << std::endl;
        std::cout << "    Time: " << result.avg_time << " ± " << result.std_time << " sec" << std::endl;
        std::cout << "    F1-Score: " << (result.avg_f1_score * 100.0) << " ± " 
                  << (result.std_f1_score * 100.0) << " %" << std::endl;
        std::cout << "    CPU Usage: " << result.avg_cpu_usage << "% (max: " 
                  << result.max_cpu_usage << "%)" << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark completed successfully!" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return 0;
}
