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
    
    // Load configuration from file or use defaults
    BenchmarkConfig config;
    std::string config_file = "benchmark_config.json";
    std::string dataset_path = "../dataset/creditcard.csv";
    
    // Parse command line arguments
    if (argc > 1) {
        config_file = argv[1];
    }
    if (argc > 2) {
        dataset_path = argv[2];
    }
    
    // Load configuration
    std::cout << "\nLoading configuration..." << std::endl;
    config = Utils::load_config_from_file(config_file);
    
    // Load or generate data
    Dataset data;
    std::string dataset_name = "creditcard";
    
    try {
        std::cout << "\nLoading credit card fraud dataset from: " << dataset_path << std::endl;
        std::cout << "Using full dataset for benchmarking..." << std::endl;
        data = DataLoader::load_creditcard_data(dataset_path, 1.0);  // Full dataset
        dataset_name = dataset_path;
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
        std::cout << "\n  " << result.n_estimators << " estimators, " 
                  << result.n_threads << " threads:" << std::endl;
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
