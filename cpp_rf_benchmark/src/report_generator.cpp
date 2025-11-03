#include "report_generator.h"
#include "utils.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

namespace rf_benchmark {

void ReportGenerator::generate_json_report(
    const std::string& filename,
    const SystemInfo& system_info,
    const BenchmarkConfig& config,
    const std::vector<BenchmarkResult>& results,
    const Dataset& train_data,
    const Dataset& test_data
) {
    json report;
    
    // System info
    report["system_info"] = {
        {"cpu_model", system_info.cpu_model},
        {"architecture", system_info.architecture},
        {"logical_cores", system_info.logical_cores},
        {"physical_cores", system_info.physical_cores},
        {"total_memory", system_info.total_memory},
        {"os_name", system_info.os_name},
        {"os_version", system_info.os_version}
    };
    
    // Dataset info
    report["dataset"] = {
        {"train_samples", train_data.n_samples},
        {"test_samples", test_data.n_samples},
        {"n_features", train_data.n_features},
        {"n_classes", train_data.n_classes}
    };
    
    // Configuration
    report["config"] = {
        {"n_estimators", config.n_estimators},
        {"max_depth", config.max_depth},
        {"num_runs", config.num_runs},
        {"random_state", config.random_state},
        {"test_size", config.test_size},
        {"n_threads", config.n_threads}
    };
    
    // Results
    json results_json = json::array();
    for (const auto& result : results) {
        json result_json = {
            {"n_estimators", result.n_estimators},
            {"avg_time", result.avg_time},
            {"std_time", result.std_time},
            {"avg_f1_score", result.avg_f1_score},
            {"std_f1_score", result.std_f1_score},
            {"avg_cpu_usage", result.avg_cpu_usage},
            {"max_cpu_usage", result.max_cpu_usage},
            {"run_times", result.run_times},
            {"run_f1_scores", result.run_f1_scores},
            {"cpu_metrics", {
                {"n_cores", result.cpu_metrics.n_cores},
                {"avg_usage", result.cpu_metrics.avg_usage},
                {"max_usage", result.cpu_metrics.max_usage},
                {"per_core_usage", result.cpu_metrics.per_core_usage},
                {"timeline", result.cpu_metrics.timeline}
            }}
        };
        results_json.push_back(result_json);
    }
    report["results"] = results_json;
    
    // Summary
    if (!results.empty()) {
        // Find best configurations
        size_t best_f1_idx = 0;
        size_t fastest_idx = 0;
        double best_efficiency = 0.0;
        size_t most_efficient_idx = 0;
        
        for (size_t i = 0; i < results.size(); ++i) {
            if (results[i].avg_f1_score > results[best_f1_idx].avg_f1_score) {
                best_f1_idx = i;
            }
            if (results[i].avg_time < results[fastest_idx].avg_time) {
                fastest_idx = i;
            }
            double efficiency = results[i].avg_f1_score / results[i].avg_time;
            if (efficiency > best_efficiency) {
                best_efficiency = efficiency;
                most_efficient_idx = i;
            }
        }
        
        report["summary"] = {
            {"best_f1_score", results[best_f1_idx].avg_f1_score},
            {"best_f1_score_estimators", results[best_f1_idx].n_estimators},
            {"fastest_time", results[fastest_idx].avg_time},
            {"fastest_estimators", results[fastest_idx].n_estimators},
            {"most_efficient_estimators", results[most_efficient_idx].n_estimators},
            {"timestamp", Utils::get_timestamp()}
        };
    }
    
    // Write to file
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file << report.dump(2);
    file.close();
    
    std::cout << "\n📄 Report saved to: " << filename << std::endl;
}

} // namespace rf_benchmark
