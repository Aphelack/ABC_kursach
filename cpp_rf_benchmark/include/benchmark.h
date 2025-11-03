#pragma once

#include "types.h"
#include "random_forest.h"
#include "cpu_monitor.h"
#include <vector>

namespace rf_benchmark {

class Benchmark {
public:
    Benchmark(const BenchmarkConfig& config);
    
    std::vector<BenchmarkResult> run(const TrainTestSplit& data);
    
private:
    BenchmarkResult run_single_config(
        int n_estimators,
        int n_threads,
        const TrainTestSplit& data
    );
    
    BenchmarkConfig config_;
};

} // namespace rf_benchmark
