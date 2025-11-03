#pragma once

#include "types.h"
#include <vector>
#include <string>

namespace rf_benchmark {

class ReportGenerator {
public:
    static void generate_json_report(
        const std::string& filename,
        const SystemInfo& system_info,
        const BenchmarkConfig& config,
        const std::vector<BenchmarkResult>& results,
        const Dataset& train_data,
        const Dataset& test_data
    );
    
private:
    static std::string escape_json_string(const std::string& str);
};

} // namespace rf_benchmark
