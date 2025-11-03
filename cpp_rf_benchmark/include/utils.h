#pragma once

#include "types.h"
#include <string>
#include <chrono>

namespace rf_benchmark {

class Utils {
public:
    // System information
    static SystemInfo get_system_info();
    static std::string get_cpu_model();
    static std::string get_sanitized_cpu_name();
    static int get_logical_cores();
    static int get_physical_cores();
    static long get_total_memory();
    
    // Time utilities
    static std::string get_timestamp();
    
    // Math utilities
    static double mean(const std::vector<double>& values);
    static double std_dev(const std::vector<double>& values);
    static double accuracy(const Labels& y_true, const Labels& y_pred);
    static double f1_score(const Labels& y_true, const Labels& y_pred, int positive_class = 1);
    static double precision(const Labels& y_true, const Labels& y_pred, int positive_class = 1);
    static double recall(const Labels& y_true, const Labels& y_pred, int positive_class = 1);
    
    // Random utilities
    static void set_seed(unsigned int seed);
    static int random_int(int min, int max);
    static double random_double(double min = 0.0, double max = 1.0);
    
    // File utilities
    static bool file_exists(const std::string& filename);
    static std::string get_report_filename(const std::string& cpu_name);
};

class Timer {
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end_time - start_time;
        return diff.count();
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_time;
};

} // namespace rf_benchmark
