#pragma once

#include "types.h"
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>

namespace rf_benchmark {

class CPUMonitor {
public:
    CPUMonitor(int n_cores);
    ~CPUMonitor();
    
    void start_monitoring();
    void stop_monitoring();
    CPUMetrics get_metrics() const;
    
private:
    void monitor_loop();
    std::vector<double> read_process_cpu_usage();
    double read_total_process_cpu_usage();
    
    int n_cores_;
    int pid_;
    std::atomic<bool> running_;
    std::thread monitor_thread_;
    
    mutable std::mutex data_mutex_;
    std::vector<double> timeline_;
    std::vector<std::vector<double>> per_core_timeline_;
    
    // For calculating process-specific CPU usage on Linux
    struct ProcessStats {
        long long utime;      // user mode time
        long long stime;      // kernel mode time
        long long cutime;     // children user mode time
        long long cstime;     // children kernel mode time
    };
    
    struct SystemStats {
        long long total_time;
    };
    
    ProcessStats prev_process_stats_;
    SystemStats prev_system_stats_;
    
    ProcessStats read_process_stats();
    SystemStats read_system_stats();
};

} // namespace rf_benchmark
