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
    std::vector<double> read_cpu_usage();
    double read_total_cpu_usage();
    
    int n_cores_;
    std::atomic<bool> running_;
    std::thread monitor_thread_;
    
    mutable std::mutex data_mutex_;
    std::vector<double> timeline_;
    std::vector<std::vector<double>> per_core_timeline_;
    
    // For calculating CPU usage on Linux
    struct CPUStats {
        long long user, nice, system, idle, iowait, irq, softirq;
    };
    
    std::vector<CPUStats> prev_stats_;
    CPUStats read_cpu_stats(int core = -1);
};

} // namespace rf_benchmark
