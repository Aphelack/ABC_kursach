#include "cpu_monitor.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <unistd.h>

namespace rf_benchmark {

CPUMonitor::CPUMonitor(int n_cores) 
    : n_cores_(n_cores)
    , pid_(getpid())
    , running_(false) {
    prev_process_stats_ = {0, 0, 0, 0};
    prev_system_stats_ = {0};
}

CPUMonitor::~CPUMonitor() {
    if (running_) {
        stop_monitoring();
    }
}

void CPUMonitor::start_monitoring() {
    if (running_) return;
    
    running_ = true;
    timeline_.clear();
    per_core_timeline_.clear();
    per_core_timeline_.resize(n_cores_);
    
    // Initialize previous stats
    prev_process_stats_ = read_process_stats();
    prev_system_stats_ = read_system_stats();
    
    monitor_thread_ = std::thread(&CPUMonitor::monitor_loop, this);
}

void CPUMonitor::stop_monitoring() {
    if (!running_) return;
    
    running_ = false;
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
}

void CPUMonitor::monitor_loop() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        double total_usage = read_total_process_cpu_usage();
        std::vector<double> core_usage = read_process_cpu_usage();
        
        std::lock_guard<std::mutex> lock(data_mutex_);
        timeline_.push_back(total_usage);
        
        for (size_t i = 0; i < core_usage.size() && i < per_core_timeline_.size(); ++i) {
            per_core_timeline_[i].push_back(core_usage[i]);
        }
    }
}

CPUMonitor::ProcessStats CPUMonitor::read_process_stats() {
    ProcessStats stats = {0, 0, 0, 0};
    
    std::string stat_path = "/proc/" + std::to_string(pid_) + "/stat";
    std::ifstream file(stat_path);
    if (!file.is_open()) {
        return stats;
    }
    
    std::string line;
    std::getline(file, line);
    
    // Parse /proc/[pid]/stat
    // Format: pid (comm) state ppid ... utime stime cutime cstime ...
    std::istringstream iss(line);
    std::string token;
    
    // Skip first 13 fields to get to utime (field 14)
    for (int i = 0; i < 13; ++i) {
        iss >> token;
    }
    
    // Read utime, stime, cutime, cstime (fields 14-17)
    iss >> stats.utime >> stats.stime >> stats.cutime >> stats.cstime;
    
    return stats;
}

CPUMonitor::SystemStats CPUMonitor::read_system_stats() {
    SystemStats stats = {0};
    
    std::ifstream file("/proc/stat");
    if (!file.is_open()) {
        return stats;
    }
    
    std::string line;
    std::getline(file, line);
    
    // First line is total CPU: cpu user nice system idle iowait irq softirq ...
    std::istringstream iss(line);
    std::string cpu;
    long long user, nice, system, idle, iowait, irq, softirq;
    
    iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq;
    stats.total_time = user + nice + system + idle + iowait + irq + softirq;
    
    return stats;
}

double CPUMonitor::read_total_process_cpu_usage() {
    ProcessStats curr_proc = read_process_stats();
    SystemStats curr_sys = read_system_stats();
    
    long long proc_time_diff = (curr_proc.utime + curr_proc.stime + curr_proc.cutime + curr_proc.cstime) -
                                (prev_process_stats_.utime + prev_process_stats_.stime + 
                                 prev_process_stats_.cutime + prev_process_stats_.cstime);
    long long sys_time_diff = curr_sys.total_time - prev_system_stats_.total_time;
    
    prev_process_stats_ = curr_proc;
    prev_system_stats_ = curr_sys;
    
    if (sys_time_diff == 0) return 0.0;
    
    // CPU usage as percentage (multiply by n_cores to normalize to 100% max per core)
    return 100.0 * n_cores_ * proc_time_diff / sys_time_diff;
}

std::vector<double> CPUMonitor::read_process_cpu_usage() {
    std::vector<double> usage(n_cores_);
    
    // Get total process CPU usage
    double total_usage = read_total_process_cpu_usage();
    
    // Distribute evenly across cores
    // (More sophisticated tracking would require monitoring thread affinity)
    double per_core = total_usage / n_cores_;
    for (int i = 0; i < n_cores_; ++i) {
        usage[i] = per_core;
    }
    
    return usage;
}

CPUMetrics CPUMonitor::get_metrics() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    CPUMetrics metrics;
    metrics.n_cores = n_cores_;
    metrics.timeline = timeline_;
    
    if (!timeline_.empty()) {
        metrics.avg_usage = std::accumulate(timeline_.begin(), timeline_.end(), 0.0) / timeline_.size();
        metrics.max_usage = *std::max_element(timeline_.begin(), timeline_.end());
    } else {
        metrics.avg_usage = 0.0;
        metrics.max_usage = 0.0;
    }
    
    // Calculate per-core average
    metrics.per_core_usage.resize(n_cores_);
    for (int i = 0; i < n_cores_; ++i) {
        if (!per_core_timeline_[i].empty()) {
            metrics.per_core_usage[i] = std::accumulate(
                per_core_timeline_[i].begin(), 
                per_core_timeline_[i].end(), 
                0.0
            ) / per_core_timeline_[i].size();
        } else {
            metrics.per_core_usage[i] = 0.0;
        }
    }
    
    return metrics;
}

} // namespace rf_benchmark
