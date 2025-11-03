#include "cpu_monitor.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>

namespace rf_benchmark {

CPUMonitor::CPUMonitor(int n_cores) 
    : n_cores_(n_cores)
    , running_(false) {
    prev_stats_.resize(n_cores_ + 1); // +1 for total CPU
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
    for (int i = 0; i <= n_cores_; ++i) {
        prev_stats_[i] = read_cpu_stats(i - 1);
    }
    
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
        
        double total_usage = read_total_cpu_usage();
        std::vector<double> core_usage = read_cpu_usage();
        
        std::lock_guard<std::mutex> lock(data_mutex_);
        timeline_.push_back(total_usage);
        
        for (size_t i = 0; i < core_usage.size() && i < per_core_timeline_.size(); ++i) {
            per_core_timeline_[i].push_back(core_usage[i]);
        }
    }
}

CPUMonitor::CPUStats CPUMonitor::read_cpu_stats(int core) {
    std::ifstream file("/proc/stat");
    std::string line;
    CPUStats stats = {0, 0, 0, 0, 0, 0, 0};
    
    std::string search_str = (core < 0) ? "cpu " : "cpu" + std::to_string(core) + " ";
    
    while (std::getline(file, line)) {
        if (line.find(search_str) == 0) {
            std::istringstream iss(line);
            std::string cpu;
            iss >> cpu >> stats.user >> stats.nice >> stats.system >> stats.idle 
                >> stats.iowait >> stats.irq >> stats.softirq;
            break;
        }
    }
    
    return stats;
}

double CPUMonitor::read_total_cpu_usage() {
    CPUStats curr = read_cpu_stats(-1);
    CPUStats& prev = prev_stats_[0];
    
    long long prev_idle = prev.idle + prev.iowait;
    long long curr_idle = curr.idle + curr.iowait;
    
    long long prev_total = prev.user + prev.nice + prev.system + prev.idle + 
                           prev.iowait + prev.irq + prev.softirq;
    long long curr_total = curr.user + curr.nice + curr.system + curr.idle + 
                           curr.iowait + curr.irq + curr.softirq;
    
    long long total_diff = curr_total - prev_total;
    long long idle_diff = curr_idle - prev_idle;
    
    prev = curr;
    
    if (total_diff == 0) return 0.0;
    return 100.0 * (total_diff - idle_diff) / total_diff;
}

std::vector<double> CPUMonitor::read_cpu_usage() {
    std::vector<double> usage(n_cores_);
    
    for (int i = 0; i < n_cores_; ++i) {
        CPUStats curr = read_cpu_stats(i);
        CPUStats& prev = prev_stats_[i + 1];
        
        long long prev_idle = prev.idle + prev.iowait;
        long long curr_idle = curr.idle + curr.iowait;
        
        long long prev_total = prev.user + prev.nice + prev.system + prev.idle + 
                               prev.iowait + prev.irq + prev.softirq;
        long long curr_total = curr.user + curr.nice + curr.system + curr.idle + 
                               curr.iowait + curr.irq + curr.softirq;
        
        long long total_diff = curr_total - prev_total;
        long long idle_diff = curr_idle - prev_idle;
        
        prev = curr;
        
        if (total_diff == 0) {
            usage[i] = 0.0;
        } else {
            usage[i] = 100.0 * (total_diff - idle_diff) / total_diff;
        }
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
