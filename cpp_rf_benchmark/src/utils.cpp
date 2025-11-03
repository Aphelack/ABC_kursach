#include "utils.h"
#include <fstream>
#include <sstream>
#include <ctime>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <thread>
#include <sys/utsname.h>
#include <sys/sysinfo.h>

namespace rf_benchmark {

// Thread-local random number generator
thread_local std::mt19937 rng;

SystemInfo Utils::get_system_info() {
    SystemInfo info;
    info.cpu_model = get_cpu_model();
    info.architecture = "x86_64";
    info.logical_cores = get_logical_cores();
    info.physical_cores = get_physical_cores();
    info.total_memory = get_total_memory();
    
    struct utsname uts;
    if (uname(&uts) == 0) {
        info.os_name = uts.sysname;
        info.os_version = std::string(uts.release) + " " + uts.version;
    }
    
    return info;
}

std::string Utils::get_cpu_model() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                std::string model = line.substr(pos + 1);
                // Trim leading whitespace
                model.erase(0, model.find_first_not_of(" \t"));
                return model;
            }
        }
    }
    return "Unknown CPU";
}

std::string Utils::get_sanitized_cpu_name() {
    std::string cpu = get_cpu_model();
    std::string sanitized;
    
    for (char c : cpu) {
        if (std::isalnum(c) || c == '_' || c == '-') {
            sanitized += c;
        } else if (c == ' ' || c == '@' || c == '(' || c == ')') {
            if (!sanitized.empty() && sanitized.back() != '_') {
                sanitized += '_';
            }
        }
    }
    
    // Remove trailing underscores
    while (!sanitized.empty() && sanitized.back() == '_') {
        sanitized.pop_back();
    }
    
    return sanitized;
}

int Utils::get_logical_cores() {
    return std::thread::hardware_concurrency();
}

int Utils::get_physical_cores() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    int max_core_id = -1;
    
    while (std::getline(cpuinfo, line)) {
        if (line.find("core id") != std::string::npos) {
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                int core_id = std::stoi(line.substr(pos + 1));
                max_core_id = std::max(max_core_id, core_id);
            }
        }
    }
    
    return max_core_id + 1;
}

long Utils::get_total_memory() {
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        return si.totalram * si.mem_unit;
    }
    return 0;
}

std::string Utils::get_timestamp() {
    auto now = std::time(nullptr);
    char buf[100];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&now));
    return std::string(buf);
}

double Utils::mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double Utils::std_dev(const std::vector<double>& values) {
    if (values.size() < 2) return 0.0;
    
    double m = mean(values);
    double sum = 0.0;
    for (double v : values) {
        sum += (v - m) * (v - m);
    }
    return std::sqrt(sum / (values.size() - 1));
}

double Utils::accuracy(const Labels& y_true, const Labels& y_pred) {
    if (y_true.size() != y_pred.size()) return 0.0;
    
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) ++correct;
    }
    
    return static_cast<double>(correct) / y_true.size();
}

void Utils::set_seed(unsigned int seed) {
    rng.seed(seed);
}

int Utils::random_int(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

double Utils::random_double(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

double Utils::precision(const Labels& y_true, const Labels& y_pred, int positive_class) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Label vectors must have the same size");
    }
    
    int true_positives = 0;
    int false_positives = 0;
    
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_pred[i] == positive_class) {
            if (y_true[i] == positive_class) {
                true_positives++;
            } else {
                false_positives++;
            }
        }
    }
    
    int predicted_positives = true_positives + false_positives;
    if (predicted_positives == 0) {
        return 0.0;
    }
    
    return static_cast<double>(true_positives) / predicted_positives;
}

double Utils::recall(const Labels& y_true, const Labels& y_pred, int positive_class) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Label vectors must have the same size");
    }
    
    int true_positives = 0;
    int actual_positives = 0;
    
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == positive_class) {
            actual_positives++;
            if (y_pred[i] == positive_class) {
                true_positives++;
            }
        }
    }
    
    if (actual_positives == 0) {
        return 0.0;
    }
    
    return static_cast<double>(true_positives) / actual_positives;
}

double Utils::f1_score(const Labels& y_true, const Labels& y_pred, int positive_class) {
    double prec = precision(y_true, y_pred, positive_class);
    double rec = recall(y_true, y_pred, positive_class);
    
    if (prec + rec == 0.0) {
        return 0.0;
    }
    
    return 2.0 * (prec * rec) / (prec + rec);
}

bool Utils::file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

std::string Utils::get_report_filename(const std::string& cpu_name) {
    return "rf_benchmark_" + cpu_name + "_" + get_timestamp() + ".json";
}

} // namespace rf_benchmark
