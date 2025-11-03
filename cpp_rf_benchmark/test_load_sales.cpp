#include "data_loader.h"
#include <iostream>
#include <map>

using namespace rf_benchmark;

int main() {
    auto data = DataLoader::load_sales_data("../dattaset/sales_data.csv");
    
    std::cout << "Samples: " << data.n_samples << std::endl;
    std::cout << "Features: " << data.n_features << std::endl;
    std::cout << "Classes: " << data.n_classes << std::endl;
    
    // Count labels
    std::map<int, int> label_counts;
    for (auto label : data.labels) {
        label_counts[label]++;
    }
    
    std::cout << "\nLabel distribution:" << std::endl;
    for (auto& p : label_counts) {
        std::cout << "  Class " << p.first << ": " << p.second << " samples" << std::endl;
    }
    
    std::cout << "\nFirst 5 samples:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), data.n_samples); ++i) {
        std::cout << "Sample " << i << ": features=[";
        for (size_t j = 0; j < data.n_features; ++j) {
            std::cout << data.features[i][j];
            if (j < data.n_features - 1) std::cout << ", ";
        }
        std::cout << "], label=" << data.labels[i] << std::endl;
    }
    
    return 0;
}
