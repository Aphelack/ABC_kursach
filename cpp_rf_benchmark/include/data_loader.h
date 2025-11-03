#pragma once

#include "types.h"
#include <string>

namespace rf_benchmark {

class DataLoader {
public:
    // Load CSV file
    static Dataset load_csv(
        const std::string& filename,
        bool has_header = true,
        char delimiter = ','
    );
    
    // Load and preprocess sales data
    static Dataset load_sales_data(const std::string& filename);
    
    // Load credit card fraud detection data
    static Dataset load_creditcard_data(
        const std::string& filename,
        double sample_fraction = 0.01  // Use 1% by default
    );
    
    // Generate synthetic data for testing
    static Dataset generate_synthetic_data(
        size_t n_samples,
        size_t n_features,
        size_t n_classes,
        int random_state = 42
    );
    
    // Train-test split
    static TrainTestSplit train_test_split(
        const Dataset& data,
        double test_size = 0.2,
        int random_state = 42
    );
    
    // Normalize features
    static void normalize(Dataset& data);
    
private:
    static std::vector<std::string> split_line(const std::string& line, char delimiter);
    static void shuffle_indices(std::vector<size_t>& indices, int random_state);
};

} // namespace rf_benchmark
