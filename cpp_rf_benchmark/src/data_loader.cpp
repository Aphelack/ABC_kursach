#include "data_loader.h"
#include "utils.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <map>

namespace rf_benchmark {

Dataset DataLoader::load_csv(const std::string& filename, bool has_header, char delimiter) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    Dataset data;
    std::string line;
    bool first_line = true;
    
    while (std::getline(file, line)) {
        if (first_line && has_header) {
            first_line = false;
            continue;
        }
        
        auto tokens = split_line(line, delimiter);
        if (tokens.empty()) continue;
        
        std::vector<double> features;
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            features.push_back(std::stod(tokens[i]));
        }
        
        int label = std::stoi(tokens.back());
        
        data.features.push_back(features);
        data.labels.push_back(label);
    }
    
    data.n_samples = data.features.size();
    data.n_features = data.n_samples > 0 ? data.features[0].size() : 0;
    
    // Count classes
    if (!data.labels.empty()) {
        int max_label = *std::max_element(data.labels.begin(), data.labels.end());
        data.n_classes = max_label + 1;
    }
    
    return data;
}

Dataset DataLoader::load_sales_data(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    Dataset data;
    std::string line;
    bool first_line = true;
    
    // Maps for label encoding
    std::map<std::string, int> sales_rep_map;
    std::map<std::string, int> region_map;
    std::map<std::string, int> category_map;
    std::map<std::string, int> customer_type_map;
    std::map<std::string, int> payment_map;
    std::map<std::string, int> channel_map;
    
    int sales_rep_id = 0, region_id = 0, category_id = 0;
    int customer_id = 0, payment_id = 0, channel_id = 0;
    
    // First pass: collect all categorical values and build encodings
    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false;
            continue;
        }
        
        auto tokens = split_line(line, ',');
        if (tokens.size() < 14) continue;
        
        // Sales_Rep (index 2)
        if (sales_rep_map.find(tokens[2]) == sales_rep_map.end()) {
            sales_rep_map[tokens[2]] = sales_rep_id++;
        }
        // Region (index 3)
        if (region_map.find(tokens[3]) == region_map.end()) {
            region_map[tokens[3]] = region_id++;
        }
        // Product_Category (index 6)
        if (category_map.find(tokens[6]) == category_map.end()) {
            category_map[tokens[6]] = category_id++;
        }
        // Customer_Type (index 9) - this is our target!
        if (customer_type_map.find(tokens[9]) == customer_type_map.end()) {
            customer_type_map[tokens[9]] = customer_id++;
        }
        // Payment_Method (index 11)
        if (payment_map.find(tokens[11]) == payment_map.end()) {
            payment_map[tokens[11]] = payment_id++;
        }
        // Sales_Channel (index 12)
        if (channel_map.find(tokens[12]) == channel_map.end()) {
            channel_map[tokens[12]] = channel_id++;
        }
    }
    
    // Reset file to beginning for second pass
    file.clear();
    file.seekg(0);
    first_line = true;
    
    // Second pass: extract features
    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false;
            continue;
        }
        
        auto tokens = split_line(line, ',');
        if (tokens.size() < 14) continue;
        
        std::vector<double> features;
        
        // Numeric features:
        // Product_ID (0), Sales_Amount (4), Quantity_Sold (5),
        // Unit_Cost (7), Unit_Price (8), Discount (10)
        try {
            features.push_back(std::stod(tokens[0]));  // Product_ID
            features.push_back(std::stod(tokens[4]));  // Sales_Amount
            features.push_back(std::stod(tokens[5]));  // Quantity_Sold
            features.push_back(std::stod(tokens[7]));  // Unit_Cost
            features.push_back(std::stod(tokens[8]));  // Unit_Price
            features.push_back(std::stod(tokens[10])); // Discount
            
            // Categorical features (encoded):
            features.push_back(sales_rep_map[tokens[2]]);   // Sales_Rep
            features.push_back(region_map[tokens[3]]);      // Region
            features.push_back(category_map[tokens[6]]);    // Product_Category
            features.push_back(payment_map[tokens[11]]);    // Payment_Method
            features.push_back(channel_map[tokens[12]]);    // Sales_Channel
            
            // Label: Customer_Type (New=0, Returning=1)
            int label = customer_type_map[tokens[9]];
            
            data.features.push_back(features);
            data.labels.push_back(label);
        } catch (const std::exception& e) {
            // Skip malformed rows
            continue;
        }
    }
    
    data.n_samples = data.features.size();
    data.n_features = data.n_samples > 0 ? data.features[0].size() : 0;
    data.n_classes = customer_type_map.size();
    
    return data;
}

Dataset DataLoader::load_creditcard_data(const std::string& filename, double sample_fraction) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    Dataset data;
    std::string line;
    
    // Skip header line(s) - the header might be multi-line
    std::getline(file, line);
    if (line.find("V20") == std::string::npos) {
        // Header continues on next line
        std::getline(file, line);
    }
    
    // Read all data first, then sample
    std::vector<std::vector<double>> all_features;
    std::vector<int> all_labels;
    
    while (std::getline(file, line)) {
        // Remove quotes and parse
        std::string cleaned;
        for (char c : line) {
            if (c != '"') cleaned += c;
        }
        
        auto tokens = split_line(cleaned, ',');
        if (tokens.empty() || tokens.size() < 31) continue;
        
        try {
            std::vector<double> features;
            
            // Time (index 0)
            features.push_back(std::stod(tokens[0]));
            
            // V1-V28 (indices 1-28) - these are PCA components
            for (int i = 1; i <= 28; ++i) {
                features.push_back(std::stod(tokens[i]));
            }
            
            // Amount (index 29)
            features.push_back(std::stod(tokens[29]));
            
            // Class (index 30) - 0=normal, 1=fraud
            int label = std::stoi(tokens[30]);
            
            all_features.push_back(features);
            all_labels.push_back(label);
        } catch (const std::exception& e) {
            // Skip malformed rows
            continue;
        }
    }
    
    // Stratified sampling to maintain class balance
    size_t total_samples = all_features.size();
    size_t sample_size = static_cast<size_t>(total_samples * sample_fraction);
    
    if (sample_size < 100) {
        sample_size = std::min(static_cast<size_t>(100), total_samples);
    }
    
    // Separate indices by class
    std::vector<size_t> fraud_indices;
    std::vector<size_t> normal_indices;
    
    for (size_t i = 0; i < total_samples; ++i) {
        if (all_labels[i] == 1) {
            fraud_indices.push_back(i);
        } else {
            normal_indices.push_back(i);
        }
    }
    
    // Calculate samples per class to maintain ratio
    size_t fraud_count = static_cast<size_t>(fraud_indices.size() * sample_fraction);
    size_t normal_count = static_cast<size_t>(normal_indices.size() * sample_fraction);
    
    // Ensure we have at least some fraud cases
    if (fraud_count < 10 && fraud_indices.size() >= 10) {
        fraud_count = std::min(size_t(10), fraud_indices.size());
    }
    
    // Shuffle each class separately
    shuffle_indices(fraud_indices, 42);
    shuffle_indices(normal_indices, 43);
    
    // Take samples from each class
    data.features.reserve(fraud_count + normal_count);
    data.labels.reserve(fraud_count + normal_count);
    
    for (size_t i = 0; i < fraud_count; ++i) {
        data.features.push_back(all_features[fraud_indices[i]]);
        data.labels.push_back(all_labels[fraud_indices[i]]);
    }
    
    for (size_t i = 0; i < normal_count; ++i) {
        data.features.push_back(all_features[normal_indices[i]]);
        data.labels.push_back(all_labels[normal_indices[i]]);
    }
    
    // Shuffle the combined data
    std::vector<size_t> final_indices(data.features.size());
    std::iota(final_indices.begin(), final_indices.end(), 0);
    shuffle_indices(final_indices, 44);
    
    std::vector<std::vector<double>> shuffled_features;
    std::vector<int> shuffled_labels;
    shuffled_features.reserve(data.features.size());
    shuffled_labels.reserve(data.labels.size());
    
    for (size_t idx : final_indices) {
        shuffled_features.push_back(data.features[idx]);
        shuffled_labels.push_back(data.labels[idx]);
    }
    
    data.features = std::move(shuffled_features);
    data.labels = std::move(shuffled_labels);
    
    data.n_samples = data.features.size();
    data.n_features = data.n_samples > 0 ? data.features[0].size() : 0;
    
    // Count classes
    if (!data.labels.empty()) {
        int max_label = *std::max_element(data.labels.begin(), data.labels.end());
        data.n_classes = max_label + 1;
    }
    
    return data;
}

Dataset DataLoader::generate_synthetic_data(
    size_t n_samples,
    size_t n_features,
    size_t n_classes,
    int random_state
) {
    Utils::set_seed(random_state);
    
    Dataset data;
    data.n_samples = n_samples;
    data.n_features = n_features;
    data.n_classes = n_classes;
    
    data.features.resize(n_samples);
    data.labels.resize(n_samples);
    
    // Generate features and labels with actual relationships
    for (size_t i = 0; i < n_samples; ++i) {
        data.features[i].resize(n_features);
        
        // Generate features
        for (size_t j = 0; j < n_features; ++j) {
            data.features[i][j] = Utils::random_double(-10.0, 10.0);
        }
        
        // Create label based on features (not random!)
        // Use a simple decision boundary: sum of first few features
        double decision_value = 0.0;
        size_t features_to_use = std::min(n_features, static_cast<size_t>(5));
        for (size_t j = 0; j < features_to_use; ++j) {
            decision_value += data.features[i][j];
        }
        
        // Add some noise
        decision_value += Utils::random_double(-2.0, 2.0);
        
        // For binary classification
        if (n_classes == 2) {
            data.labels[i] = (decision_value > 0) ? 1 : 0;
        } else {
            // For multi-class: divide the decision space
            double range = 20.0 / n_classes; // -10 to 10 divided by n_classes
            int label = static_cast<int>((decision_value + 10.0) / range);
            label = std::max(0, std::min(static_cast<int>(n_classes - 1), label));
            data.labels[i] = label;
        }
    }
    
    return data;
}

TrainTestSplit DataLoader::train_test_split(
    const Dataset& data,
    double test_size,
    int random_state
) {
    Utils::set_seed(random_state);
    
    std::vector<size_t> indices(data.n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    shuffle_indices(indices, random_state);
    
    size_t test_count = static_cast<size_t>(data.n_samples * test_size);
    size_t train_count = data.n_samples - test_count;
    
    TrainTestSplit split;
    
    // Train data
    split.train.n_features = data.n_features;
    split.train.n_classes = data.n_classes;
    split.train.n_samples = train_count;
    split.train.features.reserve(train_count);
    split.train.labels.reserve(train_count);
    
    for (size_t i = 0; i < train_count; ++i) {
        split.train.features.push_back(data.features[indices[i]]);
        split.train.labels.push_back(data.labels[indices[i]]);
    }
    
    // Test data
    split.test.n_features = data.n_features;
    split.test.n_classes = data.n_classes;
    split.test.n_samples = test_count;
    split.test.features.reserve(test_count);
    split.test.labels.reserve(test_count);
    
    for (size_t i = train_count; i < data.n_samples; ++i) {
        split.test.features.push_back(data.features[indices[i]]);
        split.test.labels.push_back(data.labels[indices[i]]);
    }
    
    return split;
}

void DataLoader::normalize(Dataset& data) {
    if (data.n_samples == 0 || data.n_features == 0) return;
    
    // Calculate mean and std for each feature
    std::vector<double> means(data.n_features, 0.0);
    std::vector<double> stds(data.n_features, 0.0);
    
    // Calculate means
    for (const auto& sample : data.features) {
        for (size_t j = 0; j < data.n_features; ++j) {
            means[j] += sample[j];
        }
    }
    for (double& m : means) {
        m /= data.n_samples;
    }
    
    // Calculate standard deviations
    for (const auto& sample : data.features) {
        for (size_t j = 0; j < data.n_features; ++j) {
            double diff = sample[j] - means[j];
            stds[j] += diff * diff;
        }
    }
    for (double& s : stds) {
        s = std::sqrt(s / data.n_samples);
        if (s < 1e-10) s = 1.0; // Avoid division by zero
    }
    
    // Normalize
    for (auto& sample : data.features) {
        for (size_t j = 0; j < data.n_features; ++j) {
            sample[j] = (sample[j] - means[j]) / stds[j];
        }
    }
}

std::vector<std::string> DataLoader::split_line(const std::string& line, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        // Trim whitespace
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);
        tokens.push_back(token);
    }
    
    return tokens;
}

void DataLoader::shuffle_indices(std::vector<size_t>& indices, int random_state) {
    Utils::set_seed(random_state);
    for (size_t i = indices.size() - 1; i > 0; --i) {
        size_t j = Utils::random_int(0, i);
        std::swap(indices[i], indices[j]);
    }
}

} // namespace rf_benchmark
