#include "decision_tree.h"
#include "utils.h"
#include <algorithm>
#include <limits>
#include <map>
#include <numeric>

namespace rf_benchmark {

DecisionTree::DecisionTree(int max_depth, int min_samples_split, size_t max_features)
    : max_depth_(max_depth)
    , min_samples_split_(min_samples_split)
    , max_features_(max_features) {
}

void DecisionTree::fit(const Matrix& X, const Labels& y, const FeatureIndices& feature_subset) {
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    root_ = build_tree(X, y, feature_subset, 0);
}

int DecisionTree::predict(const Vector& sample) const {
    return predict_node(sample, root_.get());
}

std::unique_ptr<TreeNode> DecisionTree::build_tree(
    const Matrix& X,
    const Labels& y,
    const FeatureIndices& feature_subset,
    int depth
) {
    auto node = std::make_unique<TreeNode>();
    
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Check stopping criteria
    if (depth >= max_depth_ || X.size() < static_cast<size_t>(min_samples_split_)) {
        node->is_leaf = true;
        node->predicted_class = majority_class(y, indices);
        return node;
    }
    
    // Check if all samples have same label
    bool all_same = true;
    for (size_t i = 1; i < y.size(); ++i) {
        if (y[i] != y[0]) {
            all_same = false;
            break;
        }
    }
    
    if (all_same) {
        node->is_leaf = true;
        node->predicted_class = y[0];
        return node;
    }
    
    // Find best split
    auto split = find_best_split(X, y, indices, feature_subset);
    
    if (split.gini_gain <= 0.0 || split.left_indices.empty() || split.right_indices.empty()) {
        node->is_leaf = true;
        node->predicted_class = majority_class(y, indices);
        return node;
    }
    
    // Create internal node
    node->feature_index = split.feature_index;
    node->threshold = split.threshold;
    
    // Create left subtree
    Matrix X_left;
    Labels y_left;
    for (size_t idx : split.left_indices) {
        X_left.push_back(X[idx]);
        y_left.push_back(y[idx]);
    }
    node->left = build_tree(X_left, y_left, feature_subset, depth + 1);
    
    // Create right subtree
    Matrix X_right;
    Labels y_right;
    for (size_t idx : split.right_indices) {
        X_right.push_back(X[idx]);
        y_right.push_back(y[idx]);
    }
    node->right = build_tree(X_right, y_right, feature_subset, depth + 1);
    
    return node;
}

DecisionTree::SplitResult DecisionTree::find_best_split(
    const Matrix& X,
    const Labels& y,
    const std::vector<size_t>& indices,
    const FeatureIndices& feature_subset
) {
    SplitResult best_split;
    best_split.gini_gain = -std::numeric_limits<double>::infinity();
    
    double parent_gini = calculate_gini(y, indices);
    
    for (size_t feature_idx : feature_subset) {
        // Collect unique values for this feature
        std::vector<double> values;
        for (size_t idx : indices) {
            values.push_back(X[idx][feature_idx]);
        }
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        
        // Limit the number of thresholds to check for efficiency
        // Sample at most 50 thresholds per feature (sklearn uses similar optimization)
        size_t max_thresholds = 50;
        size_t step = 1;
        if (values.size() > max_thresholds + 1) {
            step = (values.size() - 1) / max_thresholds;
            if (step == 0) step = 1;
        }
        
        // Try each threshold
        for (size_t i = 0; i < values.size() - 1; i += step) {
            double threshold = (values[i] + values[i + 1]) / 2.0;
            
            std::vector<size_t> left_indices, right_indices;
            for (size_t idx : indices) {
                if (X[idx][feature_idx] <= threshold) {
                    left_indices.push_back(idx);
                } else {
                    right_indices.push_back(idx);
                }
            }
            
            if (left_indices.empty() || right_indices.empty()) continue;
            
            double left_gini = calculate_gini(y, left_indices);
            double right_gini = calculate_gini(y, right_indices);
            
            double n = indices.size();
            double n_left = left_indices.size();
            double n_right = right_indices.size();
            
            double weighted_gini = (n_left / n) * left_gini + (n_right / n) * right_gini;
            double gini_gain = parent_gini - weighted_gini;
            
            if (gini_gain > best_split.gini_gain) {
                best_split.feature_index = feature_idx;
                best_split.threshold = threshold;
                best_split.gini_gain = gini_gain;
                best_split.left_indices = left_indices;
                best_split.right_indices = right_indices;
            }
        }
    }
    
    return best_split;
}

double DecisionTree::calculate_gini(const Labels& y, const std::vector<size_t>& indices) {
    if (indices.empty()) return 0.0;
    
    std::map<int, int> counts;
    for (size_t idx : indices) {
        counts[y[idx]]++;
    }
    
    double gini = 1.0;
    double n = indices.size();
    
    for (const auto& pair : counts) {
        double p = pair.second / n;
        gini -= p * p;
    }
    
    return gini;
}

int DecisionTree::majority_class(const Labels& y, const std::vector<size_t>& indices) {
    std::map<int, int> counts;
    for (size_t idx : indices) {
        counts[y[idx]]++;
    }
    
    int max_count = 0;
    int majority = 0;
    
    for (const auto& pair : counts) {
        if (pair.second > max_count) {
            max_count = pair.second;
            majority = pair.first;
        }
    }
    
    return majority;
}

int DecisionTree::predict_node(const Vector& sample, const TreeNode* node) const {
    if (node->is_leaf) {
        return node->predicted_class;
    }
    
    if (sample[node->feature_index] <= node->threshold) {
        return predict_node(sample, node->left.get());
    } else {
        return predict_node(sample, node->right.get());
    }
}

} // namespace rf_benchmark
