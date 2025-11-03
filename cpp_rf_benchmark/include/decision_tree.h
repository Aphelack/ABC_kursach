#pragma once

#include "types.h"
#include <memory>

namespace rf_benchmark {

struct TreeNode {
    bool is_leaf;
    int predicted_class;
    
    // For internal nodes
    size_t feature_index;
    double threshold;
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;
    
    TreeNode() : is_leaf(false), predicted_class(-1), feature_index(0), threshold(0.0) {}
};

class DecisionTree {
public:
    DecisionTree(int max_depth, int min_samples_split, size_t max_features);
    
    void fit(const Matrix& X, const Labels& y, const FeatureIndices& feature_subset);
    int predict(const Vector& sample) const;
    
private:
    std::unique_ptr<TreeNode> build_tree(
        const Matrix& X, 
        const Labels& y,
        const FeatureIndices& feature_subset,
        int depth
    );
    
    int predict_node(const Vector& sample, const TreeNode* node) const;
    
    struct SplitResult {
        size_t feature_index;
        double threshold;
        double gini_gain;
        std::vector<size_t> left_indices;
        std::vector<size_t> right_indices;
    };
    
    SplitResult find_best_split(
        const Matrix& X,
        const Labels& y,
        const std::vector<size_t>& indices,
        const FeatureIndices& feature_subset
    );
    
    double calculate_gini(const Labels& y, const std::vector<size_t>& indices);
    int majority_class(const Labels& y, const std::vector<size_t>& indices);
    
    int max_depth_;
    int min_samples_split_;
    size_t max_features_;
    std::unique_ptr<TreeNode> root_;
};

} // namespace rf_benchmark
