#pragma once

#include "types.h"
#include "decision_tree.h"
#include <vector>
#include <memory>

namespace rf_benchmark {

class RandomForest {
public:
    RandomForest(
        int n_estimators,
        int max_depth,
        int n_threads = -1,
        int random_state = 42
    );
    
    void fit(const Matrix& X, const Labels& y);
    Labels predict(const Matrix& X) const;
    double score(const Matrix& X, const Labels& y) const;
    double f1_score(const Matrix& X, const Labels& y, int positive_class = 1) const;
    
    int get_n_estimators() const { return n_estimators_; }
    int get_n_threads() const { return n_threads_; }
    
private:
    void fit_tree(
        size_t tree_idx,
        const Matrix& X,
        const Labels& y,
        size_t n_samples,
        size_t n_features
    );
    
    int predict_sample(const Vector& sample) const;
    
    std::vector<size_t> bootstrap_sample(size_t n_samples);
    FeatureIndices select_features(size_t n_features);
    
    int n_estimators_;
    int max_depth_;
    int n_threads_;
    int random_state_;
    size_t max_features_;
    
    std::vector<std::unique_ptr<DecisionTree>> trees_;
};

} // namespace rf_benchmark
