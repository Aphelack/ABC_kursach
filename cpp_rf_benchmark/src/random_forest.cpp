#include "random_forest.h"
#include "utils.h"
#include <thread>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

namespace rf_benchmark {

RandomForest::RandomForest(
    int n_estimators,
    int max_depth,
    int n_threads,
    int random_state
)
    : n_estimators_(n_estimators)
    , max_depth_(max_depth)
    , n_threads_(n_threads)
    , random_state_(random_state)
    , max_features_(0) {
    
    if (n_threads_ <= 0) {
        n_threads_ = std::thread::hardware_concurrency();
    }
    
    trees_.resize(n_estimators_);
}

void RandomForest::fit(const Matrix& X, const Labels& y) {
    size_t n_samples = X.size();
    size_t n_features = X[0].size();
    
    // Max features for each tree: sqrt(n_features)
    max_features_ = static_cast<size_t>(std::sqrt(n_features));
    if (max_features_ < 1) max_features_ = 1;
    
    // Fit trees in parallel
    std::vector<std::thread> threads;
    size_t trees_per_thread = (n_estimators_ + n_threads_ - 1) / n_threads_;
    
    for (int t = 0; t < n_threads_; ++t) {
        size_t start_idx = t * trees_per_thread;
        size_t end_idx = std::min(start_idx + trees_per_thread, static_cast<size_t>(n_estimators_));
        
        if (start_idx >= static_cast<size_t>(n_estimators_)) break;
        
        threads.emplace_back([this, &X, &y, start_idx, end_idx, n_samples, n_features, t]() {
            // Seed the thread-local RNG once per thread
            Utils::set_seed(this->random_state_ + t * 1000);
            
            for (size_t i = start_idx; i < end_idx; ++i) {
                this->fit_tree(i, X, y, n_samples, n_features);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

void RandomForest::fit_tree(
    size_t tree_idx,
    const Matrix& X,
    const Labels& y,
    size_t n_samples,
    size_t n_features
) {
    // Note: RNG already seeded in thread lambda
    
    // Bootstrap sampling
    auto bootstrap_indices = bootstrap_sample(n_samples);
    
    Matrix X_bootstrap;
    Labels y_bootstrap;
    X_bootstrap.reserve(bootstrap_indices.size());
    y_bootstrap.reserve(bootstrap_indices.size());
    
    for (size_t idx : bootstrap_indices) {
        X_bootstrap.push_back(X[idx]);
        y_bootstrap.push_back(y[idx]);
    }
    
    // Random feature selection
    auto feature_subset = select_features(n_features);
    
    // Create and fit tree
    trees_[tree_idx] = std::make_unique<DecisionTree>(max_depth_, 2, max_features_);
    trees_[tree_idx]->fit(X_bootstrap, y_bootstrap, feature_subset);
}

Labels RandomForest::predict(const Matrix& X) const {
    Labels predictions;
    predictions.reserve(X.size());
    
    for (const auto& sample : X) {
        predictions.push_back(predict_sample(sample));
    }
    
    return predictions;
}

int RandomForest::predict_sample(const Vector& sample) const {
    std::map<int, int> votes;
    
    for (const auto& tree : trees_) {
        int prediction = tree->predict(sample);
        votes[prediction]++;
    }
    
    // Find class with most votes
    int max_votes = 0;
    int best_class = 0;
    
    for (const auto& pair : votes) {
        if (pair.second > max_votes) {
            max_votes = pair.second;
            best_class = pair.first;
        }
    }
    
    return best_class;
}

double RandomForest::score(const Matrix& X, const Labels& y) const {
    Labels predictions = predict(X);
    return Utils::accuracy(y, predictions);
}

double RandomForest::f1_score(const Matrix& X, const Labels& y, int positive_class) const {
    Labels predictions = predict(X);
    return Utils::f1_score(y, predictions, positive_class);
}

std::vector<size_t> RandomForest::bootstrap_sample(size_t n_samples) {
    std::vector<size_t> indices;
    indices.reserve(n_samples);
    
    for (size_t i = 0; i < n_samples; ++i) {
        indices.push_back(Utils::random_int(0, n_samples - 1));
    }
    
    return indices;
}

FeatureIndices RandomForest::select_features(size_t n_features) {
    FeatureIndices all_features(n_features);
    std::iota(all_features.begin(), all_features.end(), 0);
    
    // Shuffle and take first max_features_
    for (size_t i = 0; i < max_features_; ++i) {
        size_t j = Utils::random_int(i, n_features - 1);
        std::swap(all_features[i], all_features[j]);
    }
    
    FeatureIndices selected(all_features.begin(), all_features.begin() + max_features_);
    return selected;
}

} // namespace rf_benchmark
