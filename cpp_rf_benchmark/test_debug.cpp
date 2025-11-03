#include "random_forest.h"
#include "decision_tree.h"
#include "utils.h"
#include <iostream>

using namespace rf_benchmark;

int main() {
    Matrix X = {{-5.0}, {-4.0}, {-3.0}, {-2.0}, {-1.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    Labels y = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    
    std::cout << "Training 3 trees with 1 thread..." << std::endl;
    RandomForest rf(3, 5, 1, 42);
    rf.fit(X, y);
    
    Vector test_neg = {-3.5};
    Vector test_pos = {3.5};
    
    int pred_neg = rf.predict({test_neg})[0];
    int pred_pos = rf.predict({test_pos})[0];
    
    std::cout << "Negative sample prediction: " << pred_neg << std::endl;
    std::cout << "Positive sample prediction: " << pred_pos << std::endl;
    
    // Test individual trees
    std::cout << "\nTesting individual trees:" << std::endl;
    FeatureIndices all_feat = {0};
    
    for (int i = 0; i < 3; ++i) {
        Utils::set_seed(42 + i * 1000);
        
        // Bootstrap
        std::vector<size_t> bootstrap;
        for (size_t j = 0; j < 10; ++j) {
            bootstrap.push_back(Utils::random_int(0, 9));
        }
        
        Matrix X_boot;
        Labels y_boot;
        for (auto idx : bootstrap) {
            X_boot.push_back(X[idx]);
            y_boot.push_back(y[idx]);
        }
        
        std::cout << "Tree " << i << " bootstrap sample labels: ";
        for (auto label : y_boot) std::cout << label << " ";
        std::cout << std::endl;
        
        DecisionTree tree(5, 2, 1);
        tree.fit(X_boot, y_boot, all_feat);
        
        int p_neg = tree.predict(test_neg);
        int p_pos = tree.predict(test_pos);
        std::cout << "  Predictions: neg=" << p_neg << ", pos=" << p_pos << std::endl;
    }
    
    return 0;
}
