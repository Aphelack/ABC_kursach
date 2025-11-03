#include "decision_tree.h"
#include "utils.h"
#include <iostream>

using namespace rf_benchmark;

int main() {
    // Simple data
    Matrix X = {{-5.0}, {-4.0}, {-3.0}, {-2.0}, {-1.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    Labels y = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    
    FeatureIndices features = {0};  // Use all features (just one)
    
    std::cout << "Training single decision tree..." << std::endl;
    DecisionTree tree(5, 2, 1);
    tree.fit(X, y, features);
    
    std::cout << "Making predictions..." << std::endl;
    Vector test1 = {-3.5};
    Vector test2 = {3.5};
    
    int pred1 = tree.predict(test1);
    int pred2 = tree.predict(test2);
    
    std::cout << "Test -3.5: predicted=" << pred1 << " (expected 0)" << std::endl;
    std::cout << "Test +3.5: predicted=" << pred2 << " (expected 1)" << std::endl;
    
    return 0;
}
