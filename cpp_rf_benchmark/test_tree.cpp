#include "random_forest.h"
#include "utils.h"
#include <iostream>

using namespace rf_benchmark;

int main() {
    // Create perfectly separable data: if x[0] > 0, class 1, else class 0
    Matrix X_train = {
        {-5.0, 1.0}, {-4.0, 2.0}, {-3.0, 1.5}, {-2.0, 3.0}, {-1.0, 2.5},
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 1.5}, {4.0, 3.0}, {5.0, 2.5}
    };
    Labels y_train = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    
    Matrix X_test = {
        {-4.5, 1.5}, {-1.5, 2.2}, {1.5, 1.8}, {4.5, 2.8}
    };
    Labels y_test = {0, 0, 1, 1};
    
    std::cout << "Training Random Forest on simple data..." << std::endl;
    RandomForest rf(10, 5, 1, 42);
    rf.fit(X_train, y_train);
    
    std::cout << "Predicting..." << std::endl;
    Labels predictions = rf.predict(X_test);
    
    std::cout << "\nResults:" << std::endl;
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "Sample " << i << ": predicted=" << predictions[i] 
                  << ", actual=" << y_test[i] 
                  << (predictions[i] == y_test[i] ? " ✓" : " ✗") << std::endl;
    }
    
    double acc = rf.score(X_test, y_test);
    std::cout << "\nAccuracy: " << (acc * 100) << "%" << std::endl;
    
    return 0;
}
