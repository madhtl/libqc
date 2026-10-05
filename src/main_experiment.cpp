#include "MakeMoons.h"
#include "QNN.h"
#include "MLP.h"
#include <iostream>
#include <chrono>

int main() {
    auto train_data = make_moons(200, 0.1, 42);
    auto test_data  = make_moons(100, 0.1, 123);

    std::cout << "=== Training QNN (2 layers, 4 params, 100 epochs) ===\n";
    auto t0 = std::chrono::steady_clock::now();
    QNN qnn(2);
    qnn.train(train_data, 100, 0.1, test_data, "qnn_log.csv");
    double qnn_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::cout << "\n=== Training MLP (2→4→1, 100 epochs) ===\n";
    auto t2 = std::chrono::steady_clock::now();
    MLP mlp;
    mlp.train(train_data, 100, 0.1, test_data, "mlp_log.csv");
    double mlp_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();

    std::cout << "\n=== Final Results ===\n";
    std::cout << "QNN  test accuracy: " << qnn.accuracy(test_data)
              << "  (train time: " << qnn_time << "s)\n";
    std::cout << "MLP  test accuracy: " << mlp.accuracy(test_data)
              << "  (train time: " << mlp_time << "s)\n";

    return 0;
}
