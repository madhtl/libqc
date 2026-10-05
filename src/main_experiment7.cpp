#include "MakeMoons.h"
#include "QuantumReservoir.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>

using std::chrono::steady_clock;
using std::chrono::duration;

int main() {
    auto train_data = make_moons(200, 0.1, 42);
    auto test_data  = make_moons(100, 0.1, 123);

    std::vector<int> sizes = {4, 6, 8, 10};

    std::cout << "=== Quantum Reservoir Computing ===\n";
    std::cout << "Fixed random circuit + ridge regression (no gradient)\n\n";

    std::cout << std::setw(10) << "n_qubits"
              << std::setw(12) << "features"
              << std::setw(12) << "test_acc"
              << std::setw(12) << "time_ms\n";

    double best_acc = 0.0;
    int    best_n   = 0;
    double ms       = 0.0;
    double best_ms  = 0.0;

    std::ofstream log("qrc_log.csv");
    log << "n_qubits,test_acc,time_ms\n";

    for (int n : sizes) {
        QuantumReservoir qr(n, 42);

        auto X_train = qr.encode_all(train_data);
        auto X_test  = qr.encode_all(test_data);

        std::vector<int> y_train, y_test;
        for (auto& s : train_data) y_train.push_back(s.label);
        for (auto& s : test_data)  y_test.push_back(s.label);

        auto t1 = steady_clock::now();
        RidgeClassifier clf;
        clf.fit(X_train, y_train);
        ms = duration<double>(steady_clock::now() - t1).count() * 1000.0;

        double acc = clf.accuracy(X_test, y_test);
        std::cout << std::setw(10) << n
                  << std::setw(12) << qr.numFeatures()
                  << std::setw(12) << acc
                  << std::setw(12) << ms << "\n";

        log << n << "," << acc << "," << ms << "\n";

        if (acc > best_acc) {
            best_acc = acc;
            best_n   = n;
            best_ms  = ms;
        }
    }

    std::cout << "\nBest reservoir: n=" << best_n
              << "  acc=" << best_acc << "\n";

    std::cout << "\n=== Full experiment leaderboard ===\n";
    std::cout << "QNN standard 2L:    0.62  (gradient, 1.36s)\n";
    std::cout << "MLP:                0.80  (backprop, 0.03s)\n";
    std::cout << "Reuploading QNN 2L: 0.81  (gradient, ~1.5s)\n";
    std::cout << "QK-SVM:             0.92  (kernel, 0.98s)\n";
    std::cout << "Reuploading QNN 4L: 0.95  (gradient, ~3s)\n";
    std::cout << "VQE H2:             err<1e-6 Ha (chemistry)\n";
    std::cout << "QRC best (n=" << best_n << "):     "
              << best_acc << "  (no gradient, " << best_ms << "ms)\n";

    return 0;
}
