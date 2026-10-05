#include "MakeMoons.h"
#include "QuantumKernel.h"
#include "SVM.h"
#include <iostream>
#include <fstream>
#include <chrono>

int main() {
    auto train_data = make_moons(200, 0.1, 42);
    auto test_data  = make_moons(100, 0.1, 123);

    const int n_train = static_cast<int>(train_data.size());
    const int n_test  = static_cast<int>(test_data.size());

    // ── 1. macierz kernela ────────────────────────────────────────────────────

    std::cout << "Building quantum kernel matrix"
              << " (4 qubits, " << n_train << " train + " << n_test << " test)...\n";

    QuantumKernel qk(4);
    auto t0 = std::chrono::steady_clock::now();

    auto K_train = qk.kernel_matrix(train_data);                // 200×200, symetryczna
    auto K_test  = qk.kernel_matrix(test_data, train_data);     // 100×200

    double qk_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    std::cout << "  done (" << qk_time << "s)\n";

    // ── 2. etykiety ───────────────────────────────────────────────────────────

    std::vector<int> train_labels, test_labels;
    train_labels.reserve(n_train);
    test_labels.reserve(n_test);
    for (const auto& s : train_data) train_labels.push_back(s.label ? 1 : -1);
    for (const auto& s : test_data)  test_labels.push_back(s.label ? 1 : -1);

    // ── 3. trening SVM ────────────────────────────────────────────────────────

    std::cout << "Training KernelSVM (C=1.0, lr=0.01, 1000 iterations)...\n";
    KernelSVM svm(1.0, 0.01);
    svm.fit(K_train, train_labels, 1000);

    // ── 4. predykcja i ocena ─────────────────────────────────────────────────

    auto pred = svm.predict(K_test);
    double acc = svm.accuracy(pred, test_labels);

    std::cout << "\n=== Model comparison ===\n";
    std::cout << "QNN           test acc: 0.620  params:  4  time: 1.36s\n";
    std::cout << "MLP           test acc: 0.800  params: 21  time: 0.03s\n";
    std::cout << "QK-SVM        test acc: " << acc
              << "  SV: " << svm.num_support_vectors()
              << "  kernel time: " << qk_time << "s\n";

    // ── 5. kernel_log.csv ─────────────────────────────────────────────────────

    std::ofstream log("kernel_log.csv");
    log << "model,test_acc,time,params\n"
        << "QNN,0.620,1.36,4\n"
        << "MLP,0.800,0.03,21\n"
        << "QK-SVM," << acc << "," << qk_time << "," << svm.num_support_vectors() << "\n";

    std::cout << "Saved: kernel_log.csv\n";
    return 0;
}
