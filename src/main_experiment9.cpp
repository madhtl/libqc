#include "NoiseModel.h"
#include "MakeMoons.h"
#include "SVM.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

int main() {
    auto train_data = make_moons(200, 0.1, 42);
    auto test_data  = make_moons(100, 0.1, 123);

    const std::vector<double> noise_levels = {0.0, 0.01, 0.02, 0.05, 0.10};

    std::cout << "=== Noise Model Experiment ===\n";
    std::cout << "Depolarizing channel: p = error probability per gate\n\n";

    std::cout << std::setw(8)  << "p"
              << std::setw(14) << "QNN acc"
              << std::setw(14) << "QK-SVM acc"
              << "\n";

    std::ofstream log("noise_log.csv");
    log << "noise_p,qnn_acc,kernel_acc\n";

    for (double p : noise_levels) {
        // Noisy QNN
        NoisyQNN noisy_qnn(2, p, 42);
        noisy_qnn.train(train_data, 100, 0.1);
        double qnn_acc = noisy_qnn.accuracy(test_data);

        // Noisy Kernel + SVM
        NoisyQuantumKernel noisy_kernel(4, p, 42);
        auto K_train = noisy_kernel.kernel_matrix(train_data);
        auto K_test  = noisy_kernel.kernel_matrix_test(test_data, train_data);

        std::vector<int> y_train, y_test;
        for (const auto& s : train_data) y_train.push_back(s.label ? 1 : -1);
        for (const auto& s : test_data)  y_test.push_back(s.label ? 1 : -1);

        KernelSVM svm(1.0);
        svm.fit(K_train, y_train);
        auto pred = svm.predict(K_test);
        double kernel_acc = svm.accuracy(pred, y_test);

        std::cout << std::setw(8)  << p
                  << std::setw(14) << qnn_acc
                  << std::setw(14) << kernel_acc
                  << "\n";

        log << p << "," << qnn_acc << "," << kernel_acc << "\n";
    }

    std::cout << "\nKey question:\n";
    std::cout << "Which method degrades faster under noise — QNN or kernel?\n";
    std::cout << "At what noise level does accuracy drop below MLP baseline (0.80)?\n";

    return 0;
}
