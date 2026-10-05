#include "MakeMoons.h"
#include "QFTKernel.h"
#include "SVM.h"
#include <iostream>
#include <fstream>
#include <chrono>

using std::chrono::steady_clock;
using std::chrono::duration;

int main() {
    auto train_data = make_moons(200, 0.1, 42);
    auto test_data  = make_moons(100, 0.1, 123);

    std::cout << "=== QFT Kernel SVM ===\n";

    QFTKernel qft(4);

    auto t1 = steady_clock::now();
    auto K_train = qft.kernel_matrix(train_data);
    auto K_test  = qft.kernel_matrix_test(test_data, train_data);
    double kernel_time = duration<double>(steady_clock::now() - t1).count();

    std::vector<int> y_train, y_test;
    for (auto& s : train_data) y_train.push_back(s.label ? 1 : -1);
    for (auto& s : test_data)  y_test.push_back(s.label ? 1 : -1);

    KernelSVM svm(1.0);
    svm.fit(K_train, y_train);
    auto pred = svm.predict(K_test);
    double acc = svm.accuracy(pred, y_test);

    std::cout << "QFT Kernel SVM test accuracy: " << acc << "\n";
    std::cout << "Kernel build time: " << kernel_time << "s\n";

    std::cout << "\n=== Kernel comparison ===\n";
    std::cout << "RY Kernel SVM  (Exp 2): 0.92\n";
    std::cout << "QFT Kernel SVM (Exp 8): " << acc << "\n";

    std::ofstream log("qft_log.csv");
    log << "model,test_acc,time\n";
    log << "RY_kernel,0.92,0.98\n";
    log << "QFT_kernel," << acc << "," << kernel_time << "\n";

    std::cout << "\n=== Complete leaderboard ===\n";
    std::cout << "QNN standard 2L:    0.62\n";
    std::cout << "MLP:                0.80\n";
    std::cout << "Reuploading QNN 2L: 0.81\n";
    std::cout << "QRC (n=10):         0.89\n";
    std::cout << "RY Kernel SVM:      0.92\n";
    std::cout << "Reuploading QNN 4L: 0.95\n";
    std::cout << "QFT Kernel SVM:     " << acc << "\n";

    return 0;
}
