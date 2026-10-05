#include "MakeMoons.h"
#include "QNN.h"
#include "QNN_Reupload.h"
#include <iostream>
#include <chrono>

using std::chrono::steady_clock;
using std::chrono::duration;

int main() {
    auto train_data = make_moons(200, 0.1, 42);
    auto test_data  = make_moons(100, 0.1, 123);

    // Standard QNN — 2 layers (same as Experiment 1)
    std::cout << "=== Standard QNN (2 layers, no reuploading) ===\n";
    QNN qnn_std(2);
    auto t1 = steady_clock::now();
    qnn_std.train(train_data, 100, 0.1, test_data, "reup_std_log.csv");
    double t_std = duration<double>(steady_clock::now() - t1).count();

    // Reuploading QNN — 2 layers
    std::cout << "\n=== Reuploading QNN (2 layers) ===\n";
    QNN_Reupload qnn_reu(2);
    auto t2 = steady_clock::now();
    qnn_reu.train(train_data, 100, 0.1, test_data, "reup_reu_log.csv");
    double t_reu = duration<double>(steady_clock::now() - t2).count();

    // Results
    std::cout << "\n=== Final Results ===\n";
    std::cout << "Standard QNN:    acc=" << qnn_std.accuracy(test_data)
              << "  time=" << t_std << "s\n";
    std::cout << "Reuploading QNN: acc=" << qnn_reu.accuracy(test_data)
              << "  time=" << t_reu << "s\n";

    // Also run with 4 layers to see if reuploading scales better
    std::cout << "\n=== 4-layer comparison ===\n";
    QNN qnn_std4(4);
    auto t3 = steady_clock::now();
    qnn_std4.train(train_data, 100, 0.1, test_data, "reup_std4_log.csv");
    double t_std4 = duration<double>(steady_clock::now() - t3).count();

    QNN_Reupload qnn_reu4(4);
    auto t4 = steady_clock::now();
    qnn_reu4.train(train_data, 100, 0.1, test_data, "reup_reu4_log.csv");
    double t_reu4 = duration<double>(steady_clock::now() - t4).count();

    std::cout << "Standard QNN 4L:    acc=" << qnn_std4.accuracy(test_data)
              << "  time=" << t_std4 << "s\n";
    std::cout << "Reuploading QNN 4L: acc=" << qnn_reu4.accuracy(test_data)
              << "  time=" << t_reu4 << "s\n";

    return 0;
}
