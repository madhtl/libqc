#include "MLP.h"
#include <random>
#include <fstream>
#include <iostream>

MLP::MLP() : W1(8), b1(4, 0.0), W2(4), b2(0.0) {
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (auto& w : W1) w = dist(rng);
    for (auto& w : W2) w = dist(rng);
}

MLP::FwdCache MLP::forward_cached(double x1, double x2) const {
    const double x[2] = {x1, x2};
    std::vector<double> h1(4);
    for (int j = 0; j < 4; ++j) {
        double net = b1[j];
        for (int i = 0; i < 2; ++i) net += W1[j*2 + i] * x[i];
        h1[j] = sigmoid(net);
    }
    double net2 = b2;
    for (int j = 0; j < 4; ++j) net2 += W2[j] * h1[j];
    return {h1, sigmoid(net2)};
}

double MLP::forward(double x1, double x2) {
    return forward_cached(x1, x2).y;
}

int MLP::predict(double x1, double x2) {
    return forward(x1, x2) > 0.5 ? 1 : 0;
}

double MLP::accuracy(const std::vector<Sample>& data) {
    if (data.empty()) return 0.0;
    int correct = 0;
    for (const auto& s : data)
        if (predict(s.x1, s.x2) == s.label) ++correct;
    return static_cast<double>(correct) / static_cast<double>(data.size());
}

void MLP::train(const std::vector<Sample>& data, int epochs, double lr,
                const std::vector<Sample>& test_data, const std::string& log_file) {
    std::ofstream log;
    if (!log_file.empty()) {
        log.open(log_file);
        log << "epoch,train_acc,test_acc\n";
    }

    const double n = static_cast<double>(data.size());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<double> dW1(8, 0.0), db1_acc(4, 0.0), dW2(4, 0.0);
        double db2_acc = 0.0;

        for (const auto& sample : data) {
            auto [h1, y] = forward_cached(sample.x1, sample.x2);
            double label = static_cast<double>(sample.label);
            // MSE: L = (y - label)^2
            double delta2 = 2.0 * (y - label) * y * (1.0 - y);

            std::vector<double> delta1(4);
            for (int j = 0; j < 4; ++j)
                delta1[j] = W2[j] * delta2 * h1[j] * (1.0 - h1[j]);

            for (int j = 0; j < 4; ++j) {
                dW2[j]       += delta2 * h1[j];
                db1_acc[j]   += delta1[j];
                dW1[j*2]     += delta1[j] * sample.x1;
                dW1[j*2 + 1] += delta1[j] * sample.x2;
            }
            db2_acc += delta2;
        }

        for (int j = 0; j < 4; ++j) {
            W2[j]       -= lr * dW2[j]       / n;
            b1[j]       -= lr * db1_acc[j]   / n;
            W1[j*2]     -= lr * dW1[j*2]     / n;
            W1[j*2 + 1] -= lr * dW1[j*2 + 1] / n;
        }
        b2 -= lr * db2_acc / n;

        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            double train_acc = accuracy(data);
            double test_acc  = test_data.empty() ? 0.0 : accuracy(test_data);
            if (log.is_open())
                log << epoch + 1 << "," << train_acc << "," << test_acc << "\n";
            std::cout << "MLP  [" << epoch + 1 << "/" << epochs << "]"
                      << "  train=" << train_acc << "  test=" << test_acc << "\n";
        } else if (log.is_open()) {
            double train_acc = accuracy(data);
            double test_acc  = test_data.empty() ? 0.0 : accuracy(test_data);
            log << epoch + 1 << "," << train_acc << "," << test_acc << "\n";
        }
    }
}
