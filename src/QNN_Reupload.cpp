#include "QNN_Reupload.h"
#include <random>
#include <fstream>
#include <iostream>
#include <numbers>

// Circuit parameter layout per layer l (4 params each):
//   4*l+0 : data  q0  (set to x1*π on every forward)
//   4*l+1 : data  q1  (set to x2*π on every forward)
//   4*l+2 : train q0
//   4*l+3 : train q1

QNN_Reupload::QNN_Reupload(int n_layers)
    : n_layers(n_layers), n_trainable(2 * n_layers), circuit(2) {

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-std::numbers::pi, std::numbers::pi);

    for (int l = 0; l < n_layers; ++l) {
        circuit.add_ry(0.0, 0);
        circuit.add_ry(0.0, 1);
        circuit.add_ry(dist(rng), 0);
        circuit.add_ry(dist(rng), 1);
        circuit.add_cnot(0, 1);
    }
}

double QNN_Reupload::forward(double x1, double x2) {
    for (int l = 0; l < n_layers; ++l) {
        circuit.set_parameter(4 * l + 0, x1 * std::numbers::pi);
        circuit.set_parameter(4 * l + 1, x2 * std::numbers::pi);
    }
    QuantumState s(2);
    circuit.run(s);
    return s.expectation_z(0);
}

int QNN_Reupload::predict(double x1, double x2) {
    return forward(x1, x2) > 0.0 ? 1 : 0;
}

double QNN_Reupload::accuracy(const std::vector<Sample>& data) {
    if (data.empty()) return 0.0;
    int correct = 0;
    for (const auto& s : data)
        if (predict(s.x1, s.x2) == s.label) ++correct;
    return static_cast<double>(correct) / static_cast<double>(data.size());
}

double QNN_Reupload::param_shift_grad(double x1, double x2, int trainable_idx) {
    // trainable_idx in [0, 2*n_layers): maps to circuit param 4*(idx/2) + 2 + (idx%2)
    int idx = 4 * (trainable_idx / 2) + 2 + (trainable_idx % 2);
    const double theta = circuit.get_parameter(idx);
    circuit.set_parameter(idx, theta + std::numbers::pi / 2.0);
    double f_plus = forward(x1, x2);
    circuit.set_parameter(idx, theta - std::numbers::pi / 2.0);
    double f_minus = forward(x1, x2);
    circuit.set_parameter(idx, theta);
    return (f_plus - f_minus) / 2.0;
}

void QNN_Reupload::train(const std::vector<Sample>& data, int epochs, double lr,
                         const std::vector<Sample>& test_data, const std::string& log_file) {
    std::ofstream log;
    if (!log_file.empty()) {
        log.open(log_file);
        log << "epoch,train_acc,test_acc\n";
    }

    const double n = static_cast<double>(data.size());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<double> grad(n_trainable, 0.0);

        for (const auto& sample : data) {
            double y         = forward(sample.x1, sample.x2);
            double label_pm1 = sample.label ? 1.0 : -1.0;
            double error     = y - label_pm1;

            for (int i = 0; i < n_trainable; ++i)
                grad[i] += 2.0 * error * param_shift_grad(sample.x1, sample.x2, i);
        }

        for (int i = 0; i < n_trainable; ++i) {
            int idx = 4 * (i / 2) + 2 + (i % 2);
            double updated = circuit.get_parameter(idx) - lr * grad[i] / n;
            circuit.set_parameter(idx, updated);
        }

        bool print = ((epoch + 1) % 10 == 0 || epoch == 0);
        if (print || log.is_open()) {
            double train_acc = accuracy(data);
            double test_acc  = test_data.empty() ? 0.0 : accuracy(test_data);
            if (log.is_open())
                log << epoch + 1 << "," << train_acc << "," << test_acc << "\n";
            if (print)
                std::cout << "Reup [" << epoch + 1 << "/" << epochs << "]"
                          << "  train=" << train_acc << "  test=" << test_acc << "\n";
        }
    }
}
