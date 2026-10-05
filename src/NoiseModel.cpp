#include "NoiseModel.h"
#include <numbers>
#include <fstream>
#include <iostream>

// ── NoiseModel ────────────────────────────────────────────────────────────────

NoiseModel::NoiseModel(double p, unsigned seed)
    : p_(p), rng_(seed), dist_(0.0, 1.0) {}

void NoiseModel::apply_depolarizing(QuantumState& state, size_t qubit) const {
    if (p_ <= 0.0) return;
    const double r = dist_(rng_);
    if      (r < p_ / 3.0)        state.apply_single_qubit(QuantumGate::X(), qubit);
    else if (r < 2.0 * p_ / 3.0)  state.apply_single_qubit(QuantumGate::Y(), qubit);
    else if (r < p_)               state.apply_single_qubit(QuantumGate::Z(), qubit);
}

// ── NoisyQNN ─────────────────────────────────────────────────────────────────

NoisyQNN::NoisyQNN(int n_layers, double noise_p, unsigned seed)
    : QNN(n_layers),
      my_n_layers(n_layers),
      my_n_trainable(2 * n_layers),
      params(2 + 2 * n_layers, 0.0),
      noise(noise_p, seed)
{
    // Match QNN's parameter initialization (seed 42, uniform[-π, π])
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-std::numbers::pi, std::numbers::pi);
    for (int l = 0; l < my_n_layers; ++l) {
        params[2 + 2*l] = dist(rng);
        params[3 + 2*l] = dist(rng);
    }
}

double NoisyQNN::forward(double x1, double x2) {
    QuantumState s(2);

    // Encoding layer: RY(x1π) q0, RY(x2π) q1
    s.apply_single_qubit(QuantumGate::RY(x1 * std::numbers::pi), 0);
    noise.apply_depolarizing(s, 0);
    s.apply_single_qubit(QuantumGate::RY(x2 * std::numbers::pi), 1);
    noise.apply_depolarizing(s, 1);

    // Trainable layers: RY(θ) q0, RY(θ) q1, CNOT(0,1)
    for (int l = 0; l < my_n_layers; ++l) {
        s.apply_single_qubit(QuantumGate::RY(params[2 + 2*l]), 0);
        noise.apply_depolarizing(s, 0);
        s.apply_single_qubit(QuantumGate::RY(params[3 + 2*l]), 1);
        noise.apply_depolarizing(s, 1);
        s.apply_cnot(0, 1);
        noise.apply_depolarizing(s, 0);
        noise.apply_depolarizing(s, 1);
    }

    return s.expectation_z(0);
}

int NoisyQNN::predict(double x1, double x2) {
    return forward(x1, x2) > 0.0 ? 1 : 0;
}

double NoisyQNN::accuracy(const std::vector<Sample>& data) {
    if (data.empty()) return 0.0;
    int correct = 0;
    for (const auto& s : data)
        if (predict(s.x1, s.x2) == s.label) ++correct;
    return static_cast<double>(correct) / static_cast<double>(data.size());
}

double NoisyQNN::param_shift_grad(double x1, double x2, int trainable_idx) {
    const int    idx   = 2 + trainable_idx;
    const double theta = params[idx];
    params[idx] = theta + std::numbers::pi / 2.0;
    double f_plus = forward(x1, x2);
    params[idx] = theta - std::numbers::pi / 2.0;
    double f_minus = forward(x1, x2);
    params[idx] = theta;
    return (f_plus - f_minus) / 2.0;
}

void NoisyQNN::train(const std::vector<Sample>& data, int epochs, double lr,
                     const std::vector<Sample>& test_data, const std::string& log_file)
{
    std::ofstream log;
    if (!log_file.empty()) {
        log.open(log_file);
        log << "epoch,train_acc,test_acc\n";
    }

    const double n = static_cast<double>(data.size());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<double> grad(my_n_trainable, 0.0);

        for (const auto& sample : data) {
            double y         = forward(sample.x1, sample.x2);
            double label_pm1 = sample.label ? 1.0 : -1.0;
            double error     = y - label_pm1;
            for (int i = 0; i < my_n_trainable; ++i)
                grad[i] += 2.0 * error * param_shift_grad(sample.x1, sample.x2, i);
        }

        for (int i = 0; i < my_n_trainable; ++i)
            params[2 + i] -= lr * grad[i] / n;

        bool print = ((epoch + 1) % 10 == 0 || epoch == 0);
        if (print || log.is_open()) {
            double train_acc = accuracy(data);
            double test_acc  = test_data.empty() ? 0.0 : accuracy(test_data);
            if (log.is_open())
                log << epoch + 1 << "," << train_acc << "," << test_acc << "\n";
            if (print)
                std::cout << "NoisyQNN [" << epoch+1 << "/" << epochs << "]"
                          << "  train=" << train_acc << "  test=" << test_acc << "\n";
        }
    }
}

// ── NoisyQuantumKernel ────────────────────────────────────────────────────────

NoisyQuantumKernel::NoisyQuantumKernel(int n_qubits, double noise_p, unsigned seed)
    : QuantumKernel(n_qubits), my_n_qubits(n_qubits), noise(noise_p, seed) {}

QuantumState NoisyQuantumKernel::encode(double x1, double x2) const {
    QuantumState s(my_n_qubits);

    const auto H = QuantumGate::H();
    for (int q = 0; q < my_n_qubits; ++q) {
        s.apply_single_qubit(H, q);
        noise.apply_depolarizing(s, q);
    }

    const auto ry1 = QuantumGate::RY(x1 * std::numbers::pi);
    const auto ry2 = QuantumGate::RY(x2 * std::numbers::pi);

    auto encode_layer = [&]() {
        s.apply_single_qubit(ry1, 0); noise.apply_depolarizing(s, 0);
        s.apply_single_qubit(ry2, 1); noise.apply_depolarizing(s, 1);
        s.apply_single_qubit(ry1, 2); noise.apply_depolarizing(s, 2);
        s.apply_single_qubit(ry2, 3); noise.apply_depolarizing(s, 3);
    };

    encode_layer();

    s.apply_cnot(0, 1);
    noise.apply_depolarizing(s, 0);
    noise.apply_depolarizing(s, 1);
    s.apply_cnot(2, 3);
    noise.apply_depolarizing(s, 2);
    noise.apply_depolarizing(s, 3);

    encode_layer();

    return s;
}

double NoisyQuantumKernel::inner_product(const QuantumState& a, const QuantumState& b) {
    Complex sum(0.0, 0.0);
    for (size_t i = 0; i < a.size(); ++i)
        sum += std::conj(a[i]) * b[i];
    return std::norm(sum);  // |<a|b>|²
}

double NoisyQuantumKernel::kernel(double x1i, double x2i, double x1j, double x2j) const {
    return inner_product(encode(x1i, x2i), encode(x1j, x2j));
}

std::vector<std::vector<double>> NoisyQuantumKernel::kernel_matrix(
    const std::vector<Sample>& data) const
{
    const int n = static_cast<int>(data.size());
    std::vector<std::vector<double>> K(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        K[i][i] = 1.0;
        for (int j = i + 1; j < n; ++j) {
            double v = kernel(data[i].x1, data[i].x2, data[j].x1, data[j].x2);
            K[i][j] = K[j][i] = v;
        }
    }
    return K;
}

std::vector<std::vector<double>> NoisyQuantumKernel::kernel_matrix_test(
    const std::vector<Sample>& test_data,
    const std::vector<Sample>& train_data) const
{
    const int nt = static_cast<int>(test_data.size());
    const int nr = static_cast<int>(train_data.size());
    std::vector<std::vector<double>> K(nt, std::vector<double>(nr, 0.0));
    for (int i = 0; i < nt; ++i)
        for (int j = 0; j < nr; ++j)
            K[i][j] = kernel(test_data[i].x1, test_data[i].x2,
                             train_data[j].x1, train_data[j].x2);
    return K;
}
