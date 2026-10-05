#ifndef LIBQC_NOISEMODEL_H
#define LIBQC_NOISEMODEL_H

#include "QuantumState.h"
#include "QuantumGate.h"
#include "QNN.h"
#include "QuantumKernel.h"
#include "MakeMoons.h"
#include <random>
#include <vector>
#include <string>

class NoiseModel {
public:
    explicit NoiseModel(double p = 0.0, unsigned seed = 42);

    // With prob p/3: apply X; p/3: Y; p/3: Z; else: nothing
    void apply_depolarizing(QuantumState& state, size_t qubit) const;

    double error_rate() const { return p_; }

private:
    double p_;
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<double> dist_;
};

// Inherits QNN but shadows forward/train/predict/accuracy with noisy versions.
// The parent's private circuit is unused; NoisyQNN maintains its own params.
class NoisyQNN : public QNN {
public:
    NoisyQNN(int n_layers, double noise_p, unsigned seed = 42);

    double forward(double x1, double x2);
    int    predict(double x1, double x2);
    void   train(const std::vector<Sample>& data, int epochs, double lr,
                 const std::vector<Sample>& test_data = {},
                 const std::string& log_file = "");
    double accuracy(const std::vector<Sample>& data);

private:
    double param_shift_grad(double x1, double x2, int trainable_idx);

    int my_n_layers;
    int my_n_trainable;
    std::vector<double> params;  // [x1_enc, x2_enc, θ_00, θ_01, θ_10, θ_11, ...]
    NoiseModel noise;
};

// Inherits QuantumKernel but shadows encode/kernel/kernel_matrix with noisy versions.
class NoisyQuantumKernel : public QuantumKernel {
public:
    NoisyQuantumKernel(int n_qubits, double noise_p, unsigned seed = 42);

    QuantumState encode(double x1, double x2) const;
    double kernel(double x1i, double x2i, double x1j, double x2j) const;

    std::vector<std::vector<double>> kernel_matrix(
        const std::vector<Sample>& data) const;
    std::vector<std::vector<double>> kernel_matrix_test(
        const std::vector<Sample>& test_data,
        const std::vector<Sample>& train_data) const;

private:
    static double inner_product(const QuantumState& a, const QuantumState& b);
    int my_n_qubits;
    NoiseModel noise;
};

#endif //LIBQC_NOISEMODEL_H
