#include "QFTKernel.h"
#include "QuantumGate.h"
#include <numbers>
#include <cmath>

QFTKernel::QFTKernel(int n_qubits) : n_qubits(n_qubits) {}

QuantumState QFTKernel::encode(double x1, double x2) const {
    QuantumState s(n_qubits);

    // Step 1: angle encode
    s.apply_single_qubit(QuantumGate::RY(x1 * std::numbers::pi), 0);
    s.apply_single_qubit(QuantumGate::RY(x2 * std::numbers::pi), 1);
    for (int q = 2; q < n_qubits; q += 2) {
        s.apply_single_qubit(QuantumGate::RY(x1 * std::numbers::pi), q);
        if (q + 1 < n_qubits)
            s.apply_single_qubit(QuantumGate::RY(x2 * std::numbers::pi), q + 1);
    }

    // Step 2: QFT (no bit reversal)
    for (int i = 0; i < n_qubits; ++i) {
        s.apply_single_qubit(QuantumGate::H(), i);
        for (int j = i + 1; j < n_qubits; ++j) {
            double theta = 2.0 * std::numbers::pi / static_cast<double>(size_t(1) << (j - i + 1));
            s.apply_controlled_phase(theta, i, j);
        }
    }

    return s;
}

double QFTKernel::inner_product(const QuantumState& a, const QuantumState& b) {
    Complex sum(0.0, 0.0);
    for (size_t i = 0; i < a.size(); ++i)
        sum += std::conj(a[i]) * b[i];
    return std::norm(sum);  // |⟨a|b⟩|²
}

double QFTKernel::kernel(double x1i, double x2i, double x1j, double x2j) const {
    return inner_product(encode(x1i, x2i), encode(x1j, x2j));
}

std::vector<std::vector<double>> QFTKernel::kernel_matrix(
        const std::vector<Sample>& data) const {
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

std::vector<std::vector<double>> QFTKernel::kernel_matrix_test(
        const std::vector<Sample>& test,
        const std::vector<Sample>& train) const {
    const int nt = static_cast<int>(test.size());
    const int nr = static_cast<int>(train.size());
    std::vector<std::vector<double>> K(nt, std::vector<double>(nr, 0.0));
    for (int i = 0; i < nt; ++i)
        for (int j = 0; j < nr; ++j)
            K[i][j] = kernel(test[i].x1, test[i].x2, train[j].x1, train[j].x2);
    return K;
}
