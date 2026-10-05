#include "QuantumKernel.h"
#include <numbers>

QuantumKernel::QuantumKernel(int n_qubits) : n_qubits(n_qubits) {}

QuantumState QuantumKernel::encode(double x1, double x2) const {
    QuantumState s(n_qubits);

    // Warstwa 1: superpozycja
    const auto H    = QuantumGate::H();
    for (int q = 0; q < n_qubits; ++q) s.apply_single_qubit(H, q);

    // Warstwy 2 + 4: enkodowanie kątowe (RY)
    const auto ry1  = QuantumGate::RY(x1 * std::numbers::pi);
    const auto ry2  = QuantumGate::RY(x2 * std::numbers::pi);
    auto encode_layer = [&]() {
        s.apply_single_qubit(ry1, 0);
        s.apply_single_qubit(ry2, 1);
        s.apply_single_qubit(ry1, 2);
        s.apply_single_qubit(ry2, 3);
    };

    encode_layer();               // warstwa 2

    // Warstwa 3: splątanie
    s.apply_cnot(0, 1);
    s.apply_cnot(2, 3);

    encode_layer();               // warstwa 4

    return s;
}

double QuantumKernel::inner_product(const QuantumState& a, const QuantumState& b) {
    Complex sum(0.0, 0.0);
    for (size_t i = 0; i < a.size(); ++i)
        sum += std::conj(a[i]) * b[i];
    return std::norm(sum);  // |<a|b>|²
}

double QuantumKernel::kernel(double x1i, double x2i, double x1j, double x2j) const {
    const auto si = encode(x1i, x2i);
    const auto sj = encode(x1j, x2j);
    return inner_product(si, sj);
}

std::vector<std::vector<double>> QuantumKernel::kernel_matrix(
    const std::vector<Sample>& data) const
{
    const int n = static_cast<int>(data.size());
    std::vector<std::vector<double>> K(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        K[i][i] = 1.0;  // K(x,x) = |<ψ|ψ>|² = 1 (stan znormalizowany)
        for (int j = i + 1; j < n; ++j) {
            double v    = kernel(data[i].x1, data[i].x2, data[j].x1, data[j].x2);
            K[i][j] = K[j][i] = v;
        }
    }
    return K;
}

std::vector<std::vector<double>> QuantumKernel::kernel_matrix(
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
