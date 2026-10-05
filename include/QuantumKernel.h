#ifndef LIBQC_QUANTUMKERNEL_H
#define LIBQC_QUANTUMKERNEL_H

#include "QuantumState.h"
#include "QuantumGate.h"
#include "MakeMoons.h"
#include <vector>

// Quantum Feature Map + Fidelity Kernel
// K(xi, xj) = |<ψ(xi)|ψ(xj)>|²
//
// Enkodowanie 4-kubitowe (2 warstwy RY + splątanie):
//   H⊗4 → RY(x1π,0) RY(x2π,1) RY(x1π,2) RY(x2π,3)
//        → CNOT(0,1) CNOT(2,3)
//        → RY(x1π,0) RY(x2π,1) RY(x1π,2) RY(x2π,3)
class QuantumKernel {
public:
    explicit QuantumKernel(int n_qubits = 4);

    QuantumState encode(double x1, double x2) const;
    double kernel(double x1i, double x2i, double x1j, double x2j) const;

    // Pełna macierz n×n (symetryczna, O(n²) enkodowań)
    std::vector<std::vector<double>> kernel_matrix(const std::vector<Sample>& data) const;

    // Macierz n_test × n_train (K_test[i][j] = K(test_i, train_j))
    std::vector<std::vector<double>> kernel_matrix(
        const std::vector<Sample>& test_data,
        const std::vector<Sample>& train_data) const;

private:
    static double inner_product(const QuantumState& a, const QuantumState& b);
    int n_qubits;
};

#endif //LIBQC_QUANTUMKERNEL_H
