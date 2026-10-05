#ifndef LIBQC_QFTKERNEL_H
#define LIBQC_QFTKERNEL_H

#include "MakeMoons.h"
#include "QuantumState.h"
#include <vector>

// QFT-based fidelity kernel.
// Feature map: angle-encode (x1,x2) then apply QFT → Fourier structured Hilbert space.
// K(xi,xj) = |⟨ψ(xi)|ψ(xj)⟩|²
class QFTKernel {
public:
    explicit QFTKernel(int n_qubits = 4);

    QuantumState encode(double x1, double x2) const;
    double kernel(double x1i, double x2i, double x1j, double x2j) const;

    // Full n×n symmetric kernel matrix (training data)
    std::vector<std::vector<double>> kernel_matrix(
        const std::vector<Sample>& data) const;

    // n_test × n_train kernel matrix
    std::vector<std::vector<double>> kernel_matrix_test(
        const std::vector<Sample>& test,
        const std::vector<Sample>& train) const;

private:
    static double inner_product(const QuantumState& a, const QuantumState& b);
    int n_qubits;
};

#endif //LIBQC_QFTKERNEL_H
