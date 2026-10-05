#include "BarrenPlateau.h"
#include "QuantumState.h"
#include "QuantumGate.h"
#include <cmath>
#include <numbers>
#include <random>

// Ansatz: L = n_qubits warstw, każda: RY na każdym kubicie + łańcuch CNOT
// Łącznie L*n params. Gradient po theta[0] (RY q0 w warstwie 0).
// Obserwable: expectation_z(qubit 0)
double BarrenPlateauExperiment::compute_gradient(int n_qubits,
                                                  const std::vector<double>& theta)
{
    const double shift = std::numbers::pi / 2.0;
    const int    L     = n_qubits;            // głębokość = liczba kubitów

    auto run_circuit = [&](double theta0) -> double {
        QuantumState state(n_qubits);
        for (int l = 0; l < L; ++l) {
            // RY na każdym kubicie; theta[0] zastąpiony przesunięciem w l=0,q=0
            for (int q = 0; q < n_qubits; ++q) {
                double angle = (l == 0 && q == 0) ? theta0 : theta[l * n_qubits + q];
                state.apply_single_qubit(QuantumGate::RY(angle), q);
            }
            // Łańcuch CNOT
            for (int q = 0; q < n_qubits - 1; ++q)
                state.apply_cnot(q, q + 1);
        }
        return state.expectation_z(0);
    };

    return (run_circuit(theta[0] + shift) - run_circuit(theta[0] - shift)) / 2.0;
}

BPResult BarrenPlateauExperiment::run(int n_qubits, int n_samples) {
    const int n_params = n_qubits * n_qubits;  // L=n warstw × n kubitów
    std::vector<double> gradients(n_samples);

    std::uniform_real_distribution<double> dist(-std::numbers::pi, std::numbers::pi);

    for (int s = 0; s < n_samples; ++s) {
        std::mt19937 rng(static_cast<unsigned>(s * 1000 + n_qubits));
        std::vector<double> theta(n_params);
        for (auto& t : theta) t = dist(rng);
        gradients[s] = compute_gradient(n_qubits, theta);
    }

    // E[g] i E[g²]
    double sum = 0.0, sum_sq = 0.0;
    for (double g : gradients) { sum += g; sum_sq += g * g; }
    double mean     = sum    / n_samples;
    double mean_sq  = sum_sq / n_samples;
    double variance = mean_sq - mean * mean;

    return {n_qubits, mean, variance, std::sqrt(std::abs(variance))};
}

std::vector<BPResult> BarrenPlateauExperiment::run_all(int max_qubits, int n_samples) {
    std::vector<BPResult> results;
    results.reserve(max_qubits - 1);
    for (int n = 2; n <= max_qubits; ++n)
        results.push_back(run(n, n_samples));
    return results;
}
