#include "VQE.h"
#include <cmath>
#include <numbers>
#include <stdexcept>

// ── PauliHamiltonian ──────────────────────────────────────────────────────────

void PauliHamiltonian::add_term(double coeff, std::string ops) {
    terms.push_back({coeff, std::move(ops)});
}

// Apply Pauli string to a copy of state, then compute Re(<original|transformed>).
double PauliHamiltonian::pauli_expectation(QuantumState state,
                                            const std::string& ops) const {
    // Save original amplitudes before applying Pauli operators.
    QuantumState original = state;

    for (size_t q = 0; q < ops.size(); ++q) {
        char c = ops[q];
        if (c == 'X') state.apply_single_qubit(QuantumGate::X(), q);
        else if (c == 'Y') state.apply_single_qubit(QuantumGate::Y(), q);
        else if (c == 'Z') state.apply_single_qubit(QuantumGate::Z(), q);
        // 'I' — identity, no-op
    }

    // <ψ|P|ψ> = Re(Σᵢ conj(ψ[i]) * (P|ψ>)[i])
    Complex sum(0.0, 0.0);
    for (size_t i = 0; i < original.size(); ++i)
        sum += std::conj(original[i]) * state[i];
    return sum.real();
}

double PauliHamiltonian::expectation(QuantumState state) const {
    double E = 0.0;
    for (const auto& term : terms)
        E += term.coeff * pauli_expectation(state, term.ops);
    return E;
}

// H₂ Hamiltonian (STO-3G, R=0.735 Å, Jordan-Wigner, 2 qubits)
// H = -0.4804·II + 0.3435·ZI + (-0.4347)·IZ + 0.5716·ZZ + 0.0910·YY + 0.0910·XX
PauliHamiltonian PauliHamiltonian::H2() {
    PauliHamiltonian H;
    H.add_term(-0.4804, "II");
    H.add_term( 0.3435, "ZI");
    H.add_term(-0.4347, "IZ");
    H.add_term( 0.5716, "ZZ");
    H.add_term( 0.0910, "YY");
    H.add_term( 0.0910, "XX");
    return H;
}

// ── VQE ──────────────────────────────────────────────────────────────────────

VQE::VQE(PauliHamiltonian hamiltonian, int n_qubits)
    : H(std::move(hamiltonian)), n_qubits(n_qubits) {}

// HEA ansatz (2 qubits, 4 parameters):
// |00> -> RY(θ0,q0) -> RY(θ1,q1) -> CNOT(0,1) -> RY(θ2,q0) -> RY(θ3,q1)
QuantumState VQE::ansatz(const std::vector<double>& theta) const {
    QuantumState s(n_qubits);
    s.apply_single_qubit(QuantumGate::RY(theta[0]), 0);
    s.apply_single_qubit(QuantumGate::RY(theta[1]), 1);
    s.apply_cnot(0, 1);
    s.apply_single_qubit(QuantumGate::RY(theta[2]), 0);
    s.apply_single_qubit(QuantumGate::RY(theta[3]), 1);
    return s;
}

double VQE::energy(const std::vector<double>& theta) {
    return H.expectation(ansatz(theta));
}

VQE::VQEResult VQE::optimize(int n_steps, double lr) {
    const int n_params = 4;
    std::vector<double> theta(n_params, 0.1);
    std::vector<double> history;
    history.reserve(n_steps);

    const double shift = std::numbers::pi / 2.0;

    for (int step = 0; step < n_steps; ++step) {
        // Parameter shift gradient: dE/dθᵢ = (E(θᵢ+π/2) - E(θᵢ-π/2)) / 2
        std::vector<double> grad(n_params);
        for (int i = 0; i < n_params; ++i) {
            auto tp = theta; tp[i] += shift;
            auto tm = theta; tm[i] -= shift;
            grad[i] = (energy(tp) - energy(tm)) / 2.0;
        }

        // Gradient descent (minimizing energy)
        for (int i = 0; i < n_params; ++i)
            theta[i] -= lr * grad[i];

        history.push_back(energy(theta));
    }

    return {energy(theta), theta, history};
}
