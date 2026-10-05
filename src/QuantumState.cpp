//
// Created by Madzia Hatala on 08/10/2025.
//

#include "../include/QuantumState.h"
#include "../include/QuantumGate.h"

QuantumState::QuantumState(size_t n)
    : n_qubits(n), states(size_t(1) << n, Complex(0.0, 0.0)) {
    states[0] = Complex(1.0, 0.0);
}

bool QuantumState::isNormalized() const {
    double sum = 0.0;
    for (const auto& a : states) sum += std::norm(a);
    return std::abs(sum - 1.0) < 1e-9;
}

void QuantumState::apply_single_qubit(const QuantumGate& gate, size_t target) {
    const Matrix& m = gate.matrix();
    const size_t dim = states.size();
    const size_t mask = size_t(1) << target;
    for (size_t i = 0; i < dim; i++) {
        if (i & mask) continue;         // rozpatruję tylko pary (i, j) raz
        size_t j = i | mask;
        Complex a = states[i];
        Complex b = states[j];
        states[i] = m(0, 0) * a + m(0, 1) * b;
        states[j] = m(1, 0) * a + m(1, 1) * b;
    }
}

void QuantumState::apply_cnot(size_t control, size_t target) {
    const size_t dim = states.size();
    const size_t ctrl_mask = size_t(1) << control;
    const size_t tgt_mask  = size_t(1) << target;
    for (size_t i = 0; i < dim; i++) {
        if (!(i & ctrl_mask)) continue;  // control musi być |1>
        if (  i & tgt_mask  ) continue;  // target musi być |0> (para i,j raz)
        size_t j = i | tgt_mask;
        std::swap(states[i], states[j]);
    }
}

size_t QuantumState::measure() {
    std::vector<double> probs(states.size());
    for (size_t i = 0; i < states.size(); i++)
        probs[i] = std::norm(states[i]);

    static std::mt19937 rng(std::random_device{}());
    std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
    size_t outcome = dist(rng);

    std::fill(states.begin(), states.end(), Complex(0.0, 0.0));
    states[outcome] = Complex(1.0, 0.0);
    return outcome;
}

void QuantumState::apply_controlled_phase(double theta, size_t control, size_t target) {
    const size_t ctrl_mask = size_t(1) << control;
    const size_t tgt_mask  = size_t(1) << target;
    const Complex phase = std::exp(Complex(0.0, theta));
    for (size_t i = 0; i < states.size(); ++i)

        if ((i & ctrl_mask) && (i & tgt_mask))
            states[i] *= phase;
}

double QuantumState::expectation_z(size_t qubit) const {
    const size_t mask = size_t(1) << qubit;
    double result = 0.0;
    for (size_t i = 0; i < states.size(); i++) {
        double eigenvalue = (i & mask) ? -1.0 : 1.0;
        result += eigenvalue * std::norm(states[i]);
    }
    return result;
}

