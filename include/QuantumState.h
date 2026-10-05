#ifndef LIBQC_QUANTUMSTATE_H
#define LIBQC_QUANTUMSTATE_H

#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <random>

typedef std::complex<double> Complex;

class QuantumGate;

class QuantumState {
public:
    explicit QuantumState(size_t n);

    Complex& operator[](size_t i)       { return states[i]; }
    const Complex& operator[](size_t i) const { return states[i]; }

    size_t size()      const { return states.size(); }
    size_t numQubits() const { return n_qubits; }

    bool   isNormalized() const;
    double probability(size_t i) const { return std::norm(states[i]); }

    // Aplikuje bramkę 2x2 na kubicie target.
    void apply_single_qubit(const QuantumGate& gate, size_t target);

    // CNOT: control=kubit sterujący, target=kubit docelowy.
    void apply_cnot(size_t control, size_t target);

    // Controlled-phase: mnoży amplitudy gdzie oba control i target = |1⟩ przez e^{iθ}.
    void apply_controlled_phase(double theta, size_t control, size_t target);

    // Losuje wynik pomiaru wg. rozkładu |amplituda|² i kolapsuje stan.
    size_t measure();

    // Wartość oczekiwana operatora Pauliego Z dla kubitu qubit: sum_i sign(i) * |a_i|²
    double expectation_z(size_t qubit) const;

private:
    size_t n_qubits;
    std::vector<Complex> states;
};

#endif //LIBQC_QUANTUMSTATE_H
