//
// Created by Madzia Hatala on 08/10/2025.
//

#ifndef LIBQC_QUANTUMCIRCUIT_H
#define LIBQC_QUANTUMCIRCUIT_H

#include "QuantumGate.h"
#include "QuantumState.h"
#include <variant>
#include <vector>
#include <stdexcept>

class QuantumCircuit {
public:
    explicit QuantumCircuit(size_t n_qubits);

    // Dodaj stałą bramkę jednokubitową.
    void add_gate(QuantumGate gate, size_t target);

    // Dodaj CNOT.
    void add_cnot(size_t control, size_t target);

    // Dodaj parametryczną bramkę RZ; zwraca indeks parametru do set_parameter().
    size_t add_rz(double theta, size_t target);

    // Analogicznie dla RY (obrót w osi Y — zmienia amplitudy, nie tylko fazy).
    size_t add_ry(double theta, size_t target);

    // Wykonaj cały obwód na stanie state.
    void run(QuantumState& state) const;

    // Zarządzanie parametrami (dla trenowania QNN).
    void   set_parameter(size_t idx, double value);
    double get_parameter(size_t idx) const;
    size_t num_parameters() const { return params.size(); }

private:
    struct SingleQubitOp { QuantumGate gate; size_t target; };
    struct CNotOp        { size_t control;   size_t target; };
    struct RzOp          { size_t target;    size_t param_idx; };
    struct RyOp          { size_t target;    size_t param_idx; };

    using Operation = std::variant<SingleQubitOp, CNotOp, RzOp, RyOp>;

    size_t n_qubits;
    std::vector<Operation> ops;
    std::vector<double>    params;
};

#endif //LIBQC_QUANTUMCIRCUIT_H
