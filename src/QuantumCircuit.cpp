//
// Created by Madzia Hatala on 08/10/2025.
//

#include "../include/QuantumCircuit.h"

QuantumCircuit::QuantumCircuit(size_t n_qubits) : n_qubits(n_qubits) {}

void QuantumCircuit::add_gate(QuantumGate gate, size_t target) {
    ops.push_back(SingleQubitOp{std::move(gate), target});
}

void QuantumCircuit::add_cnot(size_t control, size_t target) {
    ops.push_back(CNotOp{control, target});
}

size_t QuantumCircuit::add_rz(double theta, size_t target) {
    size_t idx = params.size();
    params.push_back(theta);
    ops.push_back(RzOp{target, idx});
    return idx;
}

size_t QuantumCircuit::add_ry(double theta, size_t target) {
    size_t idx = params.size();
    params.push_back(theta);
    ops.push_back(RyOp{target, idx});
    return idx;
}

void QuantumCircuit::run(QuantumState& state) const {
    for (const auto& op : ops) {
        std::visit([&](auto&& o) {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, SingleQubitOp>) {
                state.apply_single_qubit(o.gate, o.target);
            } else if constexpr (std::is_same_v<T, CNotOp>) {
                state.apply_cnot(o.control, o.target);
            } else if constexpr (std::is_same_v<T, RzOp>) {
                state.apply_single_qubit(QuantumGate::RZ(params[o.param_idx]), o.target);
            } else if constexpr (std::is_same_v<T, RyOp>) {
                state.apply_single_qubit(QuantumGate::RY(params[o.param_idx]), o.target);
            }
        }, op);
    }
}

void QuantumCircuit::set_parameter(size_t idx, double value) {
    if (idx >= params.size()) throw std::out_of_range("Parameter index out of range");
    params[idx] = value;
}

double QuantumCircuit::get_parameter(size_t idx) const {
    if (idx >= params.size()) throw std::out_of_range("Parameter index out of range");
    return params[idx];
}
