#ifndef LIBQC_VQE_H
#define LIBQC_VQE_H

#include "QuantumState.h"
#include "QuantumGate.h"
#include <string>
#include <vector>

struct PauliTerm {
    double      coeff;
    std::string ops;   // one char per qubit: 'I','X','Y','Z'; ops[0] = qubit 0
};

class PauliHamiltonian {
public:
    void add_term(double coeff, std::string ops);
    double expectation(QuantumState state) const;

    static PauliHamiltonian H2();

private:
    std::vector<PauliTerm> terms;
    double pauli_expectation(QuantumState state, const std::string& ops) const;
};

class VQE {
public:
    VQE(PauliHamiltonian hamiltonian, int n_qubits);

    QuantumState ansatz(const std::vector<double>& theta) const;
    double energy(const std::vector<double>& theta);

    struct VQEResult {
        double              energy;
        std::vector<double> theta;
        std::vector<double> history;
    };

    VQEResult optimize(int n_steps = 200, double lr = 0.1);

private:
    PauliHamiltonian H;
    int n_qubits;
};

#endif //LIBQC_VQE_H
