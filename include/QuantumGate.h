//
// Created by Madzia Hatala on 08/10/2025.
//

#ifndef LIBQC_QUANTUMGATE_H
#define LIBQC_QUANTUMGATE_H

#include "Matrix.h"
#include <string>

class QuantumGate {
public:
    QuantumGate(Matrix m, std::string n);

    const Matrix& matrix() const { return data; }
    const std::string& getName() const { return name; }

    static QuantumGate X();
    static QuantumGate Y();
    static QuantumGate Z();
    static QuantumGate H();
    static QuantumGate RZ(double theta);
    static QuantumGate RY(double theta);
    static QuantumGate CP(double theta);  // [[1,0],[0,e^{iθ}]]
    static QuantumGate CNOT();

private:
    Matrix data;
    std::string name;
};

#endif //LIBQC_QUANTUMGATE_H
