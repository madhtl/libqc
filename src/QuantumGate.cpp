//
// Created by Madzia Hatala on 08/10/2025.
//

#include "../include/QuantumGate.h"
#include <cmath>

QuantumGate::QuantumGate(Matrix m, std::string n)
    : data(std::move(m)), name(std::move(n)) {}

QuantumGate QuantumGate::X() {
    Matrix m(2, 2);
    m(0, 1) = Complex(1.0, 0.0);
    m(1, 0) = Complex(1.0, 0.0);
    return QuantumGate(m, "X");
}

QuantumGate QuantumGate::H() {
    Matrix m(2, 2);
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    m(0, 0) = Complex( inv_sqrt2, 0.0);
    m(0, 1) = Complex( inv_sqrt2, 0.0);
    m(1, 0) = Complex( inv_sqrt2, 0.0);
    m(1, 1) = Complex(-inv_sqrt2, 0.0);
    return QuantumGate(m, "H");
}

QuantumGate QuantumGate::RZ(double theta) {
    Matrix m(2, 2);
    m(0, 0) = std::exp(Complex(0.0, -theta / 2.0));
    m(1, 1) = std::exp(Complex(0.0,  theta / 2.0));
    return QuantumGate(m, "RZ");
}

QuantumGate QuantumGate::RY(double theta) {
    Matrix m(2, 2);
    m(0, 0) = Complex( std::cos(theta / 2.0), 0.0);
    m(0, 1) = Complex(-std::sin(theta / 2.0), 0.0);
    m(1, 0) = Complex( std::sin(theta / 2.0), 0.0);
    m(1, 1) = Complex( std::cos(theta / 2.0), 0.0);
    return QuantumGate(m, "RY");
}

QuantumGate QuantumGate::Y() {
    Matrix m(2, 2);
    m(0, 1) = Complex(0.0, -1.0);
    m(1, 0) = Complex(0.0,  1.0);
    return QuantumGate(m, "Y");
}

QuantumGate QuantumGate::Z() {
    Matrix m(2, 2);
    m(0, 0) = Complex( 1.0, 0.0);
    m(1, 1) = Complex(-1.0, 0.0);
    return QuantumGate(m, "Z");
}

QuantumGate QuantumGate::CP(double theta) {
    Matrix m(2, 2);
    m(0, 0) = Complex(1.0, 0.0);
    m(1, 1) = std::exp(Complex(0.0, theta));
    return QuantumGate(m, "CP");
}

// Macierz 4x4 dla referencji; obwód używa apply_cnot() bezpośrednio na statevectorze.
QuantumGate QuantumGate::CNOT() {
    Matrix m(4, 4);
    m(0, 0) = Complex(1.0, 0.0);
    m(1, 1) = Complex(1.0, 0.0);
    m(2, 3) = Complex(1.0, 0.0);
    m(3, 2) = Complex(1.0, 0.0);
    return QuantumGate(m, "CNOT");
}
