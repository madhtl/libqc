//
// Created by Madzia Hatala on 23/09/2025.
//

#include "../include/Qubit.h"
#include <cmath>

void Qubit::negate() {
    std::swap(states[0], states[1]);
}

void Qubit::rotateX(double theta) {
    Complex a = states[0], b = states[1];
    states[0] = std::cos(theta / 2.0) * a - Complex(0, 1) * std::sin(theta / 2.0) * b;
    states[1] = -Complex(0, 1) * std::sin(theta / 2.0) * a + std::cos(theta / 2.0) * b;
}

void Qubit::rotateY(double theta) {
    Complex a = states[0], b = states[1];
    states[0] =  std::cos(theta / 2.0) * a - std::sin(theta / 2.0) * b;
    states[1] =  std::sin(theta / 2.0) * a + std::cos(theta / 2.0) * b;
}

void Qubit::rotateZ(double theta) {
    states[0] *= std::exp(Complex(0.0, -theta / 2.0));
    states[1] *= std::exp(Complex(0.0,  theta / 2.0));
}

void Qubit::phase(double phi) {
    states[1] *= std::exp(Complex(0.0, phi));
}
