//
// Created by Madzia Hatala on 23/09/2025.
//

#ifndef LIBQC_QUBIT_H
#define LIBQC_QUBIT_H
#include <array>
#include <complex>
#include <iostream>
#include <vector>


using Complex = std::complex<double>;
class Qubit {
public:
    Qubit() {
        states = {Complex(1.0, 0.0), Complex(0.0, 0.0)}; //initialized in |0>
    }
    void negate();
    void rotateX(double theta);
    void rotateY(double theta);
    void rotateZ(double theta);
    void phase(double phi);

private:
    std::array<Complex, 2> states;

};


#endif //LIBQC_QUBIT_H
