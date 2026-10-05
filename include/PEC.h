#ifndef LIBQC_PEC_H
#define LIBQC_PEC_H

#include "VQE.h"
#include <array>
#include <map>
#include <vector>

struct PECPoint {
    double R;       // odległość w Angstromach
    double energy;  // energia VQE w Hartree
    double error;   // |E_VQE - E_exact| jeśli znana, inaczej 0
};

class PECCalculator {
public:
    PECCalculator();

    double                energy_at(double R);
    std::vector<PECPoint> compute_all();

    struct PECMinimum {
        double Re;    // równowagowa długość wiązania [Ang]
        double De;    // głębokość minimum (energia dysocjacji) [Ha]
        double E_min; // energia w minimum [Ha]
    };
    PECMinimum find_minimum(const std::vector<PECPoint>& pec);

private:
    // g0, g1, g2, g3, g4 (XX), g5 (YY) dla każdego R — O'Malley et al. Table I
    std::map<double, std::array<double, 6>> coeff;

    PauliHamiltonian hamiltonian_at(double R) const;
};

#endif //LIBQC_PEC_H
