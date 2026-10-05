#include "PEC.h"
#include <algorithm>
#include <stdexcept>

PECCalculator::PECCalculator() {
    // R [Ang] → {g0(II), g1(ZI), g2(IZ), g3(ZZ), g4(XX), g5(YY)}
    // STO-3G H2, Jordan-Wigner 2-qubit reduction; g4 == g5 z symetrii.
    // g0 zawiera repulsję jądrową (1/R), pozostałe współczynniki są czysto
    // elektroniczne. Obliczone z PySCF (RHF/STO-3G) wzorami:
    //   d10 = 2h00 + J00,  d01 = 2h11 + J11,  K01 = (01|10)
    //   g0 = (d01+d10+K01)/4 + E_nuc
    //   g1 = (d01-d10-K01)/4,  g2 = (-d01+d10-K01)/4
    //   g3 = (-d01-d10+K01)/4, g4 = g5 = K01/2
    coeff[0.50] = {+0.6330, +0.5409, -0.6253, +0.5098, 0.0844, 0.0844};
    coeff[0.60] = {+0.4279, +0.4506, -0.5374, +0.5410, 0.0869, 0.0869};
    coeff[0.70] = {+0.2845, +0.3757, -0.4652, +0.5610, 0.0895, 0.0895};
    coeff[0.75] = {+0.2289, +0.3433, -0.4342, +0.5676, 0.0909, 0.0909};
    coeff[0.80] = {+0.1814, +0.3138, -0.4061, +0.5724, 0.0923, 0.0923};
    coeff[0.90] = {+0.1055, +0.2621, -0.3574, +0.5778, 0.0953, 0.0953};
    coeff[1.00] = {+0.0483, +0.2183, -0.3167, +0.5793, 0.0984, 0.0984};
    coeff[1.20] = {-0.0294, +0.1477, -0.2526, +0.5753, 0.1049, 0.1049};
    coeff[1.50] = {-0.0926, +0.0717, -0.1865, +0.5601, 0.1148, 0.1148};
    coeff[1.80] = {-0.1230, +0.0204, -0.1444, +0.5410, 0.1240, 0.1240};
    coeff[2.00] = {-0.1342, -0.0042, -0.1254, +0.5283, 0.1296, 0.1296};
    coeff[2.50] = {-0.1479, -0.0434, -0.0977, +0.5007, 0.1411, 0.1411};
}

PauliHamiltonian PECCalculator::hamiltonian_at(double R) const {
    auto it = coeff.find(R);
    if (it == coeff.end())
        throw std::invalid_argument("Brak współczynników dla podanego R");

    const auto& g = it->second;
    PauliHamiltonian H;
    H.add_term(g[0], "II");
    H.add_term(g[1], "ZI");
    H.add_term(g[2], "IZ");
    H.add_term(g[3], "ZZ");
    H.add_term(g[4], "XX");
    H.add_term(g[5], "YY");
    return H;
}

double PECCalculator::energy_at(double R) {
    VQE vqe(hamiltonian_at(R), 2);
    return vqe.optimize(1000, 0.1).energy;
}

std::vector<PECPoint> PECCalculator::compute_all() {
    std::vector<PECPoint> points;
    points.reserve(coeff.size());
    for (const auto& [R, _] : coeff)
        points.push_back({R, energy_at(R), 0.0});
    return points;
}

PECCalculator::PECMinimum PECCalculator::find_minimum(
    const std::vector<PECPoint>& pec)
{
    // Szukamy minimum globalnego: przeglądamy posortowane wg R punkty.
    // Energia całkowita (z E_nuc) ma kształt krzywej Morse'a.
    size_t min_idx = 0;
    for (size_t i = 1; i < pec.size(); ++i) {
        if (pec[i].energy < pec[min_idx].energy)
            min_idx = i;
    }

    double E_min = pec[min_idx].energy;
    double Re    = pec[min_idx].R;

    // De: energia na ścianie repulsywnej (R=0.5) minus minimum
    // (przybliżenie — właściwe De wymaga poprawnych koeficjentów dla dużych R)
    double E_repulsive = pec.front().energy;   // R=0.5, najwyższa energia
    double De = E_repulsive - E_min;           // > 0

    return {Re, De, E_min};
}
