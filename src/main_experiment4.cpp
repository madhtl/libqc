#include "BarrenPlateau.h"
#include <iostream>
#include <iomanip>
#include <fstream>

int main() {
    std::cout << "=== Barren Plateaus Experiment ===\n\n";
    std::cout << "Measuring gradient variance for n=2..8 qubits\n";
    std::cout << "(200 random parameter sets per n)\n\n";

    BarrenPlateauExperiment bp;
    auto results = bp.run_all(8, 200);

    std::cout << std::setw(10) << "n_qubits"
              << std::setw(15) << "mean_grad"
              << std::setw(15) << "var_grad"
              << std::setw(15) << "std_grad\n";
    std::cout << std::string(55, '-') << "\n";

    std::cout << std::scientific << std::setprecision(4);
    for (auto& r : results) {
        std::cout << std::setw(10) << r.n_qubits
                  << std::setw(15) << r.mean_gradient
                  << std::setw(15) << r.variance_gradient
                  << std::setw(15) << r.std_gradient << "\n";
    }

    std::ofstream log("bp_log.csv");
    log << "n_qubits,mean_grad,var_grad,std_grad\n";
    for (auto& r : results)
        log << r.n_qubits << ","
            << r.mean_gradient << ","
            << r.variance_gradient << ","
            << r.std_gradient << "\n";

    std::cout << "\nSaved: bp_log.csv\n";

    std::cout << "\nVar[n+1]/var[n] ratio:\n";
    for (size_t i = 1; i < results.size(); ++i) {
        double ratio = results[i].variance_gradient /
                       results[i-1].variance_gradient;
        std::cout << std::fixed << std::setprecision(3)
                  << "  n=" << results[i-1].n_qubits
                  << " -> n=" << results[i].n_qubits
                  << ": ratio=" << ratio << "\n";
    }

    return 0;
}
