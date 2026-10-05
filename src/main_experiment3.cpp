#include "VQE.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

int main() {
    // Analytical ground state of this 2-qubit H2 Hamiltonian (eigenvalue of the 4x4 matrix)
    const double exact_energy = -1.851199;

    std::cout << "VQE for H2 (STO-3G, R=0.735 A)\n";
    std::cout << "Ansatz: HEA 2 qubits, 4 parameters\n";
    std::cout << "Optimization: gradient descent + parameter shift (200 steps, lr=0.1)\n\n";

    auto H = PauliHamiltonian::H2();
    VQE vqe(H, 2);

    auto t0 = std::chrono::steady_clock::now();
    auto result = vqe.optimize(200, 0.1);
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    double error = std::abs(result.energy - exact_energy);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "VQE energy:     " << result.energy  << " Ha\n";
    std::cout << "Exact energy:   " << exact_energy    << " Ha\n";
    std::cout << "Error:          " << error           << " Ha\n";
    std::cout << "Chemical accuracy (<0.01 Ha): "
              << (error < 0.01 ? "YES" : "NO") << "\n";
    std::cout << "Time:           " << elapsed         << " s\n";

    std::ofstream log("vqe_log.csv");
    log << "step,energy\n";
    for (size_t i = 0; i < result.history.size(); ++i)
        log << i + 1 << "," << result.history[i] << "\n";

    std::cout << "\nSaved: vqe_log.csv\n";
    return 0;
}
