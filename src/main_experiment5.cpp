#include "PEC.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

int main() {
    std::cout << "=== PEC dla H2 przez VQE ===\n\n";
    std::cout << "Obliczam energię dla 12 wartości R...\n\n";

    PECCalculator pec;

    auto t1 = std::chrono::steady_clock::now();
    auto points = pec.compute_all();
    double total_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t1).count();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(8)  << "R (A)"
              << std::setw(14) << "E_VQE (Ha)" << "\n";
    std::cout << std::string(22, '-') << "\n";
    for (auto& p : points)
        std::cout << std::setw(8)  << p.R
                  << std::setw(14) << p.energy << "\n";

    auto minPt = pec.find_minimum(points);
    std::cout << "\nWyniki:\n";
    std::cout << "Re    = " << minPt.Re    << " Ang  (ref: 0.75 Ang)\n";
    std::cout << "E_min = " << minPt.E_min << " Ha\n";
    std::cout << "De    = " << minPt.De    << " Ha  (od sciany repulsywnej R=0.5)\n";
    std::cout << "E_diss = ~-0.934 Ha (STO-3G, dwa atomy H)\n";
    std::cout << "Czas: " << total_time << " s\n";

    std::ofstream log("pec_log.csv");
    log << "R,energy\n";
    for (auto& p : points)
        log << p.R << "," << p.energy << "\n";

    std::cout << "\nZapisano pec_log.csv\n";
    return 0;
}
