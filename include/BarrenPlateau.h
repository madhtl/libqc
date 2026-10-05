#ifndef LIBQC_BARRENPLATEAU_H
#define LIBQC_BARRENPLATEAU_H

#include <vector>

struct BPResult {
    int    n_qubits;
    double mean_gradient;
    double variance_gradient;
    double std_gradient;
};

class BarrenPlateauExperiment {
public:
    BPResult              run(int n_qubits, int n_samples = 200);
    std::vector<BPResult> run_all(int max_qubits = 8, int n_samples = 200);

private:
    double compute_gradient(int n_qubits, const std::vector<double>& theta);
};

#endif //LIBQC_BARRENPLATEAU_H
