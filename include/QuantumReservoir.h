#ifndef LIBQC_QUANTUMRESERVOIR_H
#define LIBQC_QUANTUMRESERVOIR_H

#include "MakeMoons.h"
#include "QuantumCircuit.h"
#include <vector>

struct ReservoirFeatures {
    std::vector<double> features;
};

// Fixed random quantum circuit used as a nonlinear feature extractor.
// Weights are set once at construction and never updated.
class QuantumReservoir {
public:
    QuantumReservoir(int n_qubits = 6, unsigned seed = 42);

    // Encode one sample: data → fixed circuit → <Zq> for all q
    ReservoirFeatures encode(double x1, double x2) const;

    // Encode all samples in a dataset
    std::vector<ReservoirFeatures> encode_all(
        const std::vector<Sample>& data) const;

    int numFeatures() const { return n_qubits; }

private:
    int n_qubits;
    int n_layers = 3;
    std::vector<std::vector<double>> weights; // weights[layer][qubit]
};

// Closed-form ridge regression readout (no gradient descent).
// Adds a bias column and solves (X^T X + lambda I) w = X^T y.
class RidgeClassifier {
public:
    explicit RidgeClassifier(double lambda = 1e-3);

    void   fit(const std::vector<ReservoirFeatures>& X,
               const std::vector<int>& labels);
    int    predict(const ReservoirFeatures& x) const;
    double accuracy(const std::vector<ReservoirFeatures>& X,
                    const std::vector<int>& labels) const;

private:
    std::vector<double> w;
    double bias = 0.0;
    double lambda;
};

#endif //LIBQC_QUANTUMRESERVOIR_H
