#include "QuantumReservoir.h"
#include <random>
#include <numbers>
#include <cmath>
#include <stdexcept>

// ── QuantumReservoir ─────────────────────────────────────────────────────────

QuantumReservoir::QuantumReservoir(int n_qubits, unsigned seed)
    : n_qubits(n_qubits) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-std::numbers::pi, std::numbers::pi);
    weights.resize(n_layers, std::vector<double>(n_qubits));
    for (auto& layer : weights)
        for (auto& w : layer)
            w = dist(rng);
}

ReservoirFeatures QuantumReservoir::encode(double x1, double x2) const {
    // Build a fresh circuit each call — weights are embedded as constants.
    QuantumCircuit circ(n_qubits);

    // Step 1: angle-encode data on qubits 0 and 1
    circ.add_ry(x1 * std::numbers::pi, 0);
    circ.add_ry(x2 * std::numbers::pi, 1);
    for (int q = 2; q < n_qubits; ++q)
        circ.add_ry(0.0, q);

    // Step 2: n_layers fixed random layers
    for (int l = 0; l < n_layers; ++l) {
        for (int q = 0; q < n_qubits; ++q)
            circ.add_ry(weights[l][q], q);
        for (int q = 0; q < n_qubits - 1; ++q)
            circ.add_cnot(q, q + 1);
    }

    // Step 3: measure <Zq> for all qubits
    QuantumState s(n_qubits);
    circ.run(s);

    ReservoirFeatures rf;
    rf.features.resize(n_qubits);
    for (int q = 0; q < n_qubits; ++q)
        rf.features[q] = s.expectation_z(q);
    return rf;
}

std::vector<ReservoirFeatures> QuantumReservoir::encode_all(
        const std::vector<Sample>& data) const {
    std::vector<ReservoirFeatures> out;
    out.reserve(data.size());
    for (const auto& s : data)
        out.push_back(encode(s.x1, s.x2));
    return out;
}

// ── RidgeClassifier ──────────────────────────────────────────────────────────

RidgeClassifier::RidgeClassifier(double lambda) : lambda(lambda) {}

void RidgeClassifier::fit(const std::vector<ReservoirFeatures>& X,
                           const std::vector<int>& labels) {
    if (X.empty()) return;
    int n = static_cast<int>(X.size());
    int f = static_cast<int>(X[0].features.size());
    int d = f + 1; // augmented: features + bias column

    // Build X_aug (n x d) and y (n)
    std::vector<std::vector<double>> Xaug(n, std::vector<double>(d));
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < f; ++j)
            Xaug[i][j] = X[i].features[j];
        Xaug[i][f] = 1.0;
        y[i] = static_cast<double>(labels[i]);
    }

    // A = X_aug^T * X_aug + lambda * I  (d x d)
    std::vector<std::vector<double>> A(d, std::vector<double>(d, 0.0));
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j)
            for (int k = 0; k < n; ++k)
                A[i][j] += Xaug[k][i] * Xaug[k][j];
        A[i][i] += lambda;
    }

    // b = X_aug^T * y  (d)
    std::vector<double> b(d, 0.0);
    for (int i = 0; i < d; ++i)
        for (int k = 0; k < n; ++k)
            b[i] += Xaug[k][i] * y[k];

    // Solve A*w = b via Gauss-Jordan with partial pivoting
    // Augmented matrix [A | b], size d x (d+1)
    std::vector<std::vector<double>> aug(d, std::vector<double>(d + 1));
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j)
            aug[i][j] = A[i][j];
        aug[i][d] = b[i];
    }

    for (int col = 0; col < d; ++col) {
        // Partial pivot
        int pivot = col;
        for (int row = col + 1; row < d; ++row)
            if (std::abs(aug[row][col]) > std::abs(aug[pivot][col]))
                pivot = row;
        std::swap(aug[col], aug[pivot]);

        double diag = aug[col][col];
        if (std::abs(diag) < 1e-14)
            throw std::runtime_error("RidgeClassifier: singular matrix");
        for (int j = col; j <= d; ++j)
            aug[col][j] /= diag;

        for (int row = 0; row < d; ++row) {
            if (row == col) continue;
            double factor = aug[row][col];
            for (int j = col; j <= d; ++j)
                aug[row][j] -= factor * aug[col][j];
        }
    }

    w.resize(f);
    for (int i = 0; i < f; ++i) w[i] = aug[i][d];
    bias = aug[f][d];
}

int RidgeClassifier::predict(const ReservoirFeatures& x) const {
    double val = bias;
    for (int i = 0; i < static_cast<int>(w.size()); ++i)
        val += w[i] * x.features[i];
    return val > 0.5 ? 1 : 0;
}

double RidgeClassifier::accuracy(const std::vector<ReservoirFeatures>& X,
                                  const std::vector<int>& labels) const {
    if (X.empty()) return 0.0;
    int correct = 0;
    for (int i = 0; i < static_cast<int>(X.size()); ++i)
        if (predict(X[i]) == labels[i]) ++correct;
    return static_cast<double>(correct) / static_cast<double>(X.size());
}
