#ifndef LIBQC_QNN_H
#define LIBQC_QNN_H

#include "MakeMoons.h"
#include "QuantumCircuit.h"
#include <vector>
#include <string>

// 2-kubitowy QNN z n_layers warstwami RZ+CNOT.
// Enkodowanie: H(q0) H(q1) → RZ(x1π, q0) RZ(x2π, q1) [param 0,1]
// Warstwa l:   RZ(θ, q0) RZ(θ, q1) CNOT(0,1)          [param 2+2l, 3+2l]
// Pomiar:      expectation_z(q0)
class QNN {
public:
    explicit QNN(int n_layers);

    double forward(double x1, double x2);
    int    predict(double x1, double x2);
    void   train(const std::vector<Sample>& data, int epochs, double lr,
                 const std::vector<Sample>& test_data = {},
                 const std::string& log_file = "");
    double accuracy(const std::vector<Sample>& data);

private:
    double param_shift_grad(double x1, double x2, int trainable_idx);

    int n_layers;
    int n_trainable;
    QuantumCircuit circuit;
};

#endif //LIBQC_QNN_H
