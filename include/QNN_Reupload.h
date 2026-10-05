#ifndef LIBQC_QNN_REUPLOAD_H
#define LIBQC_QNN_REUPLOAD_H

#include "MakeMoons.h"
#include "QuantumCircuit.h"
#include <vector>
#include <string>

// Data reuploading QNN (Pérez-Salinas et al. 2020).
// Per layer: RY(x1π,q0) RY(x2π,q1)  ← encode data again
//            RY(θ,q0)   RY(θ,q1)    ← trainable
//            CNOT(0,1)
// measure: expectation_z(q0)
class QNN_Reupload {
public:
    explicit QNN_Reupload(int n_layers);

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

#endif //LIBQC_QNN_REUPLOAD_H
