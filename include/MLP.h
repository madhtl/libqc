#ifndef LIBQC_MLP_H
#define LIBQC_MLP_H

#include "MakeMoons.h"
#include <vector>
#include <string>
#include <cmath>

// 2 → 4 → 1, sigmoid, MSE loss, backprop
class MLP {
public:
    MLP();

    double forward(double x1, double x2);
    int    predict(double x1, double x2);
    void   train(const std::vector<Sample>& data, int epochs, double lr,
                 const std::vector<Sample>& test_data = {},
                 const std::string& log_file = "");
    double accuracy(const std::vector<Sample>& data);

private:
    static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

    struct FwdCache { std::vector<double> h1; double y; };
    FwdCache forward_cached(double x1, double x2) const;

    // W1: 4×2 (row-major), b1: 4, W2: 4, b2: scalar
    std::vector<double> W1, b1, W2;
    double b2;
};

#endif //LIBQC_MLP_H
