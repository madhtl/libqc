#include "SVM.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

KernelSVM::KernelSVM(double C, double lr) : C(C), lr(lr), bias(0.0) {}

void KernelSVM::fit(const std::vector<std::vector<double>>& K,
                    const std::vector<int>& labels,
                    int n_iter) {
    const int n = static_cast<int>(labels.size());
    alpha.assign(n, 0.0);
    train_labels = labels;
    bias = 0.0;

    // Gradient ascent na problemie dualnym:
    // L = Σ αᵢ - ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(i,j)
    // ∂L/∂αᵢ = 1 - yᵢ Σⱼ αⱼ yⱼ K(i,j)
    for (int iter = 0; iter < n_iter; ++iter) {
        for (int i = 0; i < n; ++i) {
            double decision = 0.0;
            for (int j = 0; j < n; ++j)
                decision += alpha[j] * labels[j] * K[i][j];
            double grad = 1.0 - labels[i] * decision;
            alpha[i] = std::max(0.0, std::min(C, alpha[i] + lr * grad));
        }
    }

    // Bias: średnia z support vectors
    double b_sum   = 0.0;
    int    sv_count = 0;
    for (int i = 0; i < n; ++i) {
        if (alpha[i] > 1e-5) {
            double decision = 0.0;
            for (int j = 0; j < n; ++j)
                decision += alpha[j] * labels[j] * K[i][j];
            b_sum += labels[i] - decision;
            ++sv_count;
        }
    }
    bias = sv_count > 0 ? b_sum / sv_count : 0.0;
}

std::vector<int> KernelSVM::predict(
    const std::vector<std::vector<double>>& K_test) const
{
    const int n_test  = static_cast<int>(K_test.size());
    const int n_train = static_cast<int>(alpha.size());
    std::vector<int> preds(n_test);
    for (int i = 0; i < n_test; ++i) {
        double score = bias;
        for (int j = 0; j < n_train; ++j)
            score += alpha[j] * train_labels[j] * K_test[i][j];
        preds[i] = score > 0.0 ? 1 : -1;
    }
    return preds;
}

double KernelSVM::accuracy(const std::vector<int>& pred,
                            const std::vector<int>& true_labels) const {
    if (pred.size() != true_labels.size())
        throw std::invalid_argument("Size mismatch in accuracy()");
    int correct = 0;
    for (size_t i = 0; i < pred.size(); ++i)
        if (pred[i] == true_labels[i]) ++correct;
    return static_cast<double>(correct) / static_cast<double>(pred.size());
}

int KernelSVM::num_support_vectors() const {
    int cnt = 0;
    for (double a : alpha) if (a > 1e-5) ++cnt;
    return cnt;
}
