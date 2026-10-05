#ifndef LIBQC_SVM_H
#define LIBQC_SVM_H

#include <vector>

// Kernel SVM trenowany gradient ascentem na problemie dualnym.
// Nie jest to pełny SMO — uproszczona wersja do porównania z QNN/MLP.
class KernelSVM {
public:
    explicit KernelSVM(double C = 1.0, double lr = 1e-2);

    // K: macierz kernela n×n, labels: {-1, +1}
    void fit(const std::vector<std::vector<double>>& K,
             const std::vector<int>& labels,
             int n_iter = 1000);

    // K_test: macierz n_test × n_train
    std::vector<int> predict(const std::vector<std::vector<double>>& K_test) const;

    double accuracy(const std::vector<int>& pred,
                    const std::vector<int>& true_labels) const;

    int num_support_vectors() const;

private:
    double C, lr;
    std::vector<double> alpha;
    std::vector<int>    train_labels;
    double bias;
};

#endif //LIBQC_SVM_H
