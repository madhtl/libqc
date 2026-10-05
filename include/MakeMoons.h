#ifndef LIBQC_MAKEMOONS_H
#define LIBQC_MAKEMOONS_H

#include <vector>

struct Sample {
    double x1, x2;
    int label;
};

std::vector<Sample> make_moons(int n_samples, double noise, unsigned seed = 42);

#endif //LIBQC_MAKEMOONS_H
