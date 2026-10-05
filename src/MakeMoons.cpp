#include "MakeMoons.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <numbers>

std::vector<Sample> make_moons(int n_samples, double noise, unsigned seed) {
    const int half = n_samples / 2;
    std::vector<Sample> data;
    data.reserve(n_samples);

    for (int i = 0; i < half; ++i) {
        double a = std::numbers::pi * i / half;
        data.push_back({std::cos(a), std::sin(a), 0});
    }
    for (int i = 0; i < half; ++i) {
        double a = std::numbers::pi * i / half;
        data.push_back({1.0 - std::cos(a), -std::sin(a) + 0.5, 1});
    }

    std::mt19937 rng(seed);
    if (noise > 0.0) {
        std::normal_distribution<double> nd(0.0, noise);
        for (auto& s : data) { s.x1 += nd(rng); s.x2 += nd(rng); }
    }

    double mn1 = data[0].x1, mx1 = data[0].x1;
    double mn2 = data[0].x2, mx2 = data[0].x2;
    for (const auto& s : data) {
        mn1 = std::min(mn1, s.x1); mx1 = std::max(mx1, s.x1);
        mn2 = std::min(mn2, s.x2); mx2 = std::max(mx2, s.x2);
    }
    for (auto& s : data) {
        s.x1 = 2.0 * (s.x1 - mn1) / (mx1 - mn1) - 1.0;
        s.x2 = 2.0 * (s.x2 - mn2) / (mx2 - mn2) - 1.0;
    }

    std::shuffle(data.begin(), data.end(), rng);
    return data;
}
