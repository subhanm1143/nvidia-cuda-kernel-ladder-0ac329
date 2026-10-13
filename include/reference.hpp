#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace kl {

// Row-major matrix view. Deliberately dumb: this is the oracle, and the oracle
// must be obviously correct rather than fast.
struct Matrix {
    std::vector<float> data;
    std::size_t rows = 0;
    std::size_t cols = 0;

    float& at(std::size_t r, std::size_t c) { return data[r * cols + c]; }
    float at(std::size_t r, std::size_t c) const { return data[r * cols + c]; }
};

inline Matrix make_matrix(std::size_t rows, std::size_t cols) {
    return Matrix{std::vector<float>(rows * cols, 0.0f), rows, cols};
}

// Deterministic fill. A fixed seed means a failing comparison is reproducible;
// random data that changes every run turns a real bug into a flaky one.
inline void fill_deterministic(Matrix& m, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    // Values in [-1, 1). Bounded so accumulated error stays interpretable.
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : m.data) v = dist(rng);
}

// The reference GEMM: C = A * B. Triple loop, no blocking, no intrinsics.
// This is intentionally the slowest correct implementation we can write.
inline void gemm_reference(const Matrix& a, const Matrix& b, Matrix& c) {
    for (std::size_t i = 0; i < a.rows; ++i) {
        for (std::size_t j = 0; j < b.cols; ++j) {
            // Accumulate in double. The GPU will accumulate in float, and we
            // want the reference to be MORE accurate than the thing it judges,
            // never less.
            double acc = 0.0;
            for (std::size_t k = 0; k < a.cols; ++k) {
                acc += static_cast<double>(a.at(i, k)) *
                       static_cast<double>(b.at(k, j));
            }
            c.at(i, j) = static_cast<float>(acc);
        }
    }
}

// Comparison result. We report the worst case AND the distribution, because a
// single max-error number cannot distinguish "one bad element" from "every
// element slightly off" — and those have completely different causes.
struct Comparison {
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    double mean_abs_err = 0.0;
    std::size_t mismatches = 0;
    std::size_t total = 0;

    bool within(double rel_tol) const { return max_rel_err <= rel_tol; }
};

inline Comparison compare(const Matrix& got, const Matrix& want, double rel_tol) {
    Comparison out;
    out.total = want.data.size();
    double sum_abs = 0.0;

    for (std::size_t i = 0; i < want.data.size(); ++i) {
        const double g = got.data[i];
        const double w = want.data[i];
        const double abs_err = std::fabs(g - w);
        // Relative to the magnitude of the expected value, with a floor so a
        // near-zero expectation does not produce an infinite relative error.
        const double denom = std::max(std::fabs(w), 1e-6);
        const double rel_err = abs_err / denom;

        sum_abs += abs_err;
        if (abs_err > out.max_abs_err) out.max_abs_err = abs_err;
        if (rel_err > out.max_rel_err) out.max_rel_err = rel_err;
        if (rel_err > rel_tol) ++out.mismatches;
    }
    out.mean_abs_err = sum_abs / static_cast<double>(out.total);
    return out;
}

} // namespace kl
