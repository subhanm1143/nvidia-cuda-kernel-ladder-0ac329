#include <cstdio>
#include <cuda_runtime.h>

#include "bench.hpp"
#include "reference.hpp"

namespace kl {

void launch_gemm_naive(const float*, const float*, float*, int, int, int);

struct DeviceMatrices {
    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    DeviceMatrices(const Matrix& a, const Matrix& b, std::size_t c_elems) {
        CUDA_CHECK(cudaMalloc(&dA, a.data.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, b.data.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, c_elems * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, a.data.data(), a.data.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, b.data.data(), b.data.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    ~DeviceMatrices() {
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }
};

// Validate on a small shape where the CPU reference is affordable, then
// benchmark on the large shape. Correctness and performance measured at
// different sizes is normal; skipping either is not.
bool validate(int m, int n, int k, double rel_tol = 1e-3) {
    Matrix a = make_matrix(m, k);
    Matrix b = make_matrix(k, n);
    Matrix want = make_matrix(m, n);
    fill_deterministic(a, 42);
    fill_deterministic(b, 1337);
    gemm_reference(a, b, want);

    DeviceMatrices dev(a, b, want.data.size());
    launch_gemm_naive(dev.dA, dev.dB, dev.dC, m, n, k);
    CUDA_CHECK(cudaDeviceSynchronize());

    Matrix got = make_matrix(m, n);
    CUDA_CHECK(cudaMemcpy(got.data.data(), dev.dC,
                          got.data.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const Comparison cmp = compare(got, want, rel_tol);
    std::printf("validate %dx%dx%d: max_rel=%.3e mean_abs=%.3e mismatches=%zu/%zu\n",
                m, n, k, cmp.max_rel_err, cmp.mean_abs_err,
                cmp.mismatches, cmp.total);
    return cmp.within(rel_tol);
}

} // namespace kl
