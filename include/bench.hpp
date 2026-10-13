#pragma once
#include <algorithm>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace kl {

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t _err = (expr);                                            \
        if (_err != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                         cudaGetErrorString(_err), __FILE__, __LINE__);       \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

struct Timing {
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double min_ms = 0.0;
    int samples = 0;

    // Effective throughput for a GEMM of the given dimensions.
    double tflops(std::size_t m, std::size_t n, std::size_t k) const {
        // 2 flops per multiply-add, m*n*k multiply-adds.
        const double flops = 2.0 * double(m) * double(n) * double(k);
        return flops / (p50_ms * 1e-3) / 1e12;
    }
};

// Time a device function with CUDA events. Host-side timers measure kernel
// LAUNCH, which is asynchronous and returns almost immediately — timing a
// launch tells you nothing about how long the kernel ran.
inline Timing time_kernel(const std::function<void()>& launch,
                          int warmup = 20,
                          int repeats = 200) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup. The first launches pay for context setup, JIT of the module, and
    // caches that are cold. The GPU also boosts its clocks under sustained
    // load, so early samples run at a lower frequency than steady state.
    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> samples;
    samples.reserve(repeats);

    for (int i = 0; i < repeats; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(samples.begin(), samples.end());
    auto pct = [&](double p) {
        const std::size_t idx = std::min<std::size_t>(
            samples.size() - 1,
            static_cast<std::size_t>(p * (samples.size() - 1)));
        return static_cast<double>(samples[idx]);
    };

    Timing t;
    t.p50_ms = pct(0.50);
    t.p95_ms = pct(0.95);
    t.p99_ms = pct(0.99);
    t.min_ms = samples.front();
    t.samples = static_cast<int>(samples.size());
    return t;
}

} // namespace kl
