#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "bench.hpp"

namespace kl {

// cuBLAS is the number to beat. It is written by people who do this full time,
// tuned per architecture, and dispatches to different kernels by shape. Beating
// it on a general GEMM is not the goal of this project; UNDERSTANDING the gap
// is.
//
// One trap: cuBLAS is column-major and our matrices are row-major. Rather than
// transpose the data, we exploit the identity
//
//     C^T = B^T * A^T
//
// A row-major matrix reinterpreted as column-major IS its transpose, so passing
// B and A in that order with no transpose flags computes the row-major C we
// want, with zero data movement.
struct CublasGemm {
    cublasHandle_t handle{};

    CublasGemm() { cublasCreate(&handle); }
    ~CublasGemm() { cublasDestroy(handle); }

    void run(const float* dA, const float* dB, float* dC,
             int M, int N, int K) const {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,          // swapped: computing C^T
                    &alpha,
                    dB, N,            // B first
                    dA, K,            // then A
                    &beta,
                    dC, N);
    }
};

} // namespace kl
