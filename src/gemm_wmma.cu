#include <cuda_runtime.h>
#include <mma.h>

#include "bench.hpp"

namespace kl {

using namespace nvcuda;

// Tensor Core GEMM using the WMMA API.
//
// Tensor Cores execute a small matrix multiply-accumulate as a single
// instruction. The 16x16x16 fragment shape here is not a tuning parameter — it
// is the hardware's native tile, and the API only accepts shapes the hardware
// implements.
//
// The critical numerical decision: inputs are stored as fp16, but the
// accumulator is fp32. Products are computed at reduced precision; the running
// sum is not. That combination is what makes this acceptable for neural network
// workloads, where activations tolerate fp16 range but a long fp16 accumulation
// would drift badly.
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void gemm_wmma_kernel(const __half* __restrict__ A,
                                 const __half* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    // One warp computes one 16x16 output tile. warpIdx, not threadIdx: the
    // fragment operations are warp-wide collectives, and every thread in the
    // warp must reach them together.
    const int warp_m = (blockIdx.y * blockDim.y + threadIdx.y);
    const int warp_n = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                   wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                   wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Accumulator starts at zero and stays in fp32 for the whole K loop.
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        const int a_row = warp_m * WMMA_M;
        const int b_col = warp_n * WMMA_N;

        if (a_row < M && b_col < N && k < K) {
            wmma::load_matrix_sync(a_frag, A + a_row * K + k, K);
            wmma::load_matrix_sync(b_frag, B + k * N + b_col, N);
            // One instruction: a 16x16x16 multiply-accumulate.
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    const int c_row = warp_m * WMMA_M;
    const int c_col = warp_n * WMMA_N;
    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(C + c_row * N + c_col, acc_frag, N,
                                wmma::mem_row_major);
    }
}

} // namespace kl
