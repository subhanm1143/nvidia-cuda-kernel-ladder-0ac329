#include <cuda_runtime.h>

#include "bench.hpp"

namespace kl {

constexpr int TILE = 32;

// Shared-memory tiled GEMM.
//
// The naive kernel re-read each element of A N times and each element of B M
// times. This kernel loads a TILE x TILE block of each into shared memory once,
// then every thread in the block reuses it TILE times.
//
// Arithmetic intensity goes from 0.25 flops/byte to roughly TILE/2 = 16, a
// 64x reduction in global traffic for the same arithmetic.
__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    // +1 padding on the inner dimension avoids shared-memory bank conflicts.
    // Shared memory has 32 banks of 4 bytes. With a stride of exactly 32
    // floats, every thread in a warp reading down a column hits the SAME bank,
    // serializing 32 ways. Padding to 33 shifts each row by one bank so a
    // column walk spreads across all banks.
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    const int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE + tx;
        const int a_row = row;
        const int b_col = col;
        const int b_row = t * TILE + ty;

        // Cooperative load. Out-of-range elements load as zero, which
        // contributes nothing to the dot product — cleaner than branching
        // inside the inner loop.
        As[ty][tx] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;

        // Barrier 1: every thread must finish WRITING the tile before any
        // thread starts READING it.
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        // Barrier 2: every thread must finish READING this tile before any
        // thread OVERWRITES it on the next iteration. Omitting this is a
        // classic race that produces correct results on small inputs and
        // garbage at scale.
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

void launch_gemm_tiled(const float* dA, const float* dB, float* dC,
                       int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_tiled_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kl
