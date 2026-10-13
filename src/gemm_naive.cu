#include <cuda_runtime.h>

#include "bench.hpp"

namespace kl {

// The simplest correct mapping: one thread computes one element of C.
//
// Thread (x, y) in the grid owns C[row][col]. It walks the full K dimension,
// reading one element of A and one of B per iteration, and accumulates.
//
// This is correct. It is also close to the worst possible memory behavior, for
// reasons the profiler will make explicit in step five.
__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Guard: the grid is rounded up to whole blocks, so the last block in each
    // dimension contains threads with no work. Without this they would write
    // outside C.
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        // A is read along a row: A[row][k]. Consecutive k values are adjacent
        // in memory, but consecutive THREADS (varying col) all read the SAME
        // element here — a broadcast, which is fine.
        //
        // B is read down a column: B[k][col]. Consecutive threads read
        // adjacent columns, which IS coalesced. But every thread re-reads the
        // whole column across iterations, and nothing is cached between
        // threads, so total global traffic is enormous.
        acc += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = acc;
}

void launch_gemm_naive(const float* dA, const float* dB, float* dC,
                       int M, int N, int K) {
    // 16x16 = 256 threads per block. A reasonable default: enough threads to
    // hide some latency, small enough that many blocks fit per SM.
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);
    gemm_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kl
