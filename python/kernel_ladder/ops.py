"""Bind the CUDA kernel as a PyTorch op and measure it inside a real model.

A microbenchmark speedup is a claim about a kernel. A model-level speedup is a
claim about a product. They are different numbers and the second one is always
smaller.
"""
from __future__ import annotations

import time
import torch
from torch.utils.cpp_extension import load

_ext = load(
    name="kernel_ladder",
    sources=[
        "src/binding.cpp",
        "src/gemm_tiled.cu",
        "src/gemm_wmma.cu",
    ],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_80"],
    verbose=False,
)


class LadderLinear(torch.nn.Module):
    """A Linear layer whose matmul goes through our kernel."""

    def __init__(self, in_features: int, out_features: int, variant: str = "wmma"):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device="cuda")
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self._variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._variant == "wmma":
            return _ext.gemm_wmma(x, self.weight.t().contiguous())
        if self._variant == "tiled":
            return _ext.gemm_tiled(x, self.weight.t().contiguous())
        return x @ self.weight.t()


def benchmark_model(model: torch.nn.Module, x: torch.Tensor,
                    warmup: int = 20, repeats: int = 100) -> dict:
    """End-to-end timing. Host timer is correct HERE, unlike in the kernel
    harness, because we want wall clock including launch overhead and any
    synchronization the model performs."""
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)

    samples.sort()
    return {
        "p50_ms": samples[len(samples) // 2],
        "p95_ms": samples[int(len(samples) * 0.95)],
        "min_ms": samples[0],
    }
