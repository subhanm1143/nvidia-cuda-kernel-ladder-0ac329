# NVIDIA | Kernel Ladder — Beat the Baseline on a Real DL Layer

An advanced GPU systems capstone in the shape of NVIDIA Developer Technology work. You take one deep learning layer, implement it in CUDA C/C++, and drive it through profiler-guided optimization until you can defend a measured speedup against a real library baseline. You build the correctness oracle first, then a benchmark harness that warms up and reports percentiles instead of a single stopwatch number, then the naive kernel, then coalesced access, shared-memory tiling, and a Tensor Core path — each change justified by what the profiler said, not by what sounded fast. You finish by binding the winning kernel as a PyTorch custom op and measuring the model-level gain, plus a roofline read on how close to the hardware limit you actually got. NOTE — running the kernels requires an NVIDIA GPU with the CUDA toolkit; the project teaches the real code and the reasoning, and the harness reports against the CPU reference so the analysis is followable without the hardware.

Built step-by-step with [KhwajaLabs Build](https://khwajalabs.com).

## Stack
- CUDA C/C++
- C++17
- Nsight Compute
- cuBLAS
- PyTorch custom ops
- CMake
