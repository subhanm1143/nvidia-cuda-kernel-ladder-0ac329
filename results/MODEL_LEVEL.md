# Kernel speedup vs model speedup

## The kernel, in isolation

| variant | p50 ms | speedup vs tiled |
|---|---|---|
| tiled_32 | 24.1 | 1.00x |
| tensor_core | 7.8 | 3.09x |

## The same kernel inside a 12-layer transformer block, batch 32, seq 512

| variant | p50 ms | speedup vs tiled |
|---|---|---|
| tiled_32 | 41.7 | 1.00x |
| tensor_core | 29.4 | 1.42x |

## Why 3.09x became 1.42x

The GEMM is not the whole forward pass. Profiling the model shows the time
split roughly as:

- 68% GEMM (what we optimized)
- 14% softmax and attention masking
- 11% layer norm
-  7% elementwise activations and residual adds

Amdahl's law with p = 0.68 and a 3.09x speedup on that portion:

    1 / ((1 - 0.68) + 0.68/3.09) = 1.82x theoretical

We measured 1.42x, short of the theoretical bound because the fp16 path adds
conversion overhead at the layer boundary that the microbenchmark never paid.

## The honest conclusion

A 3x kernel is worth having. It is not a 3x product. Anyone reporting the
microbenchmark number as the model number is either inexperienced or selling
something, and a customer who deploys on that promise will be unhappy.
