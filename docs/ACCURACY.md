# The accuracy trade-off, measured

Switching to Tensor Cores changes the numerics. That is a decision to make with
data, not a free speedup.

## Measured on 4096x4096x4096, inputs uniform in [-1, 1)

| variant | p50 ms | TFLOPS | max_rel_err vs fp64 reference |
|---|---|---|---|
| tiled_32 (fp32) | 24.1 | 5.70 | 3.1e-06 |
| tensor_core (fp16 in, fp32 acc) | 7.8 | 17.62 | 9.4e-04 |
| cublas SGEMM (fp32) | 6.9 | 19.92 | 3.0e-06 |

Roughly 300x more relative error, for roughly 3x the speed.

## When that trade is acceptable

**Acceptable** — neural network forward and backward passes. Activations and
gradients have limited dynamic range, training is stochastic anyway, and
mixed-precision training with fp32 accumulation is the industry default
precisely because this error level does not affect convergence.

**Not acceptable** — anything where the result is compared for equality,
accumulated over very long chains, or used in an iterative solver whose
convergence depends on residual precision. Scientific computing and financial
calculation generally sit here.

## Why fp32 accumulation is doing the heavy lifting

Storing inputs in fp16 costs precision once per element. Accumulating in fp16
would cost precision on every one of K additions, and the error compounds. At
K = 4096 an fp16 accumulator produces relative errors around 1e-2 — ten times
worse than what we measured, and enough to affect training.

The measured 9.4e-04 is almost entirely input quantization, not accumulation.
That is why this configuration is the standard one.
