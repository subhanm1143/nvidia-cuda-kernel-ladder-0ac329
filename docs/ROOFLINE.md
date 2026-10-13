# Where each variant sits, and whether to keep going

## The roofline

Two hardware ceilings bound any kernel:

    memory ceiling  = arithmetic_intensity x peak_bandwidth
    compute ceiling = peak_flops

Achievable performance is the lower of the two. The ridge point — where they
cross — is at intensity = peak_flops / peak_bandwidth. For this GPU:

    peak fp32 compute : ~19.5 TFLOPS
    peak fp16 tensor  : ~78 TFLOPS
    peak bandwidth    : ~3.0 TB/s
    ridge point (fp32): 19.5 / 3.0 = 6.5 flops/byte

Below 6.5 flops/byte you are memory bound no matter what. Above it, compute
bound.

## Every variant placed

| variant | intensity | ceiling | achieved | % of ceiling | verdict |
|---|---|---|---|---|---|
| naive | 0.25 | 0.75 TFLOPS (mem) | 0.33 | 44% | wrong algorithm |
| coalesced | 0.25 | 0.75 TFLOPS (mem) | 1.43 | — see note | measurement artifact |
| tiled_32 | 16.0 | 19.5 TFLOPS (compute) | 5.70 | 29% | real headroom |
| tensor_core | 16.0 | 78 TFLOPS (compute) | 17.62 | 23% | real headroom |
| cuBLAS | 16.0+ | 19.5 TFLOPS (compute) | 19.92 | 102% | at the fp32 limit |

Note on `coalesced` exceeding its computed ceiling: the 0.25 intensity figure
assumes zero cache reuse. Once access is coalesced, L2 serves a meaningful
fraction of the repeated reads, so effective DRAM traffic is lower than the
naive model predicts. The roofline is a bound on a model, and the model here
under-counts caching. Worth knowing before you report a number above 100%.

## Should you keep optimizing?

`tensor_core` sits at 23% of its ceiling, so there is real headroom. Closing it
would require:

- multi-stage pipelining with `cp.async` to overlap global loads with math
- register-level tiling so each warp computes multiple output fragments
- shape-specialized dispatch, which is much of what cuBLAS does

That is roughly two more weeks and lands you near a library you can call today.

## The decision

**Stop here.** The purpose was to understand the gap and be able to close it
where a library does not exist — a fused or unusual layer with no cuBLAS
equivalent. For a plain GEMM, call cuBLAS. Knowing when the library wins is
part of the skill, not an admission of failure.
