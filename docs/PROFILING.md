# Reading the profiler

Run the profiler on one kernel at a time, with a small number of replays:

```bash
ncu --set full \
    --kernel-name gemm_naive_kernel \
    --launch-count 1 \
    -o profiles/naive \
    ./build/bench --layer gemm --shape 4096x4096x4096 --repeats 1
```

## The four counters that answer "what is limiting this kernel"

| Counter | What it means | Reading for the naive kernel |
|---|---|---|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | Fraction of peak memory bandwidth achieved | 94% — saturated |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Fraction of peak compute achieved | 3% — idle |
| `smsp__average_warps_issue_stalled_long_scoreboard` | Warps waiting on global memory | dominant stall reason |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | Global load sectors requested | ~32x the ideal count |

## How to read them together

Memory near peak AND compute near zero means **memory bound**. The arithmetic
units are idle waiting for data. Adding more math, using Tensor Cores, or
tuning the instruction mix cannot help — none of those touch the constraint.

Compute near peak AND memory low means **compute bound**. Now the arithmetic
pipeline is the limit and precision or instruction choice becomes the lever.

Both low means **latency bound**: not enough parallelism in flight to hide
memory latency. The fix is occupancy — more warps resident per SM — not a
better access pattern.

## Arithmetic intensity: the number that predicts the ceiling

    intensity = flops performed / bytes moved from global memory

For the naive GEMM: 2*M*N*K flops over 2*M*N*K*4 bytes = **0.25 flops/byte**.

At roughly 3 TB/s of memory bandwidth, 0.25 flops/byte caps you at about
0.75 TFLOPS no matter how good your arithmetic is. The measured 0.33 TFLOPS is
in that neighborhood, which confirms the diagnosis: this kernel is limited by
how much data it moves, and the only fix is to move less.

Tiling raises intensity by reusing each loaded element across a whole tile
instead of once. That is why it is the next step, and why nothing else would
have worked first.
