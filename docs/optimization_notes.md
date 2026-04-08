# Optimization Analysis Notes

## A100 Hardware Parameters

| Parameter | Value |
|-----------|-------|
| SMs | 108 |
| FP32 cores/SM | 64 |
| Clock | 1.41 GHz |
| Peak FP32 | 19.5 TFLOPS |
| HBM2e bandwidth | 2.0 TB/s |
| L2 cache | 40 MB |
| Shared memory/SM | 164 KB (configurable) |
| Registers/SM | 65536 (32-bit) |
| Warp size | 32 |
| Max threads/SM | 2048 |
| Max threads/block | 1024 |

## Roofline Analysis

The ridge point (where memory-bound meets compute-bound) is:

```
Ridge OI = Peak FLOPS / Peak BW = 19500 GFLOPS / 2000 GB/s = 9.75 FLOP/byte
```

For GEMM with M=N=K=4096:
- Total FLOPs: 2 × 4096³ = 137.4 GFLOP
- Total data moved (naive): (M×K + K×N + M×N) × 4B = 192 MB
- Operational intensity (naive): 137.4G / 192M ≈ 716 FLOP/byte → compute-bound

This means for large matrices, GEMM should be compute-bound, and kernel optimization
should focus on maximizing FLOP throughput (occupancy, ILP, avoiding stalls).

For small matrices (e.g., 256×256), the OI is lower due to:
- Insufficient parallelism to saturate all SMs
- Higher relative overhead of kernel launch and synchronization
- Cache effects (data may fit in L2)

## Shared Memory Bank Conflicts

Shared memory has 32 banks, each 4 bytes wide. If multiple threads in a warp
access the same bank (but different addresses), the accesses are serialized.

In kernel 3 (smem tiling), `Bs[k][threadIdx.x]` accesses column-wise — each
thread hits a different bank → no conflict. `As[threadIdx.y][k]` has all threads
in a warp-row reading the same address → broadcast (free, not a conflict).

In kernel 6, we transpose A in shared memory: `As[k][m]`. When the compute
phase reads `As[k][thread_row * TM + m]`, different values of m within one
thread access consecutive banks → no conflicts. Across threads in a warp
reading different `thread_col` positions from Bs, there are also no conflicts.

## Register Pressure Analysis

Kernel 5 (2D block tiling, TM=TN=8):
- 64 accumulator registers (float)
- 8 a_cache + 8 b_cache = 16 temporary registers
- ~10-15 address computation registers
- Total: ~90-95 registers per thread

A100 has 65536 registers per SM, 256 threads per block:
- 65536 / 256 = 256 registers available per thread → plenty of headroom
- Occupancy limited by shared memory, not registers

Shared memory per block: BM×BK + BK×BN = 128×8 + 8×128 = 2048 floats = 8 KB
A100 has 164 KB/SM → up to 20 blocks per SM (but limited by threads: 2048/256=8)

## Communication Analysis (Tensor Parallelism)

For row-parallel AllReduce with p GPUs, M×N floats:
- Data volume per GPU: M×N×4 bytes
- AllReduce with ring algorithm: 2×(p-1)/p × M×N×4 bytes total bandwidth used
- On NVLink (A100 SXM4): 600 GB/s bidirectional per GPU pair
- For M=N=4096, p=4: 4096² × 4 = 64 MB
  - Ring AllReduce time ≈ 2 × 3/4 × 64 MB / 600 GB/s ≈ 0.16 ms
  - GEMM time (per GPU, N/p=1024): ~0.5 ms at ~18 TFLOPS
  - Comm/compute ratio ≈ 0.32 → communication is NOT dominant

For smaller matrices, the ratio increases because:
- GEMM time drops cubically with size
- Communication has fixed latency overhead (~5-10 μs)
- AllReduce data volume drops quadratically

The "crossover point" where communication dominates shifts to smaller matrices
as kernel efficiency improves — this is a key finding of the project.
