# Optimization Notes — Detailed Per-Kernel Analysis

## GPU Memory Hierarchy (NVIDIA A100)

| Level | Size | Bandwidth | Latency | Scope |
|-------|------|-----------|---------|-------|
| Registers | 256 KB/SM | ~20 TB/s | 0 cycles | Per-thread |
| Shared Memory | 164 KB/SM | ~19 TB/s | ~20 ns | Per-block |
| L1 Cache | 192 KB/SM | ~12 TB/s | ~30 ns | Per-SM |
| L2 Cache | 40 MB | ~5 TB/s | ~200 ns | Global |
| HBM2e (Global) | 80 GB | ~2 TB/s | ~400 ns | Global |

Every optimization moves data access from a slower to a faster level, or reduces total accesses.

## A100 Compute Specs
- 108 SMs, 64 FP32 cores/SM → 19.5 TFLOPS FP32
- Warp size: 32, Max threads/SM: 2048, Max threads/block: 1024
- Shared memory: up to 164 KB/SM (configurable)

---

## Kernel 1 → 2: Memory Coalescing

**Problem**: Threads in a warp accessing non-consecutive addresses → separate 32B transactions each. 32 threads × 32B = 1024B for 128B useful data (12.5% utilization).

**Fix**: Map threadIdx.x to columns (contiguous in row-major). One 128B transaction serves 32 threads (100% utilization).

**Metric**: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` — lower is better.

---

## Kernel 2 → 3: Shared Memory Tiling

**Problem**: 2K global reads per output element.

**Fix**: 32×32 tile in smem. Each element loaded once, read 32 times. Global traffic reduced 32×.

**Arithmetic intensity**: Before: 0.25 FLOP/byte → After: 8 FLOP/byte (A100 ridge: ~9.75).

---

## Kernel 3 → 4: 1D Block Tiling

**Problem**: 2 smem reads per FMA → smem bandwidth becomes bottleneck.

**Fix**: Thread computes TM=8 rows. B smem value reused across 8 A values. Smem reads/FMA: 2 → ~1.125.

---

## Kernel 4 → 5: 2D Block Tiling

**Problem**: Only B reused across rows, A still loaded once per use.

**Fix**: TM×TN=8×8 outer product. Smem reads/FMA: (8+8)/64 = 0.25 (8× better than kernel 3).

**Registers**: 64 accumulators + 16 cache = 80 regs/thread. A100 has 65536/SM, 256 threads → 256 available. Comfortable.

---

## Kernel 5 → 6: Vectorized + Transposed Smem

**float4**: 4× fewer memory instructions via LDG.128.

**Transpose A in smem**: Without: `As[row][k]` — warp threads access same k column → same bank → 32-way conflict. With: `As[k][row]` — different rows → different banks → no conflict.

---

## Kernel 6 → 7: Warp Tiling

Organize warps into 2D sub-tiles. Each warp's smem working set is WM×BK + BK×WN instead of full block tile. Better L1 locality and instruction scheduling.

---

## Communication Analysis

### AllReduce Cost (Ring)
Time ≈ 2(p-1)/p × S/B + 2(p-1) × L  
A100 NVLink: ~600 GB/s bidirectional, ~1-5 μs latency/hop.

### Crossover Point (A100 NVLink, 2 GPUs)
- 2048²: GEMM ~0.9ms, AllReduce ~0.03ms → compute dominates
- 512²: GEMM ~0.01ms, AllReduce ~0.01ms → balanced
- 128²: AllReduce dominates

This is why tensor parallelism works for large models but has diminishing returns for small ops.
