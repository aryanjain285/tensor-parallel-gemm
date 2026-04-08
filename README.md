# Scaling Matrix Multiplication to Multi-GPU Tensor Parallelism

**SC4064 GPU Programming — Course Project**
Nanyang Technological University

## Overview

This project explores GPU-based matrix multiplication (GEMM) optimization from first principles through to multi-GPU distributed execution. It has two parts:

1. **Single-GPU GEMM Optimization**: Seven progressively optimized CUDA kernels, each building on the previous to approach cuBLAS-level performance. Every kernel is benchmarked and profiled to quantify the impact of each optimization technique.

2. **Multi-GPU Tensor Parallelism**: Column-parallel and row-parallel linear layers using NCCL, forming a complete Megatron-LM-style parallel MLP block. Strong scaling, weak scaling, and communication-vs-computation analysis across multiple A100 GPUs.

## Why This Matters

GEMM is the computational core of deep learning — every linear layer, attention head, and convolution reduces to matrix multiplication. Understanding how to optimize it at the hardware level (memory coalescing, shared memory tiling, register blocking) and scale it across GPUs (tensor parallelism, NCCL collectives) is fundamental to building efficient training systems. This project bridges the gap between "call cuBLAS" and understanding *why* cuBLAS is fast.

## Project Structure

```
tensor-parallel-gemm/
├── Makefile                          # Build system
├── src/
│   ├── kernels/                      # Progressive GEMM optimizations
│   │   ├── kernels.cuh               # Kernel declarations
│   │   ├── 01_naive.cu               # Baseline: 1 thread = 1 output element
│   │   ├── 02_coalesced.cu           # Memory coalescing demonstration
│   │   ├── 03_smem_tiling.cu         # Shared memory tiling
│   │   ├── 04_1d_blocktile.cu        # Thread coarsening along M
│   │   ├── 05_2d_blocktile.cu        # 2D register blocking (TM×TN per thread)
│   │   ├── 06_vectorized.cu          # float4 vectorized loads + transposed smem
│   │   ├── 07_warptile.cu            # Warp-level tiling hierarchy
│   │   └── cublas_ref.cu             # cuBLAS reference wrapper
│   ├── tensor_parallel/
│   │   └── tensor_parallel.cu        # Column/Row parallel + Parallel MLP
│   ├── benchmark/
│   │   ├── bench_single_gpu.cu       # Correctness + GFLOPS benchmark
│   │   └── bench_multi_gpu.cu        # Scaling analysis benchmark
│   └── utils/
│       └── cuda_utils.cuh            # Error checking, timing, verification
├── scripts/
│   ├── run_benchmarks.sh             # Run everything
│   ├── nscc_job.sh                   # NSCC ASPIRE 2A job submission
│   └── plot_results.py               # Generate performance plots
├── results/                          # Benchmark outputs and plots
└── docs/
    └── optimization_notes.md         # Detailed optimization analysis
```

## Kernel Optimization Roadmap

Each kernel introduces one major optimization technique. The table below summarizes the progression:

| Kernel | Technique | Key Idea | Arithmetic Intensity |
|--------|-----------|----------|---------------------|
| 1. Naive | Baseline | 1 thread → 1 element, K global loads per FMA | O(1) FLOP/byte |
| 2. Coalesced | Memory coalescing | threadIdx.x → column for stride-1 access | O(1), fewer transactions |
| 3. Shared Memory | Tiling in SRAM | Load tile to smem, reuse TILE_SIZE times | O(TILE_SIZE) |
| 4. 1D Block Tile | Thread coarsening | Each thread computes TM=8 rows | O(TM × BK) |
| 5. 2D Block Tile | Register blocking | Each thread computes TM×TN=8×8 sub-tile | O(TM × TN) |
| 6. Vectorized | float4 loads | 128-bit memory transactions, transposed smem | Same, fewer instructions |
| 7. Warp Tile | Warp-level hierarchy | Block → Warp → Thread tiling | Same, better scheduling |

### Kernel 1: Naive
Each thread computes one element of C by iterating over the K dimension. No data reuse — every FMA requires two global memory loads. This is entirely memory-bandwidth-bound and typically achieves 1-2% of peak compute.

### Kernel 2: Global Memory Coalescing
Demonstrates the impact of memory access patterns. When threads in a warp access consecutive addresses (coalesced), the hardware combines 32 individual 4-byte requests into a single 128-byte transaction. The "uncoalesced" variant (threadIdx.x → row) can be 5-10x slower.

### Kernel 3: Shared Memory Tiling
The first major optimization. A 32×32 tile of A and B is loaded into on-chip shared memory (~20ns latency, ~19 TB/s bandwidth vs ~400ns, ~2 TB/s for global memory). Each loaded element is reused 32 times, reducing global memory traffic by 32x.

### Kernel 4: 1D Block Tiling
Each thread computes TM=8 output elements along the M dimension. A single load from B's shared memory tile is reused across 8 accumulations, increasing the compute-to-memory ratio. Block dimensions: (64, 8) = 512 threads.

### Kernel 5: 2D Block Tiling (Register Blocking)
The most important optimization. Each thread computes an 8×8 sub-tile of C entirely in registers. The outer product formulation means each shared memory load feeds TM+TN=16 FMAs producing TM×TN=64 results. Block: BM=BN=128, BK=8, 256 threads.

### Kernel 6: Vectorized Memory Access
Uses `float4` (128-bit) loads to maximize memory transaction efficiency. Also stores A transposed in shared memory (`As[k][m]` instead of `As[m][k]`) to eliminate bank conflicts during the compute phase.

### Kernel 7: Warp Tiling
Adds hierarchical tiling: Block tile (128×128) → Warp tile (32×64) → Thread tile (8×8). This ensures threads within a warp access nearby shared memory locations, improving L1 hit rates and enabling better instruction scheduling.

## Tensor Parallelism

### Column Parallelism
The weight matrix W is sharded column-wise across p GPUs. Each GPU computes a slice of the output: `Y_i = X @ W_i`. An AllGather collective assembles the full output. Used for the first linear layer in an MLP block.

### Row Parallelism
The weight matrix W is sharded row-wise. The input X is correspondingly split column-wise. Each GPU computes a partial sum: `Y_i = X_i @ W_i`. An AllReduce sums the partial results. Used for the second linear layer.

### Parallel MLP Block
By combining column-parallel (layer 1) → row-parallel (layer 2), only **one AllReduce** is needed per MLP block in the forward pass. This is exactly the strategy used by Megatron-LM for training large language models.

## Building

### Prerequisites
- CUDA Toolkit 12.x
- cuBLAS (included with CUDA)
- NCCL 2.x (for multi-GPU)
- Python 3 + matplotlib (for plotting)

### Build Commands
```bash
# Build everything (default: sm_80 for A100)
make all

# Build single-GPU benchmark only
make bench_single

# Build multi-GPU benchmark only
make bench_multi

# Target a different GPU architecture
make all GPU_ARCH=sm_90   # H100
make all GPU_ARCH=sm_70   # V100
```

## Running

### Single-GPU Benchmark
```bash
./build/bench_single_gpu
```
This runs correctness verification (against CPU reference at M=N=K=256) followed by GFLOPS benchmarks at sizes 256, 512, 1024, 2048, 4096.

### Multi-GPU Benchmark
```bash
./build/bench_multi_gpu <num_gpus>
```
Runs strong scaling, weak scaling, and communication/computation ratio experiments.

### NSCC Cluster
```bash
qsub scripts/nscc_job.sh
```

### Generate Plots
```bash
python3 scripts/plot_results.py results/single_gpu.txt
```

## Profiling

### Nsight Compute (kernel-level)
```bash
ncu --set full \
    --kernel-name gemm_2d_blocktile \
    --launch-count 1 \
    -o results/ncu_profile \
    ./build/bench_single_gpu
```
Key metrics to examine: achieved occupancy, memory throughput, compute throughput, warp stall reasons, shared memory bank conflicts.

### Nsight Systems (system-level timeline)
```bash
nsys profile -o results/timeline ./build/bench_multi_gpu 4
```
Shows the interleaving of GEMM kernels and NCCL collectives, useful for identifying overlap opportunities.

## Evaluation Plan

1. **Kernel Benchmarking**: GFLOPS across matrix sizes 256–4096 for each optimization stage. Identify the memory-bound → compute-bound transition.

2. **Roofline Analysis**: Plot each kernel on the Roofline model (A100: 19.5 TFLOPS FP32, 2 TB/s HBM bandwidth) to visualize how close each optimization gets to the hardware ceiling.

3. **Strong Scaling**: Fixed total workload (e.g., 4096×4096), increase GPUs from 1→2→4. Measure parallel efficiency = `T_1 / (p × T_p)`.

4. **Weak Scaling**: Fixed per-GPU workload (2048×2048 per GPU), increase GPUs. Ideal: constant time. Measure communication overhead growth.

5. **Bottleneck Identification**: Plot communication time vs. compute time as matrix size grows. Find the "crossover point" where communication dominates.

## References

- Jia et al. (2018). *Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking*. arXiv:1804.06826
- Shoeybi et al. (2019). *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*. arXiv:1909.08053
- Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS 2017
- Williams et al. (2009). *Roofline: An Insightful Visual Performance Model for Multicore Architectures*. CACM 52(4)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)

## Team

- Aryan Jain (ARYAN017@e.ntu.edu.sg)
- Fanyi Pu (FPU001@e.ntu.edu.sg)
- Ze Hong Maxwell Au (MAU002@e.ntu.edu.sg)
