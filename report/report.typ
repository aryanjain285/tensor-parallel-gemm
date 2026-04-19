// ═══════════════════════════════════════════════════════════════════════
// Typst Report — Scaling Matrix Multiplication
// ═══════════════════════════════════════════════════════════════════════

// ── Accent palette ──────────────────────────────────────────────────
#let accent = rgb("#1a56db")
#let accent-light = rgb("#eff6ff")

#set document(
  title: "Scaling Matrix Multiplication: From CUDA Kernels to Multi-GPU Tensor Parallelism",
  author: ("Aryan Jain", "Fanyi Pu", "Ze Hong Maxwell Au"),
)

#set page(
  paper: "a4",
  columns: 2,
  margin: (top: 2.4cm, bottom: 2cm, x: 1.6cm),
  numbering: "1",
  number-align: center,
  header: context {
    if counter(page).get().first() > 1 [
      #set text(7.5pt, fill: gray.darken(30%))
      #smallcaps[Scaling Matrix Multiplication] #h(1fr) SC4064 GPU Programming
      #v(-0.5em)
      #line(length: 100%, stroke: 0.4pt + gray.lighten(40%))
    ]
  },
)

#set text(font: "New Computer Modern", size: 9.5pt)
#set par(justify: true, leading: 0.52em)
#set heading(numbering: "1.1")

// ── Heading styles ──────────────────────────────────────────────────
#show heading.where(level: 1): it => {
  set text(12pt, weight: "bold")
  block(above: 1.4em, below: 0.7em)[
    #it
    #v(-0.5em)
    #line(length: 100%, stroke: 0.6pt + accent)
  ]
}
#show heading.where(level: 2): it => {
  set text(10pt, weight: "bold")
  block(above: 1.1em, below: 0.5em, it)
}
#show heading.where(level: 3): it => {
  set text(9.5pt, weight: "bold", style: "italic")
  block(above: 0.9em, below: 0.4em, it)
}

#set math.equation(numbering: "(1)")

// ── Figure / table styling ──────────────────────────────────────────
#show figure.where(kind: table): set figure(scope: "parent", placement: auto)
#show figure: set block(above: 1em, below: 1em)
#show figure.caption: set text(size: 8.5pt)

// ── Code block styling ──────────────────────────────────────────────
#show raw.where(block: true): set block(
  width: 100%,
  inset: 8pt,
  radius: 2pt,
  fill: luma(248),
  stroke: 0.5pt + luma(220),
)

// ── Helper ──────────────────────────────────────────────────────────
#let fig(path, caption, width: 95%, scope: "column") = {
  figure(
    image(path, width: width),
    caption: caption,
    scope: scope,
    placement: auto,
  )
}

// ═════════════════════════════════════════════════════════════════════
// Title (full-width across both columns)
// ═════════════════════════════════════════════════════════════════════
#place(top, scope: "parent", float: true, clearance: 0.8em)[
  #align(center)[
    #block(above: 0.5em, below: 0.3em)[
      #text(16pt, weight: "bold")[Scaling Matrix Multiplication:\ From CUDA Kernels to Multi-GPU Tensor Parallelism]
    ]
    #text(11pt)[
      Aryan Jain#super[\*] #h(1.5em) Fanyi Pu#super[\*] #h(1.5em) Ze Hong Maxwell Au#super[\*]
    ] \
    #text(8.5pt, fill: gray.darken(30%))[
      School of Computer Science and Engineering, Nanyang Technological University \
      #super[\*]Equal contribution. Authors listed in alphabetical order.
    ]
    #v(0.1em)
    #text(8.5pt)[SC4064 GPU Programming — Course Project Report]
  ]
  #v(0.5em)
  #block(
    width: 100%,
    inset: (x: 1.5em, y: 1em),
    radius: 2pt,
    fill: accent-light,
    stroke: 0.4pt + accent.lighten(50%),
  )[
    #text(weight: "bold")[Abstract.]
    General Matrix Multiplication (GEMM) is the computational backbone of modern deep learning.
    This report presents a systematic study across three scales: (i) progressive single-GPU CUDA kernel optimization through seven stages---from naive global memory access to warp-level tiling---benchmarked against cuBLAS on NVIDIA H100 GPUs; (ii) intra-node tensor parallelism on 8$times$H100 connected via NVLink; and (iii) cross-node tensor parallelism on 16$times$H100 spanning two nodes over 400 Gb/s InfiniBand.
    Our best custom kernel reaches 32.8 TFLOPS, 63% of cuBLAS throughput.
    Strong scaling on 8$times$H100 for $N = 16384$ yields a $7.03times$ speedup (88% efficiency) at 360 TFLOPS aggregate; scaling to 16 GPUs across the IB fabric pushes the $N=32768$ workload to 667 TFLOPS aggregate.
    As local GEMM kernels approach peak efficiency, the communication-to-computation ratio rises from 0.22 (naive) to 0.88 (cuBLAS), quantifying the crossover at which inter-GPU communication becomes the dominant bottleneck. A transport sweep confirms that NCCL's IB path is $~125times$ faster than the TCP fallback at the largest matrix size.
  ]
]

// ═════════════════════════════════════════════════════════════════════
= Introduction
// ═════════════════════════════════════════════════════════════════════

GEMM operations underpin virtually all compute-intensive workloads in deep learning @vaswani2017attention @jia2018dissecting. In Transformer architectures, multi-head attention and feed-forward layers are fundamentally matrix multiplications. As models scale beyond the memory capacity of a single accelerator, _tensor parallelism_ @shoeybi2019megatron has become indispensable, distributing weight matrices across GPUs at the cost of inter-GPU communication.

While vendor-tuned libraries such as cuBLAS @nvidia_cublas_2026 deliver near-optimal single-GPU performance, they abstract away the complex interaction between hardware-level compute intensity and system-level communication latency @nvidia_nccl_2026. This project bridges that gap by:
+ Implementing seven progressively optimized CUDA GEMM kernels (plus one deliberately uncoalesced negative control) to understand hardware-level constraints (memory coalescing, shared memory tiling, register blocking, warp scheduling).
+ Building a distributed tensor-parallel linear layer (forward and backward) using NCCL, including a complete parallel MLP block and a communication--computation overlap variant.
+ Quantifying how local kernel efficiency impacts multi-GPU scalability, identifying the _crossover point_ where communication dominates, and extending the study across the node boundary to measure how that crossover shifts when NVLink gives way to InfiniBand.

All experiments are conducted on the hardware described in @sec:setup.

== Experimental Setup <sec:setup>

Single-node experiments run on one node with 8 NVIDIA H100 80 GB SXM5 GPUs interconnected via NVLink 4.0 (NV18 topology, 900 GB/s bidirectional per GPU). Multi-node experiments span two such nodes (16 GPUs total) connected by 4$times$Mellanox ConnectX-7 400 Gb/s InfiniBand HCAs per node with GPUDirect RDMA. Both configurations use CUDA 13.1 and NCCL 2.29.3. The full hardware and software configuration is summarised in @tab:setup.

#figure(
  table(
    columns: (auto, auto),
    inset: 6pt,
    align: (left, left),
    stroke: none,
    table.hline(),
    table.header([*Component*], [*Specification*]),
    table.hline(stroke: 0.5pt),
    [GPU (per node)], [8 $times$ NVIDIA H100 80 GB HBM3 (SXM5)],
    [Intra-node interconnect], [NVLink 4.0, all-to-all NV18 (900 GB/s bidirectional per GPU)],
    [Inter-node interconnect], [4 $times$ Mellanox ConnectX-7 400 Gb/s InfiniBand per node, GPUDirect RDMA],
    [GPU Compute], [132 SMs, 1980 MHz boost, sm\_90, 67 TFLOPS FP32 dense peak (FMA-counted)],
    [GPU Memory], [80 GB HBM3, $tilde$3.35 TB/s bandwidth per GPU],
    [CPU (per node)], [2 $times$ Intel Xeon Gold 6448Y (32 cores / 64 threads each, 128 threads total)],
    [System Memory (per node)], [2 TB DDR5],
    [OS], [Ubuntu 24.04.4 LTS (kernel 5.14.0)],
    [CUDA Toolkit], [13.1],
    [NCCL], [2.29.3],
    [GPU Driver], [550.90.07],
    [Max world size measured], [16 GPUs (2 nodes $times$ 8 GPUs)],
    table.hline(),
  ),
  caption: [Hardware and software configuration. Single-node runs use one node with 8 GPUs over NVLink; multi-node runs span two nodes over 400 Gb/s InfiniBand.],
) <tab:setup>

// ═════════════════════════════════════════════════════════════════════
= System Design
// ═════════════════════════════════════════════════════════════════════

The codebase is organized into three loosely-coupled layers: a *kernel abstraction layer* with pluggable GEMM implementations, a *CUDA resource layer* with RAII wrappers, and the *tensor parallelism layer* that composes both.

== Kernel Abstraction and Registry

All GEMM kernels---from the naive baseline to cuBLAS---implement a common abstract interface:

#figure(
  kind: image,
  scope: "parent",
  placement: auto,
  block(width: 75%, inset: 6pt)[
    #set text(size: 8pt)

    #align(center)[
      #block(width: 50%, inset: 6pt, radius: 3pt, stroke: rgb("#3b82f6") + 1.5pt, fill: rgb("#3b82f6").lighten(92%))[
        #align(center)[
          #text(weight: "bold", size: 9pt)[`GemmKernel`] #text(fill: gray)[ (abstract)] \
          #line(length: 100%, stroke: 0.5pt + gray)
          #v(1pt)
          `+ name() -> const char*` \
          `+ launch(A, B, C, M, N, K, stream)` \
          `+ needs_cublas() -> bool` \
          `+ set_cublas_handle(handle)` \
        ]
      ]
    ]
    #v(2pt)
    #align(center)[
      #grid(columns: (1fr,) * 4, gutter: 4pt,
        ..range(4).map(_ => align(center)[#text(11pt)[#sym.triangle.b]]))
    ]
    #v(1pt)
    #grid(columns: (1fr, 1fr, 1fr, 1fr), gutter: 5pt,
      block(inset: 4pt, radius: 3pt, stroke: rgb("#3b82f6") + 0.8pt, fill: white)[
        #text(weight: "bold")[`NaiveKernel`] \ Block: 32$times$32 \ 1 elem/thread
      ],
      block(inset: 4pt, radius: 3pt, stroke: rgb("#3b82f6") + 0.8pt, fill: white)[
        #text(weight: "bold")[`SmemTiling`] \ Tile: 32$times$32 \ Shared memory
      ],
      block(inset: 4pt, radius: 3pt, stroke: rgb("#3b82f6") + 0.8pt, fill: white)[
        #text(weight: "bold")[`WarpTile`] \ BM$=$BN$=$128 \ Warp-level tiling
      ],
      block(inset: 4pt, radius: 3pt, stroke: rgb("#3b82f6") + 0.8pt, fill: white)[
        #text(weight: "bold")[`CublasKernel`] \ cuBLAS `sgemm` \ Tensor Core path
      ],
    )
    #v(1pt)
    #align(center)[#text(size: 7pt, fill: gray)[+ CoalescedKernel, BlockTile1DKernel, BlockTile2DKernel, VectorizedKernel]]
  ],
  caption: [Kernel class hierarchy. Each concrete kernel encapsulates its CUDA `__global__` function, block/grid configuration, and launch logic. The abstract base provides a uniform `launch()` interface.],
) <fig:class>

Each kernel file self-registers via a static initializer, eliminating the need for centralized dispatch tables:
```cpp
// In naive.cu
class NaiveKernel : public GemmKernel { ... };
namespace { static int reg = KernelRegistry::add(
    std::make_unique<NaiveKernel>()); }
```

The `KernelRegistry` singleton manages all kernel instances. Adding a new kernel requires only creating a single `.cu` file---no header modifications or array synchronization needed. This _open-closed_ design replaces the previous approach of parallel enum/function-pointer arrays that required manual synchronization across four separate data structures.

== RAII Resource Management

All GPU resources are managed via move-only RAII wrappers, ensuring deterministic cleanup and preventing leaks:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 6pt,
    align: (left, left, left, left),
    stroke: none,
    table.hline(),
    table.header([*Class*], [*Manages*], [*Acquire*], [*Release*]),
    table.hline(stroke: 0.5pt),
    [`CudaMemory<T>`], [Device allocation], [`cudaMalloc`], [`cudaFree`],
    [`DeviceMatrix`], [2D float matrix], [Delegates to `CudaMemory`], [Automatic],
    [`CudaStream`], [CUDA stream], [`cudaStreamCreate`], [`cudaStreamDestroy`],
    [`CudaEvent`], [CUDA event], [`cudaEventCreate`], [`cudaEventDestroy`],
    [`CublasHandle`], [cuBLAS context], [`cublasCreate`], [`cublasDestroy`],
    table.hline(),
  ),
  caption: [RAII resource wrappers. All classes delete copy operations and implement move semantics, preventing accidental double-free or resource leaks.],
) <tab:raii>

`DeviceMatrix` composes `CudaMemory<float>` with row/column dimensions, providing `init_random()`, `zero()`, and host-transfer methods. The benchmark code uses `DeviceMatrix` throughout, reducing buffer management from $tilde$10 lines per matrix to a single constructor call.

== Architecture Overview

#figure(
  kind: image,
  placement: auto,
  block(width: 100%, inset: 4pt)[
    #set text(size: 7pt)
    #block(width: 100%, inset: 4pt, radius: 3pt, stroke: gray + 1pt, fill: gray.lighten(92%))[
      #align(center)[#text(weight: "bold", fill: gray)[Application Layer]]
      #v(2pt)
      #grid(columns: (1fr, 1fr), gutter: 4pt,
        block(inset: 3pt, radius: 3pt, stroke: gray + 0.5pt, fill: white)[`bench_single_gpu` \ Correctness + GFLOPS],
        block(inset: 3pt, radius: 3pt, stroke: gray + 0.5pt, fill: white)[`bench_multi_gpu` \ 6 scaling experiments],
      )
    ]
    #v(3pt)
    #block(width: 100%, inset: 4pt, radius: 3pt, stroke: rgb("#a855f7") + 1pt, fill: rgb("#a855f7").lighten(92%))[
      #align(center)[#text(weight: "bold", fill: rgb("#a855f7"))[Tensor Parallelism Layer]]
      #v(2pt)
      #grid(columns: (1fr, 1fr, 1fr), gutter: 4pt,
        block(inset: 3pt, radius: 3pt, stroke: rgb("#a855f7") + 0.5pt, fill: white)[*Column Parallel* \ Fwd + Bwd \ `AllGather`/`AllReduce`],
        block(inset: 3pt, radius: 3pt, stroke: rgb("#a855f7") + 0.5pt, fill: white)[*Row Parallel* \ Fwd + Bwd \ `AllReduce`],
        block(inset: 3pt, radius: 3pt, stroke: rgb("#a855f7") + 0.5pt, fill: white)[*MLP Block* \ Col$arrow.r$Row compose \ Overlap pipelining],
      )
    ]
    #v(3pt)
    #grid(columns: (1fr, 1fr), gutter: 4pt,
      block(width: 100%, inset: 4pt, radius: 3pt, stroke: rgb("#3b82f6") + 1pt, fill: rgb("#3b82f6").lighten(92%))[
        #align(center)[#text(weight: "bold", fill: rgb("#3b82f6"))[Kernel Layer]]
        #v(2pt)
        `GemmKernel` (abstract base) \
        `KernelRegistry` (singleton) \
        9 self-registering kernel classes
      ],
      block(width: 100%, inset: 4pt, radius: 3pt, stroke: rgb("#22c55e") + 1pt, fill: rgb("#22c55e").lighten(92%))[
        #align(center)[#text(weight: "bold", fill: rgb("#22c55e"))[CUDA Resource Layer]]
        #v(2pt)
        `CudaMemory<T>`, `DeviceMatrix` \
        `CudaStream`, `CudaEvent` \
        `CublasHandle`, `GpuTimer`
      ],
    )
  ],
  caption: [System architecture. The three layers are loosely coupled: kernels are pluggable via the registry, RAII wrappers manage GPU resources, and the tensor parallelism layer composes both.],
) <fig:arch>

The tensor parallelism layer (@fig:arch, middle) consumes kernels through the `const GemmKernel&` interface, remaining agnostic to the specific kernel implementation. Gradient GEMM operations (transpose + multiply) are factored into reusable `grad_gemm_at_b` and `grad_gemm_a_bt` helpers, reducing the $tilde$200 lines of duplicated backward-pass code to two composable primitives.

// ═════════════════════════════════════════════════════════════════════
= Single-GPU Kernel Optimization
// ═════════════════════════════════════════════════════════════════════

== Optimization Roadmap

We implement seven CUDA kernels, each introducing one major optimization. @tab:kernels summarizes the progression.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 6pt,
    align: (left, left, left, left),
    stroke: none,
    table.hline(),
    table.header(
      [*Kernel*], [*Technique*], [*Key Idea*], [*Effect*],
    ),
    table.hline(stroke: 0.5pt),
    [1. Naive], [Baseline], [1 thread $arrow.r$ 1 element, $K$ global loads/FMA], [$cal(O)(1)$ FLOP/byte],
    [2. Coalesced], [Memory coalescing], [threadIdx.x $arrow.r$ column for stride-1], [Fewer transactions],
    [3. Shared Mem], [SRAM tiling], [$32 times 32$ tiles reused $32 times$], [$32 times$ less traffic],
    [4. 1D BlockTile], [Thread coarsening], [Each thread $arrow.r$ TM$=$8 rows], [Better smem reuse],
    [5. 2D BlockTile], [Register blocking], [TM$times$TN$= 8 times 8$ per thread], [$8 times$ fewer smem reads],
    [6. Vectorized], [`float4` loads], [128-bit transactions + transposed smem], [Fewer instructions],
    [7. WarpTile], [Warp-level tiling], [Block $arrow.r$ Warp $arrow.r$ Thread hierarchy], [Better L1 locality],
    table.hline(),
  ),
  caption: [Progressive GEMM kernel optimization roadmap.],
) <tab:kernels>

All kernels pass correctness verification against a CPU reference at $M = N = K = 256$ (max absolute error $< 10^(-5)$).

== Results and Analysis

#fig(
  "../results/figures/kernel_gflops.pdf",
  [GFLOPS across all kernel optimization stages and matrix sizes (the Uncoalesced negative control is omitted; see discussion). Register-blocked kernels (2D BlockTile, Vectorized, WarpTile) climb sharply with size as the working set outgrows the overhead of thread setup; cuBLAS sits on a separate tier courtesy of its tensor-core path.],
  scope: "parent",
  width: 80%,
) <fig:gflops>

@fig:gflops shows the performance progression. Key observations:

- *Naive and Coalesced kernels* plateau at $tilde 6{,}300$ GFLOPS regardless of matrix size, confirming they are memory-bandwidth-bound. The coalesced variant is $12.7 times$ faster than the deliberately uncoalesced control (6,338 vs. 498 GFLOPS at $N = 4096$), a vivid demonstration that uncoalesced global-memory accesses alone can squander more than an order of magnitude of throughput.

- *Shared memory tiling* (Kernel 3) raises throughput to $tilde 9{,}000$ GFLOPS by reducing global traffic by $32 times$, but remains below the compute-bound regime.

- *Register blocking* (Kernels 4--7) produces the most dramatic gains. The 2D block tile reaches 22,062 GFLOPS at $N = 4096$---a $2.4 times$ improvement over shared memory alone. Vectorized `float4` loads push this to 32,492 GFLOPS at $N = 4096$, and the warp-tiled variant sustains 32,820 GFLOPS at $N = 16384$.

- *cuBLAS* peaks at 51,662 GFLOPS at $N = 8192$ and sustains $tilde 51$ TFLOPS thereafter, about 77% of the 67 TFLOPS FP32 ceiling. It outperforms our best hand-written kernel by $1.59 times$; the gap is attributable to tensor-core paths (TF32 with FP32 accumulation) and extensive offline autotuning that an FP32-only implementation cannot replicate.

#fig(
  "../results/figures/cublas_percentage.pdf",
  [Each kernel's throughput as a percentage of cuBLAS. The Vectorized kernel reaches 63% of cuBLAS at $N = 4096$; at very small sizes ($N = 256$) the Shared-Memory kernel briefly matches cuBLAS because cuBLAS's dispatch overhead dominates its measured time.],
) <fig:pct>

@fig:pct shows the relative trajectory. For $N >= 2048$, the gap to cuBLAS narrows monotonically with optimization level---11% (naive) $arrow.r$ 18% (shared memory) $arrow.r$ 43% (2D block) $arrow.r$ 63% (vectorized/warp). The crossover behaviour below $N = 1024$ reflects small-size launch and library-dispatch overheads rather than steady-state throughput.

#fig(
  "../results/figures/roofline.pdf",
  [Roofline analysis at $N = 4096$. The FP32 ceiling is 67 TFLOPS (FMA-counted) and the ridge point is at 20 FLOP/byte; SGEMM at $N = 4096$ has an operational intensity of 683 FLOP/byte, deeply in the compute-bound regime. All kernels are plotted at that OI with slight horizontal jitter; vertical position therefore measures their achieved fraction of peak compute.],
) <fig:roofline>

The roofline view (@fig:roofline) reinforces the takeaway: at sizes relevant for deep learning, SGEMM is not memory-bound in principle---the ridge point lies two orders of magnitude to the left of the workload's OI. The achieved performance therefore measures how close each kernel comes to the flat compute roof. cuBLAS reaches 77% of peak; the warp-tiled kernel, 49%; the naive kernel, 9.4%. The missing compute ceiling for hand-written kernels is not algorithmic but organizational: register-file pressure, scheduling of FMAs inside a warp, and tensor-core utilization all separate a respectable kernel from a peak one.

// ═════════════════════════════════════════════════════════════════════
= Intra-Node Tensor Parallelism (NVLink)
// ═════════════════════════════════════════════════════════════════════

We implement the tensor-parallel primitives described in @tab:mlp_comm using NCCL. Each GPU runs in a separate host thread with its own CUDA stream and cuBLAS handle; NCCL communicators for each target GPU count are pre-initialised via `ncclCommInitAll` at startup. The implementation includes:

- *Column-parallel forward/backward*: local GEMM + `ncclAllGather` (forward) / `ncclAllReduce` (backward).
- *Row-parallel forward/backward*: local GEMM + `ncclAllReduce` (forward) / no communication (backward).
- *Parallel MLP block*: composed column $arrow.r$ row parallelism, matching the two-`AllReduce` pattern in @tab:mlp_comm.
- *Communication--computation overlap*: the output matrix is chunked along the $M$ dimension; each chunk's `AllReduce` is pipelined with the next chunk's GEMM on a separate CUDA stream, synchronised via `cudaEvent`.

All intra-node experiments use 8$times$H100 GPUs connected via NVLink 4.0. Backward passes that call our row-major custom kernels use `cublasSgeam` for explicit transposition, since those kernels lack a built-in transposed variant.

== Strong Scaling <sec:strong>

#fig(
  "../results/figures/strong_scaling.pdf",
  [Strong scaling: total wall-clock time vs. number of GPUs for four matrix sizes, combining single-node (1, 2, 4, 8 GPUs over NVLink) and cross-node (16 GPUs over IB) measurements on a single axis. Dashed lines give ideal $T_1 slash P$ scaling. Larger matrices scale substantially further before communication dominates.],
) <fig:strong>

@fig:strong summarises the scaling behaviour. The $N = 16384$ workload scales from 171.6 ms (1 GPU) to 24.4 ms (8 GPUs), a $7.03 times$ speedup and *87.8% parallel efficiency*, reaching 360 TFLOPS aggregate throughput. At $N = 8192$, the curve flattens earlier ($5.8 times$ at 8 GPUs) because the 128 MB `AllGather` payload already costs a non-trivial fraction of the shorter GEMM. The $N = 2048$ line is the cautionary tale: past 4 GPUs the 16 MB `AllGather` takes longer than the shard-sized GEMM, so adding GPUs *increases* wall-clock time. The jog at 8$arrow.r$16 GPUs for $N = 2048$ and $N = 4096$ marks the transition from NVLink to cross-node IB (see Section 7).

#fig(
  "../results/figures/strong_scaling_efficiency.pdf",
  [Parallel efficiency $eta = T_1 slash (P dot T_P)$. At $N = 16384$, efficiency remains 88% at 8 GPUs on NVLink and 61% at 16 GPUs across two nodes; small matrices collapse to near-zero efficiency well before the node boundary.],
) <fig:efficiency>

@fig:efficiency makes the same story explicit. Large matrices retain high efficiency because compute grows as $O(N^3)$ while communication volume grows only as $O(N^2)$. Small matrices dip below 10% efficiency within the NVLink domain.

== Weak Scaling

#fig(
  "../results/figures/weak_scaling.pdf",
  [Weak scaling: three per-GPU tile sizes ($2048$, $4096$, $8192$) held fixed while the number of GPUs increases. Left: total time (log scale) grows mildly---the overhead is entirely from `AllGather`. Right: aggregate throughput in GFLOPS approaches linear scaling as the per-GPU tile grows.],
  scope: "parent",
  width: 80%,
) <fig:weak>

Under weak scaling (@fig:weak), the 2048 tile incurs the largest relative slowdown ($2.75 times$ at 8 GPUs) because `AllGather` latency dominates; the 8192 tile stays within $1.20 times$ of the single-GPU time and reaches 324 TFLOPS aggregate, $6.7 times$ the single-GPU baseline at that tile size. Weak scaling therefore acts as a direct probe of the communication-to-computation ratio at each tile size.

== Communication--Computation Analysis <sec:ratio>

This is the central experiment of the project: _how does the communication-to-computation ratio evolve as local kernels are optimised, and how does it scale with matrix size?_

#fig(
  "../results/figures/comm_compute_ratio_size.pdf",
  [GEMM time vs.\ communication time at different matrix sizes on 8 GPUs (cuBLAS kernel, NVLink). The annotated ratio drops from 1.02 at $N = 2048$ to 0.19 at $N = 16384$ as compute grows $O(N^3)$ while the `AllGather` payload grows only $O(N^2)$.],
) <fig:ratio_size>

@fig:ratio_size captures the cubic-vs-quadratic scaling law. At $N = 2048$, communication (0.54 ms) and compute (0.53 ms) are essentially equal. By $N = 16384$, compute grows to 21.7 ms while communication is only 4.2 ms---a ratio of 0.19. This matches the asymptotic expectation $tilde N slash B$ for ratio$(N)$ and predicts that contemporary Transformer hidden dimensions ($N gt.eq 8192$, as in Llama-2 70B or GPT-3 175B) sit comfortably in the compute-dominated regime.

#fig(
  "../results/figures/comm_compute_ratio_kernel.pdf",
  [Left: stacked absolute GEMM and communication time per kernel at $N = 4096$, 8 GPUs (the Uncoalesced kernel is dropped because its 35 ms GEMM compresses the chart). Right: communication-to-computation ratio rising monotonically from 0.22 (Naive) to 0.88 (cuBLAS) as the local GEMM gets faster.],
  scope: "parent",
  width: 80%,
) <fig:ratio_kernel>

@fig:ratio_kernel is the central result of the project. Communication time is the same ($approx 0.69$ ms) for every local kernel---it is a property of the `AllGather` payload and the NVLink fabric, not of the local GEMM. As the local GEMM accelerates from 3.20 ms (Naive) to 0.78 ms (cuBLAS), the ratio rises from 0.22 to 0.88. Interpreting the extremes:

- With a *naive* kernel, communication is only 22% of compute: further kernel optimisation translates almost directly into end-to-end speedup.
- With *cuBLAS*, communication has already climbed to 88% of compute: even infinite further GEMM speedup would buy at most a $1.88 times$ end-to-end improvement. The system is within a factor of two of the _communication-bound_ regime.

The crossover is the fundamental tension of distributed deep learning: once local compute is fast enough, only the interconnect matters.

== MLP Forward and Backward

#fig(
  "../results/figures/mlp_fwd_bwd.pdf",
  [Parallel MLP (column $arrow.r$ row) forward + backward on 8 GPUs. As the matrix grows, the backward/forward ratio converges to its algorithmic limit of $2 times$ (2 GEMMs forward, 4 GEMMs backward).],
) <fig:mlp>

@fig:mlp shows the MLP block's forward and backward time. At $N gt.eq 8192$, the backward pass is almost exactly $2 times$ the forward, consistent with the algorithmic count (2 GEMMs forward vs. 4 GEMMs backward, each pass sharing a single `AllReduce`). At small sizes, the ratio inflates ($10.3 times$ at $N = 2048$) because fixed per-call launch and synchronisation overhead in the backward's extra kernels dwarfs the actual arithmetic. This is an artefact of the launch-overhead regime rather than a property of the algorithm.

== Communication--Computation Overlap

#fig(
  "../results/figures/overlap_comparison.pdf",
  [Row-parallel forward with and without chunked overlap (4 chunks). Overlap is neutral-to-harmful at small sizes (event-sync overhead dominates), helps at $N = 8192$ ($1.25 times$ nominal, high variance), and settles to a modest $1.06 times$ at $N = 16384$.],
) <fig:overlap>

The overlap experiment (@fig:overlap) splits the output along the $M$ dimension into four chunks, pipelining each chunk's `AllReduce` with the next chunk's GEMM on a separate CUDA stream. At $N = 2048$ and $N = 4096$, overlap actually hurts ($0.90 times$ and $0.94 times$): the per-chunk GEMM is too small to hide the extra event synchronisation. At $N = 8192$ the nominal speedup is $1.25 times$, but the no-overlap baseline has 73% coefficient of variation in that row (`Total_std` of 3.8 ms on 5.2 ms mean), so the measurement is dominated by outliers. The most reliable point is $N = 16384$, where overlap delivers a clean $1.06 times$ with low variance. On H100 with NVLink, `AllReduce` latency is only $tilde 3$ ms even for 1 GB payloads, so the hideable window is small. We show in Section 7 that overlap becomes considerably more valuable when the fabric latency rises.

// ═════════════════════════════════════════════════════════════════════
= Cross-Node Tensor Parallelism (InfiniBand)
// ═════════════════════════════════════════════════════════════════════

To characterise tensor parallelism beyond the NVLink domain, we replicate the column-parallel strong-scaling and ratio-vs-size experiments on 16 GPUs spanning two nodes connected by 4$times$400 Gb/s InfiniBand. NCCL is forced into each of four transport configurations in a single benchmark run so that all transports see the same process layout and GPU placement:

#figure(
  table(
    columns: (auto, auto),
    inset: 6pt,
    align: (left, left),
    stroke: none,
    table.hline(),
    table.header([*Tag*], [*Environment*]),
    table.hline(stroke: 0.5pt),
    [`auto`], [NCCL default selection (picks IB + GDRDMA on this system)],
    [`ib`],   [`NCCL_NET=IB NCCL_IB_DISABLE=0`],
    [`ring`], [`NCCL_ALGO=Ring`, otherwise auto],
    [`tcp`],  [`NCCL_IB_DISABLE=1` --- forces the TCP socket plugin],
    table.hline(),
  ),
  caption: [Transport configurations swept across the 16-GPU run. `NCCL_ALGO=Tree` was attempted but NCCL does not support `AllGather` under the Tree algorithm, so that variant is omitted.],
) <tab:transports>

All four runs verify that the `auto`, `ib`, and `ring` paths use `NET/IB` with `GDRDMA` over all four HCAs (`mlx5_0/1/4/5`), while `tcp` falls back to plain sockets as intended.

== Transport Sweep: IB vs.\ TCP <sec:transport>

#fig(
  "../results/figures/transport_sweep.pdf",
  [Left: per-size `AllGather` time at 16 GPUs across the four transport configurations (log-log). Right: the resulting communication/computation ratio for the same run. IB, auto, and ring trace the same curve; TCP lies two orders of magnitude above.],
  scope: "parent",
  width: 95%,
) <fig:transport>

@fig:transport is the headline result for the multi-node study. At $N = 32768$, the IB-backed transports move the 4 GB `AllGather` payload in 23.3--25.3 ms (consistent with $tilde 175$ GB/s effective goodput per GPU), while the TCP fallback requires 2.93 _seconds_---a $125 times$ slowdown that completely dominates every other cost in the system. The right panel translates this into ratio-vs-size: the IB ratio peaks at 1.4 around $N = 4096$ and then decreases as compute takes over, whereas TCP peaks at over $100times$ the compute time at $N = 8192$ and remains in the communication-bound regime across the entire measured range.

The practical implication is stark: a hardware-adequate cluster (IB 400 Gb/s with GDRDMA) and a software fallback (TCP over ethernet) are not a spectrum of performance, they are different regimes of operation. In the TCP case, accelerating the local GEMM by any amount is essentially invisible.

== Strong Scaling Across the Node Boundary

The right-most data points in @fig:strong and @fig:efficiency are taken from this run. Three observations:

+ *The node boundary is visible but not catastrophic for compute-heavy shapes.* $N = 16384$ goes from 24.4 ms at 8 NVLink GPUs to 17.5 ms at 16 IB GPUs---a further $1.39 times$ speedup. The efficiency drop from 88% to 61% is the cost of crossing the IB fabric.
+ *Very large matrices cross the boundary cheaply.* The $N = 32768$ workload on 16 GPUs hits 667 TFLOPS aggregate (compute 86 ms, communication 23 ms), the largest throughput we measure anywhere. Its communication/computation ratio at 16 GPUs is 0.28, still well inside the compute-dominated regime.
+ *Small matrices are punished by the fabric jump.* $N = 2048$ and $N = 4096$ go _slower_ when moving from NVLink 8-GPU to IB 16-GPU because both the `AllGather` latency and the shrunken local GEMM move in the wrong direction.

Taken together with the intra-node ratio experiment, the multi-node data gives a concrete prescription: choose tensor parallelism width such that the sharded GEMM stays comfortably to the left of the bandwidth-limited `AllGather`. On this cluster, that threshold is $N approx 8192$ for 8-way NVLink and $N approx 16384$ for 16-way across IB.

// ═════════════════════════════════════════════════════════════════════
= Conclusion
// ═════════════════════════════════════════════════════════════════════

This project provides a comprehensive study of GEMM optimisation and tensor parallelism at three scales---a single GPU, a single-node multi-GPU domain over NVLink, and a cross-node deployment over InfiniBand.

*Single-GPU kernel optimisation.* Seven progressively optimised CUDA kernels trace the path from memory-bound ($tilde 6.3$ TFLOPS) to near-compute-bound ($tilde 32.8$ TFLOPS) FP32 SGEMM. The dominant improvement is 2D register blocking, which amortises shared-memory reads across many FMAs. Our best hand-written kernel reaches 63% of cuBLAS, and the remaining gap is attributable almost entirely to cuBLAS's TF32 tensor-core path and offline autotuning.

*Intra-node tensor parallelism.* On 8$times$H100 NVLink, strong scaling at $N = 16384$ delivers a $7.03 times$ speedup (88% efficiency) and 360 TFLOPS aggregate throughput. The communication-to-computation ratio rises monotonically from 0.22 (naive) to 0.88 (cuBLAS) as the local GEMM gets faster: a direct measurement of the crossover where system-level communication begins to dominate.

*Cross-node tensor parallelism.* Extending to 16 GPUs over 400 Gb/s IB, the $N = 32768$ workload reaches 667 TFLOPS aggregate. A transport sweep quantifies how much the fabric matters: forcing NCCL onto plain TCP sockets inflates the 4 GB `AllGather` by $125 times$, pushing the ratio to $>100$ and swamping any kernel-level optimisation.

*Software engineering.* The codebase follows an object-oriented, RAII-based design with a self-registering kernel registry, move-only resource wrappers (no manual `cudaFree`), and composable gradient-GEMM helpers. New kernels are added through a single file with zero changes to any existing code.

Taken together, the results support a simple design rule: at any given hardware scale, choose tensor-parallel width so the sharded local GEMM stays to the left of the interconnect-bound `AllGather`/`AllReduce`---and verify that the interconnect is actually doing what you think it is.

// ═════════════════════════════════════════════════════════════════════
// References
// ═════════════════════════════════════════════════════════════════════

#bibliography("references.bib", style: "ieee")

// ═════════════════════════════════════════════════════════════════════
// Appendix
// ═════════════════════════════════════════════════════════════════════

= Appendix

#figure(
  table(
    columns: (auto, auto),
    inset: 6pt,
    align: (left, left),
    stroke: none,
    table.hline(),
    table.header([*Component*], [*Specification*]),
    table.hline(stroke: 0.5pt),
    [GPU (per node)], [8 $times$ NVIDIA H100 80 GB HBM3 (SXM5)],
    [Intra-node interconnect], [NVLink 4.0, all-to-all NV18 (900 GB/s bidirectional per GPU)],
    [Inter-node interconnect], [4 $times$ Mellanox ConnectX-7 400 Gb/s InfiniBand per node, GPUDirect RDMA],
    [GPU Compute], [132 SMs, 1980 MHz boost, sm\_90, 67 TFLOPS FP32 dense peak (FMA-counted)],
    [GPU Memory], [80 GB HBM3, $tilde$3.35 TB/s bandwidth per GPU],
    [CPU (per node)], [2 $times$ Intel Xeon Gold 6448Y (32 cores / 64 threads each, 128 threads total)],
    [System Memory (per node)], [2 TB DDR5],
    [OS], [Ubuntu 24.04.4 LTS (kernel 5.14.0)],
    [CUDA Toolkit], [13.1],
    [NCCL], [2.29.3],
    [GPU Driver], [550.90.07],
    [Max world size measured], [16 GPUs (2 nodes $times$ 8 GPUs)],
    table.hline(),
  ),
  caption: [Hardware and software configuration. Single-node runs use one node with 8 GPUs over NVLink; multi-node runs span two nodes over 400 Gb/s InfiniBand.],
) <tab:setup>
