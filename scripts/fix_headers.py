#!/usr/bin/env python3
"""Replace old-style block comments with Google-style copyright headers."""

import re
from pathlib import Path

COPYRIGHT = """\
// Copyright 2025 Aryan Jain, Fanyi Pu, Ze Hong Maxwell Au
// SC4064 GPU Programming, Nanyang Technological University"""

# Map: filename -> one-line description
DESCRIPTIONS = {
    # Kernels
    "naive.cu": "Naive GEMM kernel -- one thread per output element, no data reuse.",
    "coalesced.cu": "Coalesced and uncoalesced GEMM kernels for memory access comparison.",
    "smem_tiling.cu": "Shared memory tiling GEMM kernel -- 32x32 tiles, 32x traffic reduction.",
    "blocktile_1d.cu": "1D block tiling GEMM kernel -- thread coarsening along M (TM=8).",
    "blocktile_2d.cu": "2D block tiling GEMM kernel -- register blocking (TM=TN=8).",
    "vectorized.cu": "Vectorized GEMM kernel -- float4 loads with transposed shared memory.",
    "warptile.cu": "Warp tiling GEMM kernel -- Block/Warp/Thread hierarchy (128x128x8).",
    "cublas.cu": "cuBLAS GEMM kernel wrapper via cublasSgemm.",
    "gemm_kernel.cuh": "GemmKernel abstract base class for all GEMM implementations.",
    "kernel_registry.cuh": "KernelRegistry singleton -- self-registering kernel management.",
    # Tensor parallel
    "tensor_parallel.cu": (
        "Tensor parallel linear layers (column/row parallel, MLP block, overlap)."
    ),
    "tensor_parallel.cuh": "Tensor parallel layer declarations.",
    # Benchmarks
    "bench_single_gpu.cu": "Single-GPU GEMM benchmark -- correctness verification and GFLOPS.",
    "bench_multi_gpu.cu": "Multi-GPU tensor parallelism benchmark -- 6 scaling experiments.",
    # Utils
    "cuda_utils.cuh": "CUDA utility macros, error checking, timing, and verification helpers.",
    "cuda_raii.cuh": (
        "Move-only RAII wrappers for CUDA resources (memory, streams, events, handles)."
    ),
    "device_matrix.cuh": "DeviceMatrix -- RAII 2D float matrix with initialization helpers.",
    "nccl_utils.cuh": "NCCL error checking macro.",
}

# Regex to match old block comment at file start
OLD_HEADER = re.compile(
    r"^/\*\n"         # /*
    r"(?:\s*\*.*\n)*"  # * lines
    r"\s*\*/\n*",      # */
)

ROOT = Path(__file__).resolve().parent.parent / "src"


def process_file(path: Path):
    name = path.name
    desc = DESCRIPTIONS.get(name)
    if not desc:
        print(f"  SKIP {path.relative_to(ROOT)} (no description)")
        return

    text = path.read_text()

    # Strip old block comment header
    text = OLD_HEADER.sub("", text, count=1)

    # Strip leading blank lines
    text = text.lstrip("\n")

    # Build new header
    header = f"{COPYRIGHT}\n//\n// {path.name} - {desc}\n\n"

    # Don't double-add if already has copyright
    if text.startswith("// Copyright"):
        print(f"  SKIP {path.relative_to(ROOT)} (already has copyright)")
        return

    path.write_text(header + text)
    print(f"  OK   {path.relative_to(ROOT)}")


def main():
    files = sorted(ROOT.rglob("*.cu")) + sorted(ROOT.rglob("*.cuh"))
    print(f"Processing {len(files)} files:")
    for f in files:
        process_file(f)


if __name__ == "__main__":
    main()
