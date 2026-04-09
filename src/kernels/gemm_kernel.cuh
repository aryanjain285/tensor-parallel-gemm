// Copyright 2025 Aryan Jain, Fanyi Pu, Ze Hong Maxwell Au
// SC4064 GPU Programming, Nanyang Technological University
//
// gemm_kernel.cuh - GemmKernel abstract base class for all GEMM implementations.

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

class GemmKernel {
   public:
    virtual ~GemmKernel() = default;

    /// Human-readable kernel name (e.g. "Naive", "WarpTile", "cuBLAS").
    virtual const char* name() const = 0;

    /// Launch the kernel: C[M×N] = A[M×K] × B[K×N], row-major.
    virtual void launch(const float* A, const float* B, float* C, int M, int N, int K,
                        cudaStream_t stream = 0) const = 0;

    /// Whether this kernel requires a cuBLAS handle (default: false).
    virtual bool needs_cublas() const { return false; }

    /// Provide a cuBLAS handle (only meaningful for CublasKernel).
    virtual void set_cublas_handle(cublasHandle_t /*handle*/) {}

    // Non-copyable, non-movable (registry owns via unique_ptr)
    GemmKernel() = default;
    GemmKernel(const GemmKernel&) = delete;
    GemmKernel& operator=(const GemmKernel&) = delete;
    GemmKernel(GemmKernel&&) = delete;
    GemmKernel& operator=(GemmKernel&&) = delete;
};
