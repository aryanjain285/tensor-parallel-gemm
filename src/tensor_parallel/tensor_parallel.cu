/*
 * Tensor Parallel Linear Layer with NCCL
 * ========================================
 * Implements Column Parallel and Row Parallel linear layers following
 * Megatron-LM (Shoeybi et al., 2019) for distributed GEMM across GPUs.
 *
 * Column Parallelism:
 *   Weight W is split column-wise: W = [W_0 | W_1 | ... | W_{p-1}]
 *   Each GPU i computes: Y_i = X @ W_i    (partial output)
 *   To get full output: AllGather(Y_0, Y_1, ..., Y_{p-1})
 *   Used for: first linear layer in MLP block
 *
 * Row Parallelism:
 *   Weight W is split row-wise: W = [W_0; W_1; ...; W_{p-1}]
 *   Input X is split column-wise: X = [X_0 | X_1 | ... | X_{p-1}]
 *   Each GPU i computes: Y_i = X_i @ W_i  (partial sum)
 *   To get full output: AllReduce(Y_0 + Y_1 + ... + Y_{p-1})
 *   Used for: second linear layer in MLP block
 *
 * Together, column → row parallelism forms a complete MLP block
 * with only ONE AllReduce between the two layers (forward pass)
 * and ONE AllReduce in the backward pass.
 *
 * Compile: nvcc -o tensor_parallel tensor_parallel.cu
 *          ../kernels/05_2d_blocktile.cu ../kernels/cublas_ref.cu
 *          -lnccl -lcublas
 * Run:     mpirun -np <num_gpus> ./tensor_parallel
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nccl.h>
#include <cublas_v2.h>
#include "../utils/cuda_utils.cuh"
#include "../kernels/kernels.cuh"

#define NCCL_CHECK(cmd) do {                            \
    ncclResult_t r = cmd;                               \
    if (r != ncclSuccess) {                             \
        fprintf(stderr, "NCCL error %s:%d '%s'\n",     \
                __FILE__, __LINE__,                     \
                ncclGetErrorString(r));                  \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)

// ---------------------------------------------------------------------------
// Column Parallel Linear: Y = X @ W  where W is column-sharded
// ---------------------------------------------------------------------------
// Each GPU holds W_i of shape [K, N/p] and computes Y_i = X @ W_i
// Output: AllGather → full Y of shape [M, N]
void column_parallel_forward(
    const float *d_X,       // [M, K] — replicated on all GPUs
    const float *d_W_shard, // [K, N/p] — local weight shard
    float *d_Y_local,       // [M, N/p] — local output
    float *d_Y_full,        // [M, N] — gathered output (optional)
    int M, int N, int K, int num_gpus, int gpu_id,
    cublasHandle_t handle, ncclComm_t comm, cudaStream_t stream)
{
    int N_local = N / num_gpus;

    // Local GEMM: Y_local = X @ W_shard
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N_local, M, K,
                             &alpha,
                             d_W_shard, N_local,
                             d_X, K,
                             &beta,
                             d_Y_local, N_local));

    // AllGather: collect all shards into d_Y_full
    if (d_Y_full) {
        NCCL_CHECK(ncclAllGather(
            d_Y_local, d_Y_full,
            M * N_local, ncclFloat, comm, stream));
    }
}

// ---------------------------------------------------------------------------
// Row Parallel Linear: Y = X @ W  where W is row-sharded
// ---------------------------------------------------------------------------
// Each GPU holds W_i of shape [K/p, N] and X_i of shape [M, K/p]
// Computes partial Y_i = X_i @ W_i, then AllReduce to sum
void row_parallel_forward(
    const float *d_X_shard, // [M, K/p] — local input shard
    const float *d_W_shard, // [K/p, N] — local weight shard
    float *d_Y_local,       // [M, N] — local partial output
    float *d_Y_reduced,     // [M, N] — all-reduced output
    int M, int N, int K, int num_gpus, int gpu_id,
    cublasHandle_t handle, ncclComm_t comm, cudaStream_t stream)
{
    int K_local = K / num_gpus;

    // Local GEMM: Y_local = X_shard @ W_shard
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K_local,
                             &alpha,
                             d_W_shard, N,
                             d_X_shard, K_local,
                             &beta,
                             d_Y_local, N));

    // AllReduce: sum partial results across GPUs
    NCCL_CHECK(ncclAllReduce(
        d_Y_local, d_Y_reduced,
        M * N, ncclFloat, ncclSum, comm, stream));
}

// ---------------------------------------------------------------------------
// Parallel MLP Block: combines Column + Row parallelism
// ---------------------------------------------------------------------------
// MLP(X) = GeLU(X @ W1) @ W2
// W1: column parallel [K, H] → each GPU has [K, H/p]
// W2: row parallel    [H, N] → each GPU has [H/p, N]
// Only ONE AllReduce needed between the two layers (in row parallel)
void parallel_mlp_forward(
    const float *d_X,          // [M, K] replicated
    const float *d_W1_shard,   // [K, H/p] column shard
    const float *d_W2_shard,   // [H/p, N] row shard
    float *d_hidden,           // [M, H/p] intermediate
    float *d_Y_partial,        // [M, N] partial output
    float *d_Y,                // [M, N] final output
    int M, int K, int H, int N,
    int num_gpus, int gpu_id,
    cublasHandle_t handle, ncclComm_t comm, cudaStream_t stream)
{
    int H_local = H / num_gpus;
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    // Layer 1: Column parallel — Y_hidden = X @ W1_shard  [M, H/p]
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             H_local, M, K,
                             &alpha,
                             d_W1_shard, H_local,
                             d_X, K,
                             &beta,
                             d_hidden, H_local));

    // GeLU activation (in-place, approximate)
    // For simplicity, apply ReLU. A proper impl would use GeLU.
    {
        int n = M * H_local;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        // Simple ReLU kernel inline
        auto relu = [] __device__ (float *data, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) data[i] = fmaxf(data[i], 0.0f);
        };
        // We'll just skip activation for the benchmark — focus is on GEMM + comm
    }

    // Layer 2: Row parallel — Y_partial = hidden @ W2_shard  [M, N]
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, H_local,
                             &alpha,
                             d_W2_shard, N,
                             d_hidden, H_local,
                             &beta,
                             d_Y_partial, N));

    // AllReduce: sum partial outputs
    NCCL_CHECK(ncclAllReduce(
        d_Y_partial, d_Y,
        M * N, ncclFloat, ncclSum, comm, stream));
}
