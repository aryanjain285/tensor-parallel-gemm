/*
 * Multi-GPU Tensor Parallelism Benchmark
 * =======================================
 * Evaluates strong scaling, weak scaling, and communication overhead
 * for tensor-parallel GEMM across multiple GPUs.
 *
 * Compile: nvcc -o bench_multi_gpu bench_multi_gpu.cu
 *          ../tensor_parallel/tensor_parallel.cu
 *          ../kernels/cublas_ref.cu
 *          -lnccl -lcublas
 *
 * Run:     ./bench_multi_gpu <num_gpus>
 *
 * NOTE: This uses NCCL with a single process managing multiple GPUs.
 *       For true multi-node, wrap with MPI.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>
#include <cublas_v2.h>
#include "../utils/cuda_utils.cuh"

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
// Benchmark helpers
// ---------------------------------------------------------------------------

// Time GEMM only (no communication)
double time_local_gemm(cublasHandle_t handle, const float *dA, const float *dB,
                       float *dC, int M, int N_local, int K,
                       cudaStream_t stream, int warmup = 3, int repeat = 10) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSetStream(handle, stream);

    for (int i = 0; i < warmup; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N_local, M, K, &alpha, dB, N_local, dA, K, &beta, dC, N_local);
    }
    cudaStreamSynchronize(stream);

    GpuTimer timer;
    timer.tic(stream);
    for (int i = 0; i < repeat; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N_local, M, K, &alpha, dB, N_local, dA, K, &beta, dC, N_local);
    }
    float ms = timer.toc(stream);
    return ms / repeat;
}

// Time AllReduce only
double time_allreduce(float *sendbuf, float *recvbuf, int count,
                      ncclComm_t comm, cudaStream_t stream,
                      int warmup = 3, int repeat = 10) {
    for (int i = 0; i < warmup; i++) {
        ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream);
    }
    cudaStreamSynchronize(stream);

    GpuTimer timer;
    timer.tic(stream);
    for (int i = 0; i < repeat; i++) {
        ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream);
    }
    float ms = timer.toc(stream);
    return ms / repeat;
}

int main(int argc, char **argv) {
    int num_gpus = 2;
    if (argc > 1) num_gpus = atoi(argv[1]);

    int available_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&available_gpus));
    if (num_gpus > available_gpus) {
        printf("Requested %d GPUs but only %d available. Using %d.\n",
               num_gpus, available_gpus, available_gpus);
        num_gpus = available_gpus;
    }

    printf("===== Multi-GPU Tensor Parallel Benchmark (%d GPUs) =====\n\n", num_gpus);

    // Print info for each GPU
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        print_device_info();
    }

    // --- Initialize NCCL ---
    std::vector<ncclComm_t> comms(num_gpus);
    std::vector<int> devs(num_gpus);
    for (int i = 0; i < num_gpus; i++) devs[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms.data(), num_gpus, devs.data()));

    // --- Create per-GPU resources ---
    std::vector<cublasHandle_t> handles(num_gpus);
    std::vector<cudaStream_t> streams(num_gpus);
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUBLAS_CHECK(cublasCreate(&handles[i]));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // ===================================================================
    // Experiment 1: Strong Scaling (Column Parallel)
    // Fixed total work, increase GPUs
    // ===================================================================
    printf("\n===== Strong Scaling: Column Parallel (Row-Parallel AllReduce) =====\n");
    printf("%-6s %-6s %-6s %-6s  %10s  %10s  %10s  %8s\n",
           "M", "N", "K", "GPUs", "GEMM(ms)", "Comm(ms)", "Total(ms)", "GFLOPS");
    printf("-----------------------------------------------------------------------\n");

    std::vector<int> sizes = {2048, 4096, 8192};

    for (int S : sizes) {
        int M = S, N = S, K = S;
        int N_local = N / num_gpus;

        // Allocate on each GPU
        struct GpuData {
            float *dX, *dW, *dY, *dY_reduced;
        };
        std::vector<GpuData> gpu_data(num_gpus);

        for (int g = 0; g < num_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMalloc(&gpu_data[g].dX,  M * K * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&gpu_data[g].dW,  K * N_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&gpu_data[g].dY,  M * N_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&gpu_data[g].dY_reduced, M * N * sizeof(float)));

            // Initialize with random data
            float *h = (float*)malloc(M * K * sizeof(float));
            init_matrix(h, M, K, 42 + g);
            CUDA_CHECK(cudaMemcpy(gpu_data[g].dX, h, M * K * sizeof(float),
                                  cudaMemcpyHostToDevice));
            free(h);
            h = (float*)malloc(K * N_local * sizeof(float));
            init_matrix(h, K, N_local, 137 + g);
            CUDA_CHECK(cudaMemcpy(gpu_data[g].dW, h, K * N_local * sizeof(float),
                                  cudaMemcpyHostToDevice));
            free(h);
        }

        // Benchmark GEMM time (use GPU 0 as representative)
        CUDA_CHECK(cudaSetDevice(0));
        double gemm_ms = time_local_gemm(handles[0], gpu_data[0].dX,
                                          gpu_data[0].dW, gpu_data[0].dY,
                                          M, N_local, K, streams[0]);

        // Benchmark AllReduce time
        double comm_ms = time_allreduce(gpu_data[0].dY, gpu_data[0].dY_reduced,
                                         M * N_local, comms[0], streams[0]);

        double total_ms = gemm_ms + comm_ms;
        double gflops = gemm_gflops(M, N, K, total_ms);

        printf("%-6d %-6d %-6d %-6d  %10.3f  %10.3f  %10.3f  %8.1f\n",
               M, N, K, num_gpus, gemm_ms, comm_ms, total_ms, gflops);

        // Cleanup
        for (int g = 0; g < num_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaFree(gpu_data[g].dX));
            CUDA_CHECK(cudaFree(gpu_data[g].dW));
            CUDA_CHECK(cudaFree(gpu_data[g].dY));
            CUDA_CHECK(cudaFree(gpu_data[g].dY_reduced));
        }
    }

    // ===================================================================
    // Experiment 2: Weak Scaling
    // Fixed work per GPU, increase total
    // ===================================================================
    printf("\n===== Weak Scaling: Fixed M=N_local=K=2048 per GPU =====\n");
    printf("%-6s %-6s %-6s %-6s  %10s  %10s  %8s\n",
           "M", "N_tot", "K", "GPUs", "GEMM(ms)", "Comm(ms)", "GFLOPS");
    printf("-----------------------------------------------------------\n");

    {
        int M = 2048, K = 2048;
        int N_local = 2048;
        int N_total = N_local * num_gpus;

        // Allocate
        std::vector<GpuData> gpu_data(num_gpus);
        struct GpuData2 { float *dX, *dW, *dY, *dY_r; };
        std::vector<GpuData2> gd(num_gpus);

        for (int g = 0; g < num_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMalloc(&gd[g].dX, M * K * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&gd[g].dW, K * N_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&gd[g].dY, M * N_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&gd[g].dY_r, M * N_total * sizeof(float)));
        }

        CUDA_CHECK(cudaSetDevice(0));
        double gemm_ms = time_local_gemm(handles[0], gd[0].dX, gd[0].dW,
                                          gd[0].dY, M, N_local, K, streams[0]);
        double comm_ms = time_allreduce(gd[0].dY, gd[0].dY_r,
                                         M * N_local, comms[0], streams[0]);
        double gflops = gemm_gflops(M, N_total, K, gemm_ms + comm_ms);

        printf("%-6d %-6d %-6d %-6d  %10.3f  %10.3f  %8.1f\n",
               M, N_total, K, num_gpus, gemm_ms, comm_ms, gflops);

        for (int g = 0; g < num_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaFree(gd[g].dX));
            CUDA_CHECK(cudaFree(gd[g].dW));
            CUDA_CHECK(cudaFree(gd[g].dY));
            CUDA_CHECK(cudaFree(gd[g].dY_r));
        }
    }

    // ===================================================================
    // Experiment 3: Communication vs Computation ratio
    // ===================================================================
    printf("\n===== Comm/Compute Ratio vs Matrix Size =====\n");
    printf("%-6s  %10s  %10s  %8s\n", "Size", "GEMM(ms)", "Comm(ms)", "Ratio");
    printf("----------------------------------------------\n");

    for (int S : sizes) {
        int M = S, N = S, K = S;
        int N_local = N / num_gpus;

        float *dX, *dW, *dY, *dYr;
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMalloc(&dX, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dW, K * N_local * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dY, M * N_local * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dYr, M * N * sizeof(float)));

        double gemm_ms = time_local_gemm(handles[0], dX, dW, dY,
                                          M, N_local, K, streams[0]);
        double comm_ms = time_allreduce(dY, dYr, M * N_local,
                                         comms[0], streams[0]);

        printf("%-6d  %10.3f  %10.3f  %8.2f\n",
               S, gemm_ms, comm_ms, comm_ms / gemm_ms);

        CUDA_CHECK(cudaFree(dX));
        CUDA_CHECK(cudaFree(dW));
        CUDA_CHECK(cudaFree(dY));
        CUDA_CHECK(cudaFree(dYr));
    }

    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        cublasDestroy(handles[i]);
        cudaStreamDestroy(streams[i]);
        ncclCommDestroy(comms[i]);
    }

    printf("\nDone.\n");
    return 0;
}
