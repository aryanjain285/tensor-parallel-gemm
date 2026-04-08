#!/bin/bash
# ===========================================================================
# NSCC ASPIRE 2A Job Script (PBS)
# ===========================================================================
# Submit: qsub scripts/nscc_job.sh
# Or if using Slurm: sbatch scripts/nscc_job.sh
#
# Adjust the directives below based on your NSCC allocation.
# ===========================================================================

#PBS -N gemm_benchmark
#PBS -q gpu
#PBS -l select=1:ngpus=4:ncpus=16:mem=64gb
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o results/nscc_output.log

# --- For Slurm-based clusters, uncomment below instead ---
# #SBATCH --job-name=gemm_benchmark
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:4
# #SBATCH --cpus-per-task=16
# #SBATCH --mem=64G
# #SBATCH --time=01:00:00
# #SBATCH --output=results/nscc_output.log

cd $PBS_O_WORKDIR 2>/dev/null || cd $SLURM_SUBMIT_DIR 2>/dev/null || true

# Load modules (adjust for your NSCC environment)
module load cuda/12.2
module load nccl/2.18
# module load gcc/11  # if needed

echo "=== Job started at $(date) ==="
echo "=== Node: $(hostname) ==="
nvidia-smi

# Build
make clean
make all GPU_ARCH=sm_80  # A100 = sm_80

# Run
echo ""
echo "=== Single-GPU Benchmark ==="
./build/bench_single_gpu 2>&1 | tee results/single_gpu.txt

echo ""
echo "=== Multi-GPU Benchmark (4 GPUs) ==="
./build/bench_multi_gpu 4 2>&1 | tee results/multi_gpu.txt

# Profile with Nsight Compute (single kernel, small size for fast profiling)
echo ""
echo "=== Nsight Compute Profiling ==="
ncu --set full \
    --kernel-name gemm_2d_blocktile \
    --launch-count 1 \
    -o results/ncu_profile \
    ./build/bench_single_gpu 2>&1 | tee results/ncu_summary.txt

# Nsight Systems timeline for multi-GPU
echo ""
echo "=== Nsight Systems Timeline ==="
nsys profile \
    -o results/nsys_timeline \
    ./build/bench_multi_gpu 4

echo ""
echo "=== Job finished at $(date) ==="
