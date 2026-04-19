#!/bin/bash
# ===========================================================================
# Launch the cross-node benchmark.  Reads PyTorch / K8s PyTorchJob env vars
# (MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK) and invokes the binary once
# per node.  No MPI / srun / qsub required.
#
# Expected contract (set by the job orchestrator, one value per pod):
#   MASTER_ADDR  — hostname of the node whose RANK=0
#   MASTER_PORT  — TCP port for NCCL-id rendezvous (any free port)
#   WORLD_SIZE   — number of NODES (not GPUs).  Expected: 2 for this project.
#   RANK         — this node's index, 0..WORLD_SIZE-1
#
# Optional:
#   KERNEL_ID    — which local GEMM kernel to use (default: cuBLAS)
#   RESULTS_DIR  — where to write the log (default: results/)
#   GPU_ARCH     — used if build is needed (default: sm_90)
# ===========================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${MASTER_ADDR:?MASTER_ADDR not set}"
: "${MASTER_PORT:?MASTER_PORT not set}"
: "${WORLD_SIZE:?WORLD_SIZE not set}"
: "${RANK:?RANK not set}"

RESULTS_DIR="${RESULTS_DIR:-results}"
GPU_ARCH="${GPU_ARCH:-sm_90}"
KERNEL_ID="${KERNEL_ID:-}"
BIN="./build/bench_multi_node"

mkdir -p "$RESULTS_DIR"

if [ ! -x "$BIN" ] || [ "${REBUILD:-0}" = "1" ]; then
    echo "Building bench_multi_node (GPU_ARCH=$GPU_ARCH)..."
    make bench_node GPU_ARCH="$GPU_ARCH"
fi

# Helpful NCCL env
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
# Pin the interface if the cluster has multiple (comment out if not needed)
# export NCCL_SOCKET_IFNAME=eth0

# Derive total GPU count to name the output file; RANK 0 writes, others stay silent
NLOCAL=$(nvidia-smi -L | wc -l)
NTOTAL=$((WORLD_SIZE * NLOCAL))
OUT="$RESULTS_DIR/multi_node_${NTOTAL}gpu.txt"

echo "[rank $RANK] WORLD_SIZE=$WORLD_SIZE local_gpus=$NLOCAL total=$NTOTAL"
echo "[rank $RANK] MASTER=$MASTER_ADDR:$MASTER_PORT"

if [ "$RANK" = "0" ]; then
    "$BIN" ${KERNEL_ID:+$KERNEL_ID} 2>&1 | tee "$OUT"
else
    # Workers don't print the table; redirect to a worker-scoped log to help debugging
    WORKER_LOG="$RESULTS_DIR/multi_node_worker_rank${RANK}.log"
    "$BIN" ${KERNEL_ID:+$KERNEL_ID} >"$WORKER_LOG" 2>&1
    echo "[rank $RANK] finished; worker log at $WORKER_LOG"
fi
