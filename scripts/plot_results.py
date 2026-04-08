#!/usr/bin/env python3
"""
Plot benchmark results from bench_single_gpu and bench_multi_gpu.

Usage:
    ./bench_single_gpu | tee results/single_gpu.txt
    python3 scripts/plot_results.py results/single_gpu.txt
"""

import sys
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_single_gpu(filename):
    """Parse the GFLOPS table from bench_single_gpu output."""
    kernels = {}
    sizes = []
    in_table = False

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if 'Performance Benchmark' in line:
                in_table = True
                continue
            if in_table and line.startswith('Kernel'):
                sizes = [int(x) for x in line.split()[1:]]
                continue
            if in_table and line.startswith('---'):
                continue
            if in_table and line and not line.startswith('Done'):
                parts = line.split()
                name = parts[0]
                gflops = [float(x) for x in parts[1:]]
                kernels[name] = gflops
            if 'Done' in line:
                break

    return sizes, kernels


def plot_gflops_comparison(sizes, kernels, outfile='results/gflops_comparison.png'):
    """Bar chart of GFLOPS across kernel optimizations."""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(sizes))
    n = len(kernels)
    width = 0.8 / n

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))

    for i, (name, gflops) in enumerate(kernels.items()):
        ax.bar(x + i * width, gflops, width, label=name, color=colors[i])

    ax.set_xlabel('Matrix Size (M=N=K)', fontsize=12)
    ax.set_ylabel('GFLOPS', fontsize=12)
    ax.set_title('GEMM Kernel Performance Comparison', fontsize=14)
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")


def plot_cublas_percentage(sizes, kernels, outfile='results/cublas_percentage.png'):
    """Line chart showing each kernel as % of cuBLAS performance."""
    if 'cuBLAS' not in kernels:
        print("No cuBLAS data found, skipping percentage plot.")
        return

    cublas = np.array(kernels['cuBLAS'])
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, gflops in kernels.items():
        if name == 'cuBLAS':
            continue
        pct = np.array(gflops) / cublas * 100
        ax.plot(sizes, pct, 'o-', label=name, linewidth=2, markersize=6)

    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='cuBLAS (100%)')
    ax.set_xlabel('Matrix Size (M=N=K)', fontsize=12)
    ax.set_ylabel('% of cuBLAS Performance', fontsize=12)
    ax.set_title('Kernel Performance Relative to cuBLAS', fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")


def plot_roofline(peak_tflops=19.5, peak_bw_tb=2.0,
                  outfile='results/roofline.png'):
    """Roofline model for A100 with kernel data points."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Operational intensity range
    oi = np.logspace(-2, 3, 500)

    # Roofline: min(peak_compute, peak_bw * OI)
    # peak_compute in GFLOPS, peak_bw in GB/s
    peak_gflops = peak_tflops * 1000
    peak_bw = peak_bw_tb * 1000  # GB/s

    perf = np.minimum(peak_gflops, peak_bw * oi)

    ax.loglog(oi, perf, 'k-', linewidth=2, label='Roofline')
    ax.axhline(y=peak_gflops, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=peak_gflops / peak_bw, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Operational Intensity (FLOP/Byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax.set_title(f'Roofline Model — A100 ({peak_tflops} TFLOPS, {peak_bw_tb} TB/s)',
                 fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 plot_results.py <results_file>")
        print("Generating roofline model only...")
        plot_roofline()
        sys.exit(0)

    sizes, kernels = parse_single_gpu(sys.argv[1])
    if sizes and kernels:
        plot_gflops_comparison(sizes, kernels)
        plot_cublas_percentage(sizes, kernels)
    plot_roofline()
