// Copyright 2025 Aryan Jain, Fanyi Pu, Ze Hong Maxwell Au
// SC4064 GPU Programming, Nanyang Technological University
//
// bench_stats.cuh - Statistical summary of per-iteration timing samples.
//
// All three benchmarks record one latency sample per repeated iteration
// (rather than averaging many back-to-back calls into a single number) so
// we can report mean, median, stddev, min, max.  That lets plots show
// error bars and lets reviewers see whether a number is noisy.

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

struct BenchStats {
    double mean;    // arithmetic mean of samples
    double median;  // 50th percentile
    double stddev;  // sample standard deviation (n-1 denominator)
    double min_v;
    double max_v;
    int n;
};

/// Compute summary statistics over a set of positive latency samples (ms).
/// The input vector is mutated (sorted) for median extraction.
inline BenchStats compute_stats(std::vector<double>& samples) {
    const int n = static_cast<int>(samples.size());
    if (n == 0) return BenchStats{0, 0, 0, 0, 0, 0};

    const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    const double mean = sum / n;

    double sq = 0.0;
    for (double v : samples) sq += (v - mean) * (v - mean);
    const double stddev = (n > 1) ? std::sqrt(sq / (n - 1)) : 0.0;

    std::sort(samples.begin(), samples.end());
    const double median =
        (n % 2 == 1) ? samples[n / 2] : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);

    return BenchStats{mean, median, stddev, samples.front(), samples.back(), n};
}
