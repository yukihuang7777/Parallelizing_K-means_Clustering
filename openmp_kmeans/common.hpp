#pragma once
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>
#include "../dataset.hpp"

#ifndef _OPENMP
#error Compile with OpenMP enabled
#endif

using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct Options {
    long long n = 10000;
    int d = 8, k = 10, max_iter = 100;
    unsigned int seed = 42;
    double tol = 1e-6;
    bool centroids = false, cpu = false;
};

inline Options options(int argc, char** argv, bool allow_cpu = false) {
    Options o;
    // Keep the six positional arguments identical to serial and MPI.
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        if (s == "--centroids") { o.centroids = true; continue; }
        if (s == "--cpu" && allow_cpu) { o.cpu = true; continue; }
        std::size_t end = 0;
        switch (positional++) {
        case 0: o.n = std::stoll(s, &end); break;
        case 1: o.d = std::stoi(s, &end); break;
        case 2: o.k = std::stoi(s, &end); break;
        case 3: {
            auto seed = std::stoull(s, &end);
            if (s[0] == '-' || seed > UINT_MAX) throw std::invalid_argument("Invalid seed");
            o.seed = static_cast<unsigned int>(seed); break;
        }
        case 4: o.max_iter = std::stoi(s, &end); break;
        case 5: o.tol = std::stod(s, &end); break;
        default: throw std::invalid_argument("Too many arguments");
        }
        if (end != s.size()) throw std::invalid_argument("Invalid numeric argument: " + s);
    }
    if (o.n <= 0 || o.d <= 0 || o.k <= 0 || o.k > o.n || o.max_iter <= 0 ||
        !std::isfinite(o.tol) || o.tol < 0 ||
        static_cast<long long>(o.k) * o.d > 10000000 || o.n > INT_MAX / o.d)
        throw std::invalid_argument("Invalid input parameters (same size limits as MPI)");
    return o;
}

inline int nearest(const double* point, const double* centers, int d, int k) {
    double best = 1e300;
    int label = 0;
    for (int c = 0; c < k; ++c) {
        double distance = 0;
        for (int j = 0; j < d; ++j) {
            double diff = point[j] - centers[c * d + j];
            distance += diff * diff;
        }
        if (distance < best) { best = distance; label = c; }
    }
    return label;
}

inline double update_centers(std::vector<double>& centers,
    const std::vector<double>& sums, const std::vector<long long>& counts, int d, int k) {
    double max_shift = 0;
    for (int c = 0; c < k; ++c) {
        if (counts[c] == 0) continue;
        double shift = 0;
        for (int j = 0; j < d; ++j) {
            int idx = c * d + j;
            double value = sums[idx] / counts[c];
            double diff = value - centers[idx];
            shift += diff * diff;
            centers[idx] = value;
        }
        max_shift = std::max(max_shift, shift);
    }
    return std::sqrt(max_shift);
}

inline double wcss(const std::vector<double>& data, const std::vector<double>& centers,
                   long long n, int d, int k) {
    double total = 0;
    for (long long i = 0; i < n; ++i) {
        const double* p = data.data() + static_cast<std::size_t>(i) * d;
        int c = nearest(p, centers.data(), d, k);
        double distance = 0;
        for (int j = 0; j < d; ++j) {
            double diff = p[j] - centers[c * d + j];
            distance += diff * diff;
        }
        total += distance;
    }
    return total;
}

inline void print_centers(const std::vector<double>& centers, int d, int k) {
    std::cerr << std::setprecision(12);
    for (int c = 0; c < k; ++c) {
        std::cerr << "Centroid " << c << ":";
        for (int j = 0; j < d; ++j) std::cerr << " " << centers[c * d + j];
        std::cerr << "\n";
    }
}

// Separate accumulators avoid atomics on every coordinate. Merge in thread order.
struct ThreadStats {
    int threads, d, k;
    std::vector<double> sums;
    std::vector<long long> counts;
    ThreadStats(int t, int dim, int clusters) : threads(t), d(dim), k(clusters),
        sums(static_cast<std::size_t>(t) * dim * clusters),
        counts(static_cast<std::size_t>(t) * clusters) {}

    void assign(const std::vector<double>& data, const std::vector<double>& centers,
                long long n, std::vector<double>& result_sums,
                std::vector<long long>& result_counts) {
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        #pragma omp parallel num_threads(threads)
        {
            int t = omp_get_thread_num();
            double* s = sums.data() + static_cast<std::size_t>(t) * k * d;
            long long* cnt = counts.data() + static_cast<std::size_t>(t) * k;
            #pragma omp for schedule(static)
            for (long long i = 0; i < n; ++i) {
                const double* p = data.data() + static_cast<std::size_t>(i) * d;
                int c = nearest(p, centers.data(), d, k);
                ++cnt[c];
                for (int j = 0; j < d; ++j) s[c * d + j] += p[j];
            }
        }
        std::fill(result_sums.begin(), result_sums.end(), 0.0);
        std::fill(result_counts.begin(), result_counts.end(), 0);
        for (int t = 0; t < threads; ++t) {
            for (int j = 0; j < k * d; ++j)
                result_sums[j] += sums[static_cast<std::size_t>(t) * k * d + j];
            for (int c = 0; c < k; ++c)
                result_counts[c] += counts[static_cast<std::size_t>(t) * k + c];
        }
    }
};

inline int actual_threads() {
    omp_set_dynamic(0);
    int t = 1;
    #pragma omp parallel
    {
        #pragma omp single
        t = omp_get_num_threads();
    }
    return t;
}
