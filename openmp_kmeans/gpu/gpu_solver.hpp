#pragma once
#include "../common.hpp"

struct GPUResult {
    std::vector<double> centers;
    double time = 0, setup = 0, assignment = 0, objective = 0;
    int iterations = 0;
    bool offloaded = false;
};

inline GPUResult solve_assignment_offload(const std::vector<double>& data,
    std::vector<double> centers, const Options& o) {
    // KMEANS_HOST_ONLY exists only for local CPU-path tests without a target compiler.
#ifdef KMEANS_HOST_ONLY
    if (!o.cpu) throw std::runtime_error("This build is CPU-only; GPU execution is unavailable");
#else
    if (!o.cpu && omp_get_num_devices() < 1)
        throw std::runtime_error("No OpenMP target device; refusing CPU fallback");
    if (!o.cpu) {
        int on_host = 1;
        #pragma omp target map(from:on_host)
        { on_host = omp_is_initial_device(); }
        if (on_host) throw std::runtime_error("Target probe ran on CPU; refusing fallback");
    }
#endif
    const long long n = o.n;
    const int d = o.d, k = o.k, kd = k * d;
    const std::size_t nd = data.size();
    const double* x = data.data();
    double* c = centers.data();
    std::vector<int> labels_storage(static_cast<std::size_t>(n));
    int* labels = labels_storage.data();
    std::vector<double> sums(kd);
    std::vector<long long> counts(k);
    GPUResult result;
    auto setup_start = Clock::now();
#ifndef KMEANS_HOST_ONLY
    #pragma omp target data if(!o.cpu) map(to:x[0:nd]) map(alloc:c[0:kd],labels[0:n])
#endif
    {
        result.setup = seconds(setup_start);
        auto start = Clock::now();
        for (int it = 0; it < o.max_iter; ++it) {
            auto assignment_start = Clock::now();
            if (o.cpu) {
                for (long long i = 0; i < n; ++i)
                    labels[i] = nearest(x + static_cast<std::size_t>(i) * d, c, d, k);
            } else {
#ifndef KMEANS_HOST_ONLY
                #pragma omp target update to(c[0:kd])
                int on_host = 1;
                #pragma omp target teams distribute parallel for map(alloc:x[0:nd],c[0:kd],labels[0:n]) map(from:on_host)
                for (long long i = 0; i < n; ++i) {
                    if (i == 0) on_host = omp_is_initial_device();
                    int best_cluster = 0;
                    double best = 1e300;
                    for (int cluster = 0; cluster < k; ++cluster) {
                        double dist = 0;
                        for (int j = 0; j < d; ++j) {
                            double diff = x[static_cast<std::size_t>(i) * d + j] - c[cluster * d + j];
                            dist += diff * diff;
                        }
                        if (dist < best) { best = dist; best_cluster = cluster; }
                    }
                    labels[i] = best_cluster;
                }
                if (on_host) { result.offloaded = false; break; }
                #pragma omp target update from(labels[0:n])
                result.offloaded = true;
#endif
            }
            // Includes center upload, label download, launch and synchronization.
            result.assignment += seconds(assignment_start);
            std::fill(sums.begin(), sums.end(), 0.0);
            std::fill(counts.begin(), counts.end(), 0);
            for (long long i = 0; i < n; ++i) {
                int cluster = labels[i];
                ++counts[cluster];
                for (int j = 0; j < d; ++j)
                    sums[cluster * d + j] += x[static_cast<std::size_t>(i) * d + j];
            }
            double shift = update_centers(centers, sums, counts, d, k);
            result.iterations = it + 1;
            if (shift < o.tol) break;
        }
        result.time = seconds(start);
    }
    if (!o.cpu && !result.offloaded)
        throw std::runtime_error("Assignment kernel ran on CPU; refusing fallback");
    if (std::accumulate(counts.begin(), counts.end(), 0LL) != n)
        throw std::runtime_error("Cluster counts do not sum to N");
    result.objective = wcss(data, centers, n, d, k);
    result.centers = std::move(centers);
    return result;
}
