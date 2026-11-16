#include <mpi.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <limits>
#include <string>
#include "../dataset.hpp"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int P, ID;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &ID);

    long long N = (argc > 1) ? std::atoll(argv[1]) : 10000;
    int D = (argc > 2) ? std::atoi(argv[2]) : 8;
    int K = (argc > 3) ? std::atoi(argv[3]) : 10;
    unsigned int seed = (argc > 4) ? static_cast<unsigned int>(std::strtoul(argv[4], nullptr, 10)) : 42;
    int max_iter = (argc > 5) ? std::atoi(argv[5]) : 100;
    double tolerance = (argc > 6) ? std::atof(argv[6]) : 1e-6;
    bool print_centroids = (argc > 7 && std::string(argv[7]) == "--centroids");

    if (N <= 0 || K <= 0 || D <= 0 || max_iter <= 0 || tolerance < 0 ||
        K > N || static_cast<long long>(K) * D > 10000000 ||
        N > INT_MAX / static_cast<long long>(D)) {
        if (ID == 0) std::cerr << "Invalid input parameters.\n";
        MPI_Finalize();
        return 1;
    }

    long long count = N / P;
    long long remainder = N % P;
    if (ID < remainder) count += 1;
    long long lb = ID * (N / P) + std::min<long long>(ID, remainder);

    std::vector<double> full_data;
    std::vector<double> centroids(static_cast<size_t>(K) * D, 0.0);
    if (ID == 0) {
        full_data = generate_dataset(static_cast<size_t>(N), D, K, seed);
        centroids = initialize_centroids(full_data, D, K);
    }

    std::vector<int> sendcounts(P, 0), displacements(P, 0);
    for (int r = 0; r < P; r++) {
        long long rows = N / P + (r < remainder ? 1 : 0);
        long long first_row = r * (N / P) + std::min<long long>(r, remainder);
        sendcounts[r] = static_cast<int>(rows * D);
        displacements[r] = static_cast<int>(first_row * D);
    }

    std::vector<double> data(static_cast<size_t>(count) * D, 0.0);
    MPI_Scatterv(ID == 0 ? full_data.data() : nullptr,
                 sendcounts.data(), displacements.data(), MPI_DOUBLE,
                 data.data(), static_cast<int>(count * D), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Bcast(centroids.data(), K * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double>().swap(full_data);

    std::vector<double> local_sums(static_cast<size_t>(K) * D, 0.0);
    std::vector<double> global_sums(static_cast<size_t>(K) * D, 0.0);
    std::vector<long long> local_counts(K, 0), global_counts(K, 0);

    MPI_Barrier(MPI_COMM_WORLD);
    auto begin = std::chrono::steady_clock::now();
    double comm_time = 0.0;
    int iterations_done = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        std::fill(local_sums.begin(), local_sums.end(), 0.0);
        std::fill(local_counts.begin(), local_counts.end(), 0);

        for (long long i = 0; i < count; i++) {
            int closest = 0;
            double min_dist = 1e300;
            for (int k = 0; k < K; k++) {
                double dist = 0.0;
                for (int d = 0; d < D; d++) {
                    double diff = data[static_cast<size_t>(i) * D + d]
                                - centroids[k * D + d];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    closest = k;
                }
            }
            local_counts[closest]++;
            for (int d = 0; d < D; d++) {
                local_sums[closest * D + d] += data[static_cast<size_t>(i) * D + d];
            }
        }

        auto comm_begin = std::chrono::steady_clock::now();
        MPI_Allreduce(local_sums.data(), global_sums.data(), K * D,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts.data(), global_counts.data(), K,
                      MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
        auto comm_end = std::chrono::steady_clock::now();
        comm_time += std::chrono::duration<double>(comm_end - comm_begin).count();

        double max_shift_squared = 0.0;
        for (int k = 0; k < K; k++) {
            if (global_counts[k] == 0) continue;
            double shift_squared = 0.0;
            for (int d = 0; d < D; d++) {
                int j = k * D + d;
                double updated = global_sums[j] / global_counts[k];
                double diff = updated - centroids[j];
                shift_squared += diff * diff;
                centroids[j] = updated;
            }
            max_shift_squared = std::max(max_shift_squared, shift_squared);
        }
        iterations_done = iter + 1;
        if (std::sqrt(max_shift_squared) < tolerance) break;
    }

    auto end = std::chrono::steady_clock::now();
    double runtime = std::chrono::duration<double>(end - begin).count();

    double total_max = 0.0, comm_max = 0.0;
    MPI_Reduce(&runtime, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double local_wcss = 0.0, global_wcss = 0.0;
    for (long long i = 0; i < count; i++) {
        double min_dist = 1e300;
        for (int k = 0; k < K; k++) {
            double dist = 0.0;
            for (int d = 0; d < D; d++) {
                double diff = data[static_cast<size_t>(i) * D + d]
                            - centroids[k * D + d];
                dist += diff * diff;
            }
            min_dist = std::min(min_dist, dist);
        }
        local_wcss += min_dist;
    }
    MPI_Reduce(&local_wcss, &global_wcss, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (ID == 0) {
        std::cout << std::setprecision(12);
        std::cout << "N,K,D,ranks,iterations,time_s,comm_max_s,wcss\n";
        std::cout << N << "," << K << "," << D << "," << P << ","
                  << iterations_done << "," << total_max << ","
                  << comm_max << "," << global_wcss << "\n";
        if (print_centroids) {
            std::cerr << std::setprecision(12);
            for (int k = 0; k < K; ++k) {
                std::cerr << "Centroid " << k << ":";
                for (int d = 0; d < D; ++d)
                    std::cerr << " " << centroids[k * D + d];
                std::cerr << "\n";
            }
        }
    }

    MPI_Finalize();
    return 0;
}
