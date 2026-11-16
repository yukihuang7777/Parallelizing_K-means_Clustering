
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <climits>

#include "../dataset.hpp"

// Arguments match MPI: [N] [D] [K] [seed] [max_iter] [tolerance] [--centroids]
// Strong-scaling baseline: ./serial 8000000 8 10 42 50 0
// tolerance=0 disables early stopping, matching the MPI benchmark scripts.

struct KMeansResult {
    std::vector<double> centroids;
    double sse;
    int iterations;
    double runtime;
};

KMeansResult serial_kmeans(
    const std::vector<double>& data,
    std::vector<double> centroids,
    std::size_t N,
    int D,
    int K,
    int max_iter = 100,
    double tol = 1e-6
) {
    int iterations = 0;
    std::vector<double> sums(static_cast<std::size_t>(K) * D, 0.0);
    std::vector<long long> counts(K, 0);

    // Start timing the K-means iterations
    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < max_iter; ++iter) {
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);

        // Assignment step
        for (std::size_t i = 0; i < N; ++i) {
            int best_cluster = 0;
            double min_dist = std::numeric_limits<double>::max();

            for (int k = 0; k < K; ++k) {
                double dist = 0.0;

                for (int j = 0; j < D; ++j) {
                    double diff = data[i * D + j]
                                - centroids[k * D + j];
                    dist += diff * diff;
                }

                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }
            }

            counts[best_cluster]++;

            for (int j = 0; j < D; ++j) {
                sums[best_cluster * D + j] += data[i * D + j];
            }
        }

        // Update centroids
        double max_shift_squared = 0.0;

        for (int k = 0; k < K; ++k) {
            if (counts[k] == 0) {
                continue;
            }

            double shift_squared = 0.0;

            for (int j = 0; j < D; ++j) {
                std::size_t idx = k * D + j;

                double new_center = sums[idx] / counts[k];
                double diff = new_center - centroids[idx];

                shift_squared += diff * diff;
                centroids[idx] = new_center;
            }

            if (shift_squared > max_shift_squared) {
                max_shift_squared = shift_squared;
            }
        }

        iterations = iter + 1;

        if (std::sqrt(max_shift_squared) < tol) {
            break;
        }
    }

    auto end = std::chrono::steady_clock::now();

    double runtime =
        std::chrono::duration<double>(end - start).count();

    // Calculate final SSE
    double sse = 0.0;

    for (std::size_t i = 0; i < N; ++i) {
        double min_dist = std::numeric_limits<double>::max();

        for (int k = 0; k < K; ++k) {
            double dist = 0.0;

            for (int j = 0; j < D; ++j) {
                double diff = data[i * D + j]
                            - centroids[k * D + j];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
            }
        }

        sse += min_dist;
    }

    return {centroids, sse, iterations, runtime};
}

int main(int argc, char* argv[]) {
    try {
        long long input_N = 10000;
        int D = 8;
        int K = 10;
        unsigned int seed = 42;
        int max_iter = 100;
        double tolerance = 1e-6;
        bool print_centroids = false;

        if (argc > 1) input_N = std::stoll(argv[1]);
        if (argc > 2) D = std::stoi(argv[2]);
        if (argc > 3) K = std::stoi(argv[3]);
        if (argc > 4) seed = std::stoul(argv[4]);
        if (argc > 5) max_iter = std::stoi(argv[5]);
        if (argc > 6) tolerance = std::stod(argv[6]);
        if (argc > 7) {
            if (std::string(argv[7]) != "--centroids") {
                throw std::invalid_argument("Expected --centroids as the final argument");
            }
            print_centroids = true;
        }
        if (argc > 8 || input_N <= 0 || D <= 0 || K <= 0 ||
            max_iter <= 0 || !std::isfinite(tolerance) || tolerance < 0 ||
            K > input_N || static_cast<long long>(K) * D > 10000000 ||
            input_N > INT_MAX / static_cast<long long>(D)) {
            throw std::invalid_argument("Invalid input parameters");
        }
        // Use the same problem-size limits as the MPI implementation.
        const std::size_t N = static_cast<std::size_t>(input_N);

        auto data = generate_dataset(N, D, K, seed);
        auto initial = initialize_centroids(data, D, K);

        KMeansResult result =
            serial_kmeans(data, initial, N, D, K, max_iter, tolerance);

        std::cout << std::setprecision(12);

        // Same schema as MPI; serial uses one process and has no communication.
        std::cout << "N,K,D,ranks,iterations,time_s,comm_max_s,wcss\n";
        std::cout << N << "," << K << "," << D << ",1,"
                  << result.iterations << "," << result.runtime << ",0,"
                  << result.sse << "\n";

        if (print_centroids) {
            std::cerr << std::setprecision(12);
            for (int k = 0; k < K; ++k) {
                std::cerr << "Centroid " << k << ":";
                for (int j = 0; j < D; ++j) {
                    std::cerr << " " << result.centroids[k * D + j];
                }
                std::cerr << "\n";
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
