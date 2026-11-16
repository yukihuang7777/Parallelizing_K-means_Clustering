
#pragma once

#include <vector>
#include <random>
#include <stdexcept>
#include <cstddef>

inline std::vector<double> generate_dataset(
    std::size_t N,
    int D,
    int K,
    unsigned int seed = 42,
    double noise_std = 0.5
) {
    if (D <= 0 || K <= 0 ||
        N < static_cast<std::size_t>(K) ||
        noise_std < 0.0) {
        throw std::invalid_argument("Invalid dataset parameters");
    }

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> noise(0.0, noise_std);

    std::vector<double> data(N * static_cast<std::size_t>(D));

    for (std::size_t i = 0; i < N; ++i) {
        int cluster = static_cast<int>(i % K);

        for (int j = 0; j < D; ++j) {
            double true_center = 5.0 * cluster + 0.5 * j;

            data[i * D + j] = true_center + noise(rng);
        }
    }

    return data;
}

inline std::vector<double> initialize_centroids(
    const std::vector<double>& data,
    int D,
    int K
) {
    if (D <= 0 || K <= 0 ||
        data.size() < static_cast<std::size_t>(D) * K) {
        throw std::invalid_argument("Invalid centroid parameters");
    }

    return std::vector<double>(
        data.begin(),
        data.begin() + static_cast<std::size_t>(K) * D
    );
}
