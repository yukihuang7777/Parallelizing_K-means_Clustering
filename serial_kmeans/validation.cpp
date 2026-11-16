
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <exception>

// Reuse the existing serial K-means implementation.
// Rename its main function to avoid duplicate main().
#define main serial_program_main
#include "serial.cpp"
#undef main

bool close_enough(double a, double b, double tol = 1e-9) {
    return std::isfinite(a) && std::isfinite(b)
        && std::abs(a - b) <= tol;
}

// Test 1: Known centroids and SSE
bool test_known_answer() {
    std::vector<double> data = {
         0.0,  0.0,
         0.0,  2.0,
        10.0, 10.0,
        10.0, 12.0
    };

    std::vector<double> initial = {
         0.0,  0.0,
        10.0, 10.0
    };

    auto result = serial_kmeans(data, initial, 4, 2, 2);

    std::vector<double> expected = {
         0.0,  1.0,
        10.0, 11.0
    };

    bool ok = close_enough(result.sse, 4.0)
           && result.iterations == 2;

    for (std::size_t i = 0; i < expected.size(); ++i) {
        ok = ok && close_enough(
            result.centroids[i], expected[i]
        );
    }

    return ok;
}

// Test 2: Empty cluster handling
bool test_empty_cluster() {
    // Two identical initial centroids.
    // Ties are assigned to the first cluster.
    std::vector<double> data = {
        0.0, 0.0, 0.0
    };

    std::vector<double> initial = {
        0.0, 0.0
    };

    auto result = serial_kmeans(data, initial, 3, 1, 2);

    return close_enough(result.centroids[0], 0.0)
        && close_enough(result.centroids[1], 0.0)
        && close_enough(result.sse, 0.0)
        && result.iterations >= 1;
}

// Test 3: Reproducibility
bool test_reproducibility() {
    auto data1 = generate_dataset(120, 2, 3, 42);
    auto data2 = generate_dataset(120, 2, 3, 42);

    if (data1 != data2) {
        return false;
    }

    auto initial1 = initialize_centroids(data1, 2, 3);
    auto initial2 = initialize_centroids(data2, 2, 3);

    auto r1 = serial_kmeans(data1, initial1, 120, 2, 3);
    auto r2 = serial_kmeans(data2, initial2, 120, 2, 3);

    return r1.centroids == r2.centroids
        && r1.sse == r2.sse
        && r1.iterations == r2.iterations;
}

// A converged dataset must still run all requested benchmark iterations at tol=0.
bool test_fixed_iterations() {
    std::vector<double> data = {0.0, 2.0, 10.0, 12.0};
    std::vector<double> initial = {0.0, 10.0};
    auto result = serial_kmeans(data, initial, 4, 1, 2, 50, 0.0);
    return result.iterations == 50
        && close_enough(result.sse, 4.0)
        && close_enough(result.centroids[0], 1.0)
        && close_enough(result.centroids[1], 11.0);
}

int main() {
    int passed = 0;
    const int total = 4;

    auto run = [&](const std::string& name, bool (*test)()) {
        bool ok = test();

        std::cout << name << ": "
                  << (ok ? "PASS" : "FAIL") << "\n";

        if (ok) {
            ++passed;
        }
    };

    run("Known Centroids and SSE", test_known_answer);
    run("Empty Cluster", test_empty_cluster);
    run("Reproducibility", test_reproducibility);
    run("Fixed 50 Iterations", test_fixed_iterations);

    std::cout << "\nTests passed: "
              << passed << "/" << total << "\n";

    return passed == total ? 0 : 1;
}
