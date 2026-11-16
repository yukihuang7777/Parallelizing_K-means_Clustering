#include "gpu/gpu_solver.hpp"

bool close(double a, double b) { return std::abs(a-b) <= 1e-9; }

int main(int argc, char** argv) {
    try {
        bool gpu = argc == 2 && std::string(argv[1]) == "--gpu";
        omp_set_dynamic(0);
        // Exercise ties, empty clusters, threads with no points, and uneven partitions.
        if (!gpu) {
          for (int t : {1, 2, 4}) {
            std::vector<double> data = {0, 2, 10, 12, 14};
            std::vector<double> centers = {0, 10};
            std::vector<double> sums(2);
            std::vector<long long> counts(2);
            ThreadStats stats(t, 1, 2);
            stats.assign(data, centers, 5, sums, counts);
            if (counts != std::vector<long long>{2,3} || sums != std::vector<double>{2,36})
                throw std::runtime_error("Thread-private assignment failed");
            std::vector<double> zeros = {0,0,0};
            centers = {0,0};
            stats.assign(zeros, centers, 3, sums, counts);
            update_centers(centers, sums, counts, 1, 2);
            if (counts != std::vector<long long>{3,0} || centers != std::vector<double>{0,0})
                throw std::runtime_error("Tie / empty cluster test failed");
            stats.assign({}, centers, 0, sums, counts);
            if (counts != std::vector<long long>{0,0})
                throw std::runtime_error("Empty local partition failed");
          }
        }
        for (bool empty : {false, true}) {
            Options o;
            o.n = empty ? 3 : 4; o.d = 1; o.k = 2; o.max_iter = 50; o.tol = 0; o.cpu = !gpu;
            std::vector<double> data = empty ? std::vector<double>{0,0,0} : std::vector<double>{0,2,10,12};
            std::vector<double> centers = empty ? std::vector<double>{0,0} : std::vector<double>{0,10};
            auto r = solve_assignment_offload(data, centers, o);
            if (r.iterations != 50 || !close(r.objective, empty ? 0 : 4) ||
                !close(r.centers[0], empty ? 0 : 1) || !close(r.centers[1], empty ? 0 : 11) ||
                r.offloaded != gpu) throw std::runtime_error("Fixed-iteration solver test failed");
            o.tol = 1e-6;
            r = solve_assignment_offload(data, centers, o);
            if (r.iterations != (empty ? 1 : 2)) throw std::runtime_error("Convergence test failed");
        }
        std::cout << "PASS: ";
        if (!gpu) std::cout << "thread-private reductions (1/2/4 threads), uneven/empty partitions, ";
        std::cout << "ties, empty clusters, known centers/WCSS, fixed 50 iterations, convergence; device="
                  << (gpu ? "GPU" : "CPU") << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
