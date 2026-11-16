#include "gpu_solver.hpp"

// One standalone process / one GPU. MPI is unnecessary for this single-device case.
int main(int argc, char** argv) {
    try {
        auto o = options(argc, argv, true);
        omp_set_dynamic(0);
        omp_set_num_threads(1); // Host accumulation is deliberately serial.
        auto data = generate_dataset(static_cast<std::size_t>(o.n), o.d, o.k, o.seed);
        auto centers = initialize_centroids(data, o.d, o.k);
        auto r = solve_assignment_offload(data, centers, o);
        std::cout << std::setprecision(12)
            << "N,K,D,ranks,iterations,time_s,comm_max_s,wcss,threads,gpus,offloaded,assignment_s,device_setup_s\n"
            << o.n << ',' << o.k << ',' << o.d << ",1," << r.iterations << ',' << r.time
            << ",0," << r.objective << ",1," << (r.offloaded ? 1 : 0) << ','
            << (r.offloaded ? 1 : 0) << ',' << r.assignment << ',' << r.setup << '\n';
        if (o.centroids) print_centers(r.centers, o.d, o.k);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
