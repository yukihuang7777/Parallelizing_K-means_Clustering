#include <mpi.h>
#include "../common.hpp"

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED unavailable");
        const auto o = options(argc, argv);
        int threads = actual_threads(), min_t = 0, max_t = 0;
        MPI_Allreduce(&threads, &min_t, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&threads, &max_t, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (min_t != max_t) throw std::runtime_error("Ranks have different thread counts");
        const int kd = o.k * o.d;
        long long n = o.n / ranks + (rank < o.n % ranks ? 1 : 0);
        std::vector<double> full, centers(kd);
        if (rank == 0) {
            full = generate_dataset(static_cast<std::size_t>(o.n), o.d, o.k, o.seed);
            centers = initialize_centroids(full, o.d, o.k);
        }
        std::vector<int> sizes(ranks), offsets(ranks);
        for (int r = 0; r < ranks; ++r) {
            sizes[r] = static_cast<int>((o.n / ranks + (r < o.n % ranks ? 1 : 0)) * o.d);
            offsets[r] = static_cast<int>((r * (o.n / ranks) + std::min<long long>(r, o.n % ranks)) * o.d);
        }
        std::vector<double> data(static_cast<std::size_t>(n) * o.d);
        MPI_Scatterv(rank == 0 ? full.data() : nullptr, sizes.data(), offsets.data(), MPI_DOUBLE,
                     data.data(), static_cast<int>(n * o.d), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(centers.data(), kd, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        std::vector<double>().swap(full);
        ThreadStats stats(threads, o.d, o.k);
        std::vector<double> sums(kd), global_sums(kd);
        std::vector<long long> counts(o.k), global_counts(o.k);
        int iterations = 0;
        double comm = 0;
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = Clock::now();
        for (int it = 0; it < o.max_iter; ++it) {
            stats.assign(data, centers, n, sums, counts);
            auto cs = Clock::now();
            MPI_Allreduce(sums.data(), global_sums.data(), kd, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(counts.data(), global_counts.data(), o.k, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
            comm += seconds(cs);
            double shift = update_centers(centers, global_sums, global_counts, o.d, o.k);
            iterations = it + 1;
            if (shift < o.tol) break;
        }
        double elapsed = seconds(start), max_time = 0, max_comm = 0;
        MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comm, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        // Validation is outside the benchmark timer.
        if (std::accumulate(global_counts.begin(), global_counts.end(), 0LL) != o.n)
            throw std::runtime_error("Cluster counts do not sum to N");
        double local = wcss(data, centers, n, o.d, o.k), total = 0;
        MPI_Reduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << std::setprecision(12)
                << "N,K,D,ranks,iterations,time_s,comm_max_s,wcss,threads,gpus,offloaded\n"
                << o.n << ',' << o.k << ',' << o.d << ',' << ranks << ',' << iterations
                << ',' << max_time << ',' << max_comm << ',' << total << ',' << threads << ",0,0\n";
            if (o.centroids) print_centers(centers, o.d, o.k);
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << ": " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
