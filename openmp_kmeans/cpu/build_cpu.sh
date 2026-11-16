#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin
mpicxx -O3 -std=c++17 -fopenmp mpi_openmp.cpp -o bin/mpi_openmp
mpicxx -O3 -std=c++17 ../../mpi_kmeans/mpi_kmeans.cpp -o bin/mpi_reference
# Use the same wrapper's C++ compiler and flags for the serial baseline.
mpicxx -O3 -std=c++17 ../../serial_kmeans/serial.cpp -o bin/serial_reference
mpicxx -O3 -std=c++17 -fopenmp -DKMEANS_HOST_ONLY ../validation.cpp -o bin/validate_cpu
./bin/validate_cpu
