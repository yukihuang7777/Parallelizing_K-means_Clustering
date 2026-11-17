#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin
mpicxx -O3 -std=c++17 -fopenmp mpi_openmp.cpp -o bin/mpi_openmp
mpicxx -O3 -std=c++17 ../../mpi_kmeans/mpi_kmeans.cpp -o bin/mpi_reference
