#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin
nvc++ -O3 -std=c++17 -mp=gpu gpu_openmp.cpp -o bin/gpu_openmp
nvc++ -O3 -std=c++17 -mp=gpu ../validation.cpp -o bin/validate_gpu
# Matching standard library / random generator for the GPU comparison.
nvc++ -O3 -std=c++17 ../../serial_kmeans/serial.cpp -o bin/serial_gpu_reference
