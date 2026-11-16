#!/bin/bash
# Source from a PBS script, after its initial SCRIPT_STARTED message.
set -Eeuo pipefail
cd "${PBS_O_WORKDIR:?Submit from openmp_kmeans/cpu}"
test -f mpi_openmp.cpp
test -f ../compare.py
mkdir -p results
export RUN_DIR="$PWD/results/${PBS_JOBNAME}_${PBS_JOBID}"
mkdir -p "$RUN_DIR"
echo "Detailed log: $RUN_DIR/job.log"
exec > >(tee -a "$RUN_DIR/job.log") 2>&1
trap 'rc=$?; echo "JOB_EXIT code=$rc time=$(date -Is)"' EXIT
trap 'rc=$?; echo "FAILED line=$LINENO command=$BASH_COMMAND code=$rc"' ERR
echo "STAGE 1: shell and shared directory available"
date -Is
hostname
pwd
id
printf 'Nodefile:\n'
sort "$PBS_NODEFILE" | uniq -c
nodes=$(sort -u "$PBS_NODEFILE" | wc -l)
if [ "$nodes" -lt 1 ] || [ "$nodes" -gt 2 ] || [ "$(wc -l < "$PBS_NODEFILE")" -ne 64 ]; then
    echo "Expected 64 PBS MPI slots on one or two hosts; got $nodes host(s)"
    exit 1
fi
export KMEANS_HOST_COUNT="$nodes"
# Save PBS's effective allocation, including any site-rewritten resources.
qstat -f "$PBS_JOBID" > "$RUN_DIR/allocation.txt" 2>&1 || true
echo "STAGE 2: compiler and MPI environment"
module load mpich/ge/gcc/64/4.1.1
module list
mpicxx -show
python3 --version
command -v timeout
export FI_PROVIDER=tcp
export OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=close
export OMP_NUM_THREADS=1
# MPICH/Hydra-specific bounded launch time; no background retry loops.
export MPIEXEC_TIMEOUT=180
export KMEANS_RUN_TIMEOUT=240
unset MPIEXEC
echo "STAGE 3: one MPI process per host (no explicit CPU binding yet)"
if timeout 180s mpirun -np "$KMEANS_HOST_COUNT" -ppn 1 sh -c '
    echo "HOST=$(hostname)"
    test -d "$PBS_O_WORKDIR/results" || exit 1
    test -w "$PBS_O_WORKDIR/results" || exit 1
    lscpu
    taskset -pc $$
' > "$RUN_DIR/hosts.txt" 2>&1; then
    cat "$RUN_DIR/hosts.txt"
else
    rc=$?
    echo "MPI host probe failed (exit=$rc). See launcher output below."
    cat "$RUN_DIR/hosts.txt"
    exit "$rc"
fi
echo "STAGE 4: compile and validate local kernels inside allocation"
bash build_cpu.sh
