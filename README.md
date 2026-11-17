# Parallelizing K-means Clustering
Parallelizing K-means Clustering with MPI and OpenMP


## How to Run

The experiments were conducted on the NUS Vanda HPC cluster using PBS. Clone the repository to Vanda and submit the corresponding PBS scripts.

Replace `YOUR_PROJECT` with your own HPC project name.

### Setup

Replace `YOUR_USERNAME` with your NUS HPC username. Make sure you are connected to the NUS network or VPN when accessing Vanda.

- Clone the repository to your local computer:

```bash
git clone https://github.com/yukihuang7777/Parallelizing_K-means_Clustering.git
```

- Upload the project folder to Vanda using your local terminal:

```bash
rsync -av --exclude='.git/' --exclude='.DS_Store' \
  Parallelizing_K-means_Clustering/ \
  YOUR_USERNAME@vanda.nus.edu.sg:~/Parallelizing_K-means_Clustering/
```
If you cloned the repository into a different directory, replace `Parallelizing_K-means_Clustering/` with `PATH_TO_DIRECTORY/Parallelizing_K-means_Clustering/`, where `PATH_TO_DIRECTORY` is the path to your chosen directory.


- Connect to Vanda:

```bash
ssh YOUR_USERNAME@vanda.nus.edu.sg
```

- Navigate to the project directory:

```bash
cd ~/Parallelizing_K-means_Clustering
```

### 1. Serial K-means

```bash
cd ~/Parallelizing_K-means_Clustering/serial_kmeans
qsub -P "$PROJECT" serial_scaling.pbs
```
The script automatically compiles the serial implementation and runs the scaling experiments with different dataset sizes.

### 2. MPI K-means

**Strong Scaling**

```bash
cd ~/Parallelizing_K-means_Clustering/mpi_kmeans
qsub -P "$PROJECT"  strong_scaling.pbs
```

**Weak Scaling**

```bash
cd ~/Parallelizing_K-means_Clustering/mpi_kmeans
qsub -P "$PROJECT" weak_scaling.pbs
```
Both scripts automatically compile the MPI implementation and run experiments using 1, 2, 4, 8, 16, 32, 64, 128 and 256 MPI ranks, with five repetitions per configuration.


### 3. MPI + OpenMP on CPUs

Compares pure MPI 64 x 1 with hybrid 64 x 1,
32 x 2 and 16 x 4 (ranks x threads), with five repetitions each.

```bash
cd ~/Parallelizing_K-means_Clustering/openmp_kmeans/cpu
qsub -P "$PROJECT" hybrid_all.pbs
```

### 4. Single-GPU OpenMP Offload

Offloads nearest-centroid assignment to one GPU; accumulation and center updates
remain on the CPU. Benchmarks serial, CPU pipeline
and GPU versions, with five repetitions each.

```bash
cd ~/Parallelizing_K-means_Clustering/openmp_kmeans/gpu
qsub -P "$PROJECT" gpu_single.pbs
```

### Check Jobs and Results

```bash
qstat -u "$USER"
```

CSV results are saved under each version's `results/` directory. OpenMP CPU and
GPU runs create job-specific subdirectories there. GPU result rows must show
`offloaded=1`. Submit each command once; duplicate jobs of the same type can
conflict through shared build or temporary files.