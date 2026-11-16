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
qsub -P <YOUR_PROJECT> serial_scaling.pbs
```
The script automatically compiles the serial implementation and runs the scaling experiments with different dataset sizes.

### 2. MPI K-means

**Strong Scaling**

```bash
cd ~/Parallelizing_K-means_Clustering/mpi_kmeans
qsub -P <YOUR_PROJECT> strong_scaling.pbs
```

**Weak Scaling**

```bash
cd ~/Parallelizing_K-means_Clustering/mpi_kmeans
qsub -P <YOUR_PROJECT> weak_scaling.pbs
```
Both scripts automatically compile the MPI implementation and run experiments using 1 to 256 MPI ranks, with five repetitions per configuration.
