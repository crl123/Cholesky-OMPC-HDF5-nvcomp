# Cholesky Factorization on Heterogeneous Clusters
This project implements a high-performance, out-of-core Cholesky factorization algorithm. It is designed to efficiently handle large, dense matrices that exceed a single node's memory capacity. The solution leverages a combination of MPI, OpenMP, CUDA, and HDF5 to distribute computational tasks and manage data across a cluster of nodes equipped with GPUs.
# Key Features
* **Out-of-Core Execution:** Processes matrices larger than available RAM by storing blocks on disk using the HDF5 library.
* **Hybrid Parallelism:** Combines MPI for inter-node communication and OpenMP for offloading computational tasks to multiple GPUs within a single node.
* **GPU Acceleration:** Utilizes NVIDIA's CUDA Toolkit, cuBLAS, and cuSOLVER libraries for highly optimized linear algebra operations.
* **Data Compression:** Employs nvcomp for on-the-fly compression and decompression of matrix blocks, reducing disk I/O and storage requirements.
# Requirements
To build and run this project, you need the following dependencies installed on your system.
* OmpCluster Runtime
* **CUDA Toolkit:** The NVIDIA CUDA Toolkit (version 11.0).
* **nvcomp:** NVIDIA's GPU-accelerated compression library (version 4.2.11).
* **HDF5:** A version of the HDF5 library with parallel I/O support (version 1.14.6 or newer).
# Building the Project
Follow these steps to build the project. The build process uses CMake to configure the project with the necessary library paths.
```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
mkdir build
cd build
export CC=clang
export CXX=clang++
cmake .. \
    -DCMAKE_PREFIX_PATH=/usr/lib;/usr/local;/opt/libraries/ \
    -Dnvcomp_DIR=/usr/local/nvcomp/lib/cmake/nvcomp/ \
    -DZSTD_ROOT=/usr/local/zstd/
```
# How to Run
After building, you can run the executable using the srun command.
```
export LIBOMP_NUM_HIDDEN_HELPER_THREADS=<num_hidden>
export OMP_NUM_THREADS=<omp_num>
export OMPCLUSTER_NUM_EXEC_EVENT_HANDLERS=<num_exec>
srun --mpi=pmi2 -N $SLURM_JOB_NUM_NODES singularity ./cholesky <dim> <nb> <data_path> <r> <workers>
```
**Parameter	Description**
<table>
  <tr>
    <td> ``<dim>` </td>
    <td> The total dimension of the square matrix. </td>
  </tr>
  <tr>
    <td> `<nb>` </d>
    <td> The block size.</td>
  </tr>
  <tr>
    <td> `<data_path>` </td>
    <td> The file path for the HDF5 matrix data example /home/usr/dataset/prove.h5.</td>
  </tr>
  <tr>
    <td> `<r>` </td>
    <td> Represents the side length of the Square Block Cyclical (SBC) grid. </td>
  </tr>
  <tr>
    <td> `<workers>` </td>
    <td> The number of OMPC worker nodes. </td>
  </tr>
</table>
	  
**Runtime Configuration**
<table>
  <tr>
    <td>`<num_hidden>` </td>
    <td> `<num_exec>`</td>
  </tr>
  <tr>
    <td> Specifies the number of helper threads managed by the head node to manage the work on the worker nodes. </td>
    <td> Determines the number of event handler threads per worker node. These threads are responsible for processing offloading events and managing GPU kernel launches.</td>
  </tr>
</table>   
For more detailed information on these parameters, please refer to the official [OMPCluster documentation] (https://ompcluster.readthedocs.io/en/latest/).
make -j48
For more detailed information on these parameters, please refer to the official OMPCluster documentation.
