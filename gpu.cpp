/******************************************************************************
 * Cholesky + GPU + HDF5                 			      	      *
 * Author: Carla Cardoso Cusihuallpa					      *
 *						      			      *
 * This example demonstrates the execution of the Cholesky factorization      *
 * using the OMPC runtime to distribute tasks across MPI processes,           *
 * CUDA routines to perform linear algebra operations on the GPU.             *
 * HDF5 is also employed to store matrix blocks on disk efficiently,          * 
 * enabling out-of-core execution and scalable data management.               *
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <string>

#pragma omp declare target
extern "C" {
  #include "hdf5.h"
}
#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#pragma omp end declare target

/**
 * @brief Initializes a matrix block with a specific pattern.
 *
 * This function fills a matrix block of size nx * ny with a structured pattern
 * based on its global position (block_i, block_j) and the overall matrix dimensions.
 * For diagonal blocks (i == j), it ensures the matrix is symmetric and diagonally
 * dominant to guarantee a successful Cholesky factorization.
 *
 * @param nx The width of the block.
 * @param ny The height of the block.
 * @param block_i The row index of the block in the global matrix.
 * @param block_j The column index of the block in the global matrix.
 * @param N The total dimension of the matrix.
 * @param BS The block size.
 * @param array A pointer to the block data to be filled.
 */
void FillBlock(int nx, int ny, int ng, int block_i, int block_j, int N, int BS, float **array) {
    int init = block_i*N*BS+block_j*BS;
    int i, j;
    for (j = 0; j < ny+2*ng; j++) {
        for (i = 0; i < nx+2*ng; i++) {
              array[j][i] = 0.0;
        }
    }
  
    for (j = ng; j < ny+ng; j++) {
        for (i = ng; i < nx+ng; i++) {
              array[j][i] = (float) (init+(i-ng)+(j-ng)*N)/((float) N*N);
        }
    } 

    if (block_i == block_j) {
        for (long i = ng; i < ny+ng; i++) {
            array[i][i] = array[i][i] + N;
            for (long j = ng; j < i; j++) {
                array[i][j] = array[j][i];
            }
        }
    }

}

/**
 * @brief Creates an HDF5 file on the file system.
 *
 * This function initializes an HDF5 file to store compressed matrix blocks. It
 * uses MPI parallel I/O settings to ensure efficient access in high-performance
 * computing environments. The function only creates the file and the main
 * dataset for subsequent writing operations.
 *
 * @param tmp The file path for the HDF5 file to be created.
 * @param p The number of processes in the x-dimension of the process grid.
 * @param q The number of processes in the y-dimension of the process grid.
 * @param dim The total dimension of the square matrix.
 * @param nb The size of a matrix block (tile).
 */
void CreateFile(const char* tmp, int p, int q,long dim, long nb){
    int nro_devices = (p+1)*q;
    int vector[nro_devices];
    int64_t OMP_FILE_ACCESS= H5P_FILE_ACCESS;
    int len = strlen(tmp);

    #pragma omp parallel
    #pragma omp single
    {
        for(int i=0;i<nro_devices;i++)
        {
            vector[i] = i;
            #pragma omp target nowait firstprivate(OMP_FILE_ACCESS) \
                firstprivate(len,i,dim,nb) \
                map(to:tmp[:len])\
                depend(in:vector[i]) device(i)
            {
                int nx, ny, mx, my;

                nx = nb;
                ny = nb;
                int ng = 2;

                int rank;
                hsize_t dimens_2d[2];
                hsize_t chunk_dims[2];
                hid_t err;

                hid_t dataspace, dataset;
                hid_t acc_template;

                // define an info object to store MPI-IO information 
                hid_t file_identifier;

                int MyPE, NumPEs;
                int ierr;
                            
                ierr = MPI_Comm_size(MPI_COMM_WORLD, &NumPEs);
                ierr = MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);

                mx = dim;
                my = dim;

                acc_template = H5Pcreate(OMP_FILE_ACCESS);

                ierr = H5Pset_fapl_mpio(acc_template, MPI_COMM_WORLD, MPI_INFO_NULL);

                file_identifier = H5Fcreate(tmp, H5F_ACC_TRUNC, H5P_DEFAULT,acc_template);

                ierr = H5Pclose(acc_template);

                rank = 2;

		int nro_blocks = dim/nb;
		int nro_tile = (nro_blocks+1)*nro_blocks/2;

                dimens_2d[0] = ((nro_tile+nro_blocks-1)/nro_blocks)*nx;
                dimens_2d[1] = my;

                dataspace = H5Screate_simple(rank, dimens_2d, NULL);

                acc_template = H5Pcreate(H5P_DATASET_CREATE);
                float fill_value = 0; 
                H5Pset_fill_value(acc_template, H5T_NATIVE_FLOAT, &fill_value);
                chunk_dims[0] = ny;
                chunk_dims[1] = nx;
                H5Pset_chunk(acc_template, rank, chunk_dims);

                dataset = H5Dcreate(file_identifier, "data", H5T_NATIVE_FLOAT,dataspace, H5P_DEFAULT,acc_template,H5P_DEFAULT);
                
                ierr = H5Pclose(acc_template);
                H5Sclose(dataspace);
                H5Dclose(dataset);

                H5Fclose(file_identifier);
            }
        }
    }
}

/**
 * @brief Generates and compresses matrix blocks, then writes them to an HDF5 file.
 *
 * This function is responsible for creating the initial dataset. It generates
 * each matrix block, and writes it to the
 * previously created HDF5 file. The operations are distributed via OpenMP
 * to multiple GPU devices for parallelism.
 *
 * @param tmp The file path for the HDF5 file where the data will be written.
 * @param dim The total dimension of the matrix.
 * @param nb The block size.
 * @param dev A grid that maps blocks to specific devices.
 * @param p The number of processes in the x-dimension of the process grid.
 * @param q The number of processes in the y-dimension of the process grid.
 * @param r The size of the process grid.
 * @param nro_size An array to store the size of each compressed block.
 */
void GenerateBlocks(const char *tmp,long dim, long nb, int ** dev, int p, int q,int r){
    int nro_blocks = dim/nb;
    int len = strlen(tmp);
    int *vector = (int *)malloc(p * q *sizeof(int));
    int64_t OMP_FILE_ACCESS= H5P_FILE_ACCESS;

    int nro_devices = (p+1)*q;
    hid_t file_id[nro_devices];

    #pragma omp parallel
    #pragma omp single
    {
        for(int i=0;i<nro_devices;i++)
        {
            #pragma omp target nowait firstprivate(OMP_FILE_ACCESS) \
                firstprivate(len,i,dim,nb) \
                map(tofrom:file_id[i:1]) map(to:tmp[:len])\
                depend(inout:vector[i]) device(i)
            {
                int MyPE, NumPEs;
                int  ierr;
                ierr = MPI_Comm_size(MPI_COMM_WORLD, &NumPEs);
                ierr = MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
                hid_t acc_template;
                MPI_Info FILE_INFO_TEMPLATE;

                ierr = MPI_Info_create(&FILE_INFO_TEMPLATE);

                acc_template = H5Pcreate(OMP_FILE_ACCESS);

                ierr = H5Pset_fapl_mpio(acc_template, MPI_COMM_WORLD, FILE_INFO_TEMPLATE);
		std::string filename(tmp, len);
                file_id[i] =H5Fopen(filename.c_str(),H5F_ACC_RDWR,acc_template);

                ierr = H5Pclose(acc_template);
                ierr = MPI_Info_free(&FILE_INFO_TEMPLATE);
            }
        }
        for(long i=0; i< nro_blocks;i++)
        {
            for(long j=0; j < i+1;j++)
            {
                int node_local = dev[i%r][j%r];

                #pragma omp target nowait firstprivate(OMP_FILE_ACCESS) \
                    firstprivate(len,i,j,node_local,dim,nb) \
                    map(to:tmp[:len]) map(to:file_id[node_local:1])\
                    depend(inout:vector[node_local]) \
                    device(node_local)
                {
                    int MyPE, NumPEs;
                    int  ierr;
                    ierr = MPI_Comm_size(MPI_COMM_WORLD, &NumPEs);
                    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);

                    int r, nx, ny, mx, my, ng;

                    nx = nb;
                    ny = nb;
                    mx = dim;
                    my = dim;
                    ng = 2;

                    float *dataArray;
                    float **data;

                    dataArray = (float *) malloc(sizeof(float)*(nx+2*ng)*(ny+2*ng));
                    data = (float **) malloc(sizeof(float *)*(ny+2*ng));

                    for (r = 0; r < ny+2*ng; r++) {
                        data[r] = dataArray + r*(nx+2*ng);
                    }

                    FillBlock(nx, ny, ng, i, j, dim, nb, data);

                    int rank;
                    hsize_t dimens_2d[2];
                    hsize_t chunk_dims[2];
                    hid_t err;

                    hsize_t start_2d[2];
                    hsize_t stride_2d[2];
                    hsize_t count_2d[2];

                    hid_t dataspace, memspace, dataset;
                    hid_t acc_template, plist_id;

                    // Get identifier 
                    dataset = H5Dopen(file_id[node_local], "/data", H5P_DEFAULT); 
                    dataspace =  H5Dget_space(dataset);

		    int nro_blocks = dim/nb;
		    int id_blocks = j*nro_blocks+i-(j*(j+1)/2);
		    int i_ = id_blocks / nro_blocks;
		    int j_ = id_blocks - i_*nro_blocks; 

                    start_2d[0] = i_*ny;
                    start_2d[1] = j_*nx;

                    stride_2d[0] = 1;
                    stride_2d[1] = 1;

                    count_2d[0] = ny;
                    count_2d[1] = nx;

                    err = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start_2d, stride_2d, count_2d, NULL);

                    rank =2;
                    dimens_2d[0] = ny+2*ng;
                    dimens_2d[1] = nx+2*ng;

                    memspace = H5Screate_simple(rank, dimens_2d, NULL);
                    
                    start_2d[0] = ng;
                    start_2d[1] = ng;

                    stride_2d[0] = 1;
                    stride_2d[1] = 1;

                    count_2d[0] = ny;
                    count_2d[1] = nx;

                    err = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, start_2d, stride_2d, count_2d, NULL);

                    acc_template = H5Pcreate(H5P_DATASET_XFER);
                    ierr = H5Pset_dxpl_mpio(acc_template, H5FD_MPIO_INDEPENDENT);
                    
                    err = H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, dataspace,acc_template, &(data[0][0]));

                    H5Pclose(acc_template);
                    H5Sclose(memspace);

                    H5Sclose(dataspace);
                    H5Dclose(dataset);

                    free (data);
                    free (dataArray);
                }
            }
        }
        for(int i=0;i<nro_devices;i++)
        {
            #pragma omp target nowait \
                firstprivate(i) \
                map(tofrom:file_id[i:1])\
                depend(in:vector[i]) device(i)
            {
                H5Fclose(file_id[i]);              
            }
        }

    }

}

/**
 * @brief Reads a compressed block from an HDF5 file into host memory.
 *
 * This function reads a specific compressed data block from the HDF5 file
 * using the given block coordinates. It reads the complete block into a host
 * memory buffer for subsequent transfer or decompression.
 *
 * @param dim The total dimension of the matrix.
 * @param nb The block size.
 * @param i The row index of the block to read.
 * @param j The column index of the block to read.
 * @param file_id The HDF5 file identifier.
 * @param compressed_size The size of the compressed block.
 * @param compressed_data_out A pointer to the pointer where the read data will be stored.
 */
void ReadBlock(long dim, long nb, int i, int j, hid_t file_id, float* data){
    hid_t acc_template;
    hid_t dataspace, memspace, dataset;
    hsize_t start_2d[2];
    hsize_t stride_2d[2];
    hsize_t count_2d[2];
                
    int rank = 2, ierr;
                
    dataset = H5Dopen(file_id, "/data", H5P_DEFAULT); 

    dataspace = H5Dget_space(dataset);

    int nro_blocks = dim/nb;
    int id_blocks = j*nro_blocks+i-(j*(j+1)/2);
    int i_ = id_blocks / nro_blocks;
    int j_ = id_blocks - i_*nro_blocks;

    start_2d[0] = i_*nb;
    start_2d[1] = j_*nb;

    stride_2d[0] = 1;
    stride_2d[1] = 1;

    count_2d[0] = nb;
    count_2d[1] = nb;

    memspace = H5Screate_simple(rank, count_2d, NULL);

    ierr = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start_2d, stride_2d, count_2d, NULL);

    acc_template = H5Pcreate(H5P_DATASET_XFER);
    ierr = H5Pset_dxpl_mpio(acc_template, H5FD_MPIO_INDEPENDENT);
                    
    H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, acc_template, data);

    H5Pclose(acc_template);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
}

/**
 * @brief Writes a compressed block to an HDF5 file.
 *
 * This function takes a compressed data block from a memory buffer and writes
 * it to the HDF5 file at the location of the specified block (i,j). It handles
 * the exact size of the compressed data and any necessary padding for the write operation.
 *
 * @param dim The total dimension of the matrix.
 * @param nb The block size.
 * @param i The row index of the block to write.
 * @param j The column index of the block to write.
 * @param file_id The HDF5 file identifier.
 * @param compressed_data A pointer to the pointer holding the compressed data to be written.
 * @param compressed_size The size of the compressed data.
 */
void WriteBlock(long dim, long nb, int i, int j, hid_t file_id, float* data){

    int rank, ng =2 ,ierr;
    hsize_t start_2d[2];
    hsize_t stride_2d[2];
    hsize_t count_2d[2];

    hsize_t dimens_2d[2];
    hid_t dataspace, memspace, dataset;
    hid_t err;
    hid_t acc_template, plist_id;

    // Get identifier 
    dataset = H5Dopen(file_id, "/data", H5P_DEFAULT); 

    dataspace =  H5Dget_space(dataset);

    int nro_blocks = dim/nb;
    int id_blocks = j*nro_blocks+i-(j*(j+1)/2);
    int i_ = id_blocks / nro_blocks;
    int j_ = id_blocks - i_*nro_blocks;

    start_2d[0] = i_*nb;
    start_2d[1] = j_*nb;

    stride_2d[0] = 1;
    stride_2d[1] = 1;

    count_2d[0] = nb;
    count_2d[1] = nb;

    err = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start_2d, stride_2d, count_2d, NULL);

    rank =2;
    dimens_2d[0] = nb;
    dimens_2d[1] = nb;

    memspace = H5Screate_simple(rank, dimens_2d, NULL);
                    
    start_2d[0] = 0;
    start_2d[1] = 0;

    stride_2d[0] = 1;
    stride_2d[1] = 1;

    count_2d[0] = nb;
    count_2d[1] = nb;

    err = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, start_2d, stride_2d, count_2d, NULL);

    acc_template = H5Pcreate(H5P_DATASET_XFER);
    ierr = H5Pset_dxpl_mpio(acc_template, H5FD_MPIO_INDEPENDENT);                    
    err = H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, acc_template, &(data[0]));

    H5Pclose(acc_template);
    H5Sclose(memspace);

    H5Sclose(dataspace);
    H5Dclose(dataset);

}

/**
 * @brief Executes the out-of-core Cholesky factorization on a large matrix.
 *
 * This is the main loop of the Cholesky algorithm. It performs the factorization
 * by blocks, reading compressed data from HDF5, performing
 * linear algebra calculations on the GPU, and write back to disk.
 *
 * @param a A pointer to a mapping matrix for distributing blocks to devices.
 * @param tmp The file path for the HDF5 file.
 * @param dim The dimension of the matrix.
 * @param nb The block size.
 * @param p The number of processes in the x-dimension of the process grid.
 * @param q The number of processes in the y-dimension of the process grid.
 * @param r The size of the process grid.
 * @param nro_size An array containing the compressed sizes of each block.
 */
void Cholesky_Factorization(float * a,const char * tmp, long dim, long nb, int p, int q,int r){
    long nro_blocks = dim/nb;
    int nro_devices = (p+1)*q;
    hid_t file_id[nro_devices];
    int len = strlen(tmp);
    int64_t OMP_FILE_ACCESS= H5P_FILE_ACCESS;
    int *vector = (int *)malloc(p * q *sizeof(int));

    #pragma omp parallel
    #pragma omp master
    {
        for(int i=0;i<nro_devices;i++)
        {
            #pragma omp target nowait firstprivate(OMP_FILE_ACCESS) \
                firstprivate(len,i,dim,nb) \
                map(tofrom:file_id[i:1]) map(to:tmp[:len])\
                depend(inout:vector[i]) device(i)
            {
                int MyPE, NumPEs;
                int  ierr;
                ierr = MPI_Comm_size(MPI_COMM_WORLD, &NumPEs);
                ierr = MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
                hid_t acc_template;
                MPI_Info FILE_INFO_TEMPLATE;

                ierr = MPI_Info_create(&FILE_INFO_TEMPLATE);

                acc_template = H5Pcreate(OMP_FILE_ACCESS);

                ierr = H5Pset_fapl_mpio(acc_template, MPI_COMM_WORLD, FILE_INFO_TEMPLATE);
		std::string filename(tmp, len);

                file_id[i] =H5Fopen(filename.c_str(),H5F_ACC_RDWR,acc_template);

                ierr = H5Pclose(acc_template);
                ierr = MPI_Info_free(&FILE_INFO_TEMPLATE);
            }
        }
        for (long k = 0; k < nro_blocks; k++) {
            float* Block_Diag = &a[k*nro_blocks+k];
            int dev_spotrf = (int)a[k*nro_blocks+k];
            #pragma omp target nowait depend(inout: *Block_Diag) \
                depend(inout:vector[dev_spotrf])\
                firstprivate(k,nb,dev_spotrf,dim) \
                map(to:file_id[dev_spotrf:1])\
                device(dev_spotrf)
            {

                float *BDiag = (float*)malloc(nb*nb*sizeof(float));
                ReadBlock(dim,nb,k,k, file_id[dev_spotrf], BDiag);                
                
                int num_devices;
                cudaGetDeviceCount(&num_devices);
                int world_rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
                cudaSetDevice(world_rank%num_devices);
                
                long mem_size = nb*nb*sizeof(float);
                cusolverDnHandle_t handle;
                cusolverDnCreate(&handle);

                // Allocate memory on the GPU
                float* dA;
                cudaMalloc((void **) &dA, mem_size);
                cudaMemcpy(dA, BDiag, mem_size, cudaMemcpyHostToDevice);
                    
                // Parameters for cuSOLVER
                int work_size = 0;
                int *devInfo;
                 cudaMalloc((void**)&devInfo, sizeof(int));

                //Workspace
                cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, nb, dA, nb, &work_size);
                float *work;
                cudaMalloc((void**)&work, work_size * sizeof(float));

                cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, nb, dA, nb, work, work_size, devInfo);

                cudaMemcpy(BDiag, dA, mem_size, cudaMemcpyDeviceToHost); 

                cudaFree(dA);
                cudaFree(work);
                cudaFree(devInfo);
                cusolverDnDestroy(handle);
                
                WriteBlock(dim,nb,k,k, file_id[dev_spotrf], BDiag);
                free(BDiag);
            }
            for (long m = k + 1; m < nro_blocks; m++) {
                Block_Diag = &a[k*nro_blocks+k];
                float *Block_mk = &a[m*nro_blocks+k];
                int dev_strsm = (int)a[m*nro_blocks+k];
                #pragma omp target nowait depend(in: *Block_Diag) \
                    depend(inout: *Block_mk) depend(inout:vector[dev_strsm])\
                    firstprivate(m,k,dim,nb,len,dev_strsm) \
                    map(to:tmp[:len]) device(dev_strsm)
                {
                    float *BDiag = (float*)malloc(nb*nb*sizeof(float));
                    float *Bmk = (float*)malloc(nb*nb*sizeof(float));
                    ReadBlock(dim,nb,k,k, file_id[dev_strsm], BDiag);
                    ReadBlock(dim,nb,m,k, file_id[dev_strsm], Bmk);
                    
                    
                    cudaGetDeviceCount(&num_devices);
                    int world_rank;
                    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
                    cudaSetDevice(world_rank%num_devices);

                    const float alpha = 1.0f;
                    cublasHandle_t handle;
                    cudaEvent_t start, stop;

                    long mem_size = nb*nb*sizeof(float);

                    float *dA, *dB, *temp;
                    cudaMalloc((void **) &dA, mem_size);
                    cudaMalloc((void **) &dB, mem_size);

                    // Copy data from Host to Device
                    cudaMemcpy(dA, BDiag, mem_size, cudaMemcpyHostToDevice);
                    cudaMemcpy(dB, Bmk, mem_size, cudaMemcpyHostToDevice);
                    
                    free(BDiag);

                    // Create a handle
                    cublasCreate(&handle);

                    // DO MatMul
                    cublasStrsm(handle,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,nb,nb, &alpha,dA,nb,dB,nb);

                    // copy result from device to host
                    cudaMemcpy(Bmk, dB, mem_size, cudaMemcpyDeviceToHost);          

                    // Destroy the handle
                    cublasDestroy(handle);

                    // Erase device buffers
                    cudaFree(dA);
                    cudaFree(dB);

                    WriteBlock(dim,nb,m,k, file_id[dev_strsm], Bmk);
                    free(Bmk);
                }
            }
            for (long m = k + 1; m < nro_blocks; m++) {
                float *Block_mk = &a[m*nro_blocks+k];
                float *Block_mm = &a[m*nro_blocks+m];
                int dev_ssyrk = (int)a[m*nro_blocks+m];
                #pragma omp target nowait depend(in: *Block_mk) \
                    depend(inout: *Block_mm)  depend(inout:vector[dev_ssyrk]) \
                    firstprivate(m,k,dim,nb,dev_ssyrk) \
                    map(to:file_id[dev_ssyrk:1]) device(dev_ssyrk)
                {

                    float *Bmk = (float*)malloc(nb*nb*sizeof(float));
                    float *Bmm = (float*)malloc(nb*nb*sizeof(float));
                    ReadBlock(dim,nb,m,k, file_id[dev_ssyrk], Bmk);
                    ReadBlock(dim,nb,m,m, file_id[dev_ssyrk], Bmm);
                    
                    int num_devices;
                    cudaGetDeviceCount(&num_devices);
                    int world_rank;
                    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
                    cudaSetDevice(world_rank%num_devices);

                    const float alpha = -1.0f;
                    const float beta  = 1.0f;
                    cublasHandle_t handle;
                    cudaEvent_t start, stop;

                    long mem_size = nb*nb*sizeof(float);

                    float *dA,  *dC, *temp;
                    cudaMalloc((void **) &dA, mem_size);
                    cudaMalloc((void **) &dC, mem_size);

                    // Copy data from Host to Device
                    cudaMemcpy(dA, Bmk, mem_size, cudaMemcpyHostToDevice);
                    free(Bmk);
                    cudaMemcpy(dC, Bmm, mem_size, cudaMemcpyHostToDevice);

                    // Create a handle
                    cublasCreate(&handle);

                    // DO MatMul
                    cublasSsyrk(handle,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,nb,nb,&alpha,dA,nb,&beta,dC,nb);
                  
                    // copy result from device to host
                    cudaMemcpy(Bmm, dC, mem_size, cudaMemcpyDeviceToHost);          

                    // Destroy the handle
                    cublasDestroy(handle);

                    // Erase device buffers
                    cudaFree(dA);
                    cudaFree(dC);
                    
                    WriteBlock(dim,nb,m,m, file_id[dev_ssyrk], Bmm);
                    free(Bmm);
                }

                for (long n = k + 1; n < m; n++) {
                    float *Block_nk = &a[n*nro_blocks+k];
                    float *Block_mn = &a[m*nro_blocks+n];
                    int dev_sgemm = (int)a[m*nro_blocks+n];
                    #pragma omp target nowait depend(in: *Block_mk, *Block_nk) \
                        depend(inout: *Block_mn) depend(inout:vector[dev_sgemm]) \
                        firstprivate(m,k,n,dim,nb,len,dev_sgemm) \
                        map(to:tmp[:len]) device(dev_sgemm)
                    {
                        float *Bmk = (float*)malloc(nb*nb*sizeof(float));
                        float *Bnk = (float*)malloc(nb*nb*sizeof(float));
                        float *Bmn = (float*)malloc(nb*nb*sizeof(float));
                        ReadBlock(dim,nb,m,k, file_id[dev_sgemm], Bmk);
                        ReadBlock(dim,nb,n,k, file_id[dev_sgemm], Bnk);
                        ReadBlock(dim,nb,m,n, file_id[dev_sgemm], Bmn);                        
                
                        int num_devices;
                        cudaGetDeviceCount(&num_devices);
                        int world_rank;
                        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
                        cudaSetDevice(world_rank%num_devices);

                        const float alpha = -1.0f;
                        const float beta  = 1.0f;
                        cublasHandle_t handle;
                        cudaEvent_t start, stop;

                        long mem_size = nb*nb*sizeof(float);
            
                        float *dA, *dB, *dC, *temp;
                        cudaMalloc((void **) &dA, mem_size);
                        cudaMalloc((void **) &dB, mem_size);
                        cudaMalloc((void **) &dC, mem_size);

                        // Copy data from Host to Device
                        cudaMemcpy(dA, Bmk, mem_size, cudaMemcpyHostToDevice);
                        free(Bmk);
                        cudaMemcpy(dB, Bnk, mem_size, cudaMemcpyHostToDevice);
                        free(Bnk);
                        cudaMemcpy(dC, Bmn, mem_size, cudaMemcpyHostToDevice);

                        // Create a handle
                        cublasCreate(&handle);

                        // DO MatMul
                        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb, nb, nb, &alpha, dA, nb, dB, nb, &beta, dC, nb);
                    
                        // copy result from device to host
                        cudaMemcpy(Bmn, dC, mem_size, cudaMemcpyDeviceToHost);          

                        // Destroy the handle
                        cublasDestroy(handle);

                        // Erase device buffers
                        cudaFree(dA);
                        cudaFree(dB);
                        cudaFree(dC);

                        WriteBlock(dim,nb,m,n, file_id[dev_sgemm], Bmn);
                        free(Bmn);
                    }
                }
            }
        }

        #pragma omp taskwait
        
        for(int i=0;i<nro_devices;i++)
        {
            #pragma omp target nowait \
                firstprivate(i) \
                map(tofrom:file_id[i:1])\
                depend(in:vector[i]) device(i)
            {
                H5Fclose(file_id[i]);              
            }
        }
    }
}

int main(int argc, char **argv) {
    double t;

    if (argc != 6) {
        fprintf(stderr, "[USAGE] %s dim nb direction_data r workers\n", argv[0]);
        fprintf(stderr, "  nb must divide dim\n");
        return 1;
    }
    const long N = atoi(argv[1]);
    const long BS = atoi(argv[2]);
    const char * tmp =  argv[3];
    const int r = atoi(argv[4]);
    const int workers = atoi(argv[5]);

    if (N % BS != 0) {
        fprintf(stderr, "[USAGE] nb = %s must divide dim = %s\n", argv[1], argv[2]);
        return 1;
    }

    // dev
    int **dev = (int **)malloc(r * sizeof(int *));
    for (int i = 0; i < r; i++) {
        dev[i] = (int *)malloc(r * sizeof(int));
    }

    int dev1 = 0;
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<i;j++)
        {
            dev[i][j] = dev1;
            dev[j][i] = dev1;
            dev1++;
        }
    }

    int p;
    if(r%2==1){ //impar 
        p = r*(r-1)/2;
        // extended
        int l;
        int middle = r/2;
        for(l=0;l<r;l++)
        {
            dev[l][l] = dev[l][middle];
            middle++;
            if(middle==r) break;
        }
        middle = r/2+1;
        for(int k= l+1; k<r;k++){
            dev[k][k] = dev[k-(l+1)][middle];
            middle++;
        }
    }
    else{ //par
        // simple p= r*r/2
        p = r*r/2;
        int dev_diag =  dev1;
        for(int i=0;i<r;i++)
        {
            dev[i][i] = dev_diag;
            dev_diag++;
            if (dev_diag >= p) dev_diag = dev1;
        }
    }
    int threads = p/workers;
    long nro_blocks = N/BS; 

    for(int i=0;i<r;i++){
        for(int j=0;j<r;j++){
            dev[i][j] += (threads-1);
        }
    } 

    t = omp_get_wtime();
    CreateFile(tmp,workers,threads,N,BS);
    t = omp_get_wtime() - t;
    printf("Create file %0.6lf\n",t);
    fflush(stdout);

    t = omp_get_wtime();
    GenerateBlocks(tmp,N,BS,dev,workers,threads,r);
    t = omp_get_wtime() - t;
    printf("Time second target %0.6lf\n",t);
    fflush(stdout);

    float* a = (float *)malloc(nro_blocks * nro_blocks * sizeof(float));
    for(long j=0; j<nro_blocks; j++){
        for(long i=j; i<nro_blocks; i++){
            a[i*nro_blocks+j] = dev[i%r][j%r];
        }
    }

    t = omp_get_wtime();
    Cholesky_Factorization(a,tmp,N,BS,workers,threads,r);
    t = omp_get_wtime() - t;
    printf("Cholesky Factorization time %0.6lf\n",t);

    for (int i = 0; i < r; i++) {
        free(dev[i]);
    }
    free(dev);
    free(a);

    return 0;
}

