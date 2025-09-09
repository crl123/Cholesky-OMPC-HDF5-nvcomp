/******************************************************************************
 * Cholesky + GPU + SNAPPY + HDF5				              *
 * Author: Carla Cardoso Cusihuallpa					      *
 *						      			      *
 * This example demonstrates the execution of the Cholesky factorization      *
 * using the OMPC runtime to distribute tasks across MPI processes,           *
 * CUDA routines to perform linear algebra operations on the GPU,             *
 * and SNAPPY as the compression algorithm from the NVCOMP library.           *
 * HDF5 is also employed to store matrix blocks on disk efficiently,          * 
 * enabling out-of-core execution and scalable data management.               *
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <string>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <cstring>

#pragma omp declare target
extern "C" {
  #include "hdf5.h"
}
#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <nvcomp/snappy.hpp>
#include <nvcomp.hpp>
#pragma omp end declare target

#define PORC 2.4

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
void FillBlock(int nx, int ny, int block_i, int block_j, int N, int BS, float *array) {
    int init = block_i*N*BS+block_j*BS;
    int i, j;
    for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
              array[j*BS+i] = 0.0;
        }
    }
  
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
              array[i*BS+j] = (float) (init+i+j*N)/((float) N*N);
        }
    } 

    if (block_i == block_j) {
        for (long i = 0; i < ny; i++) {
            array[i*BS+i] = array[i*BS+i] + N;
            for (long j = 0; j < i; j++) {
                array[i*BS+j] = array[j*BS+i];
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
            //int nro_dev = i /p;
            #pragma omp target nowait firstprivate(OMP_FILE_ACCESS) \
                firstprivate(len,i,dim,nb) \
                map(to:tmp[:len])\
                depend(in:vector[i]) device(i)
            {
                int nx, ny, mx, my;

                nx = nb*PORC;
                ny = nb*PORC;
                int ng = 0;

                int rank;
                hsize_t dimens_2d[2];
                hsize_t chunk_dims[2];
                hid_t err;

                hid_t dataspace, dataset;
                hid_t acc_template;

                // define an info object to store MPI-IO information 
                hid_t file_identifier;

                int ierr;
                         
                mx = dim*PORC;
                my = dim*PORC;

                acc_template = H5Pcreate(OMP_FILE_ACCESS);

                ierr = H5Pset_fapl_mpio(acc_template, MPI_COMM_WORLD, MPI_INFO_NULL);
		std::string filename(tmp,len);

                file_identifier = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,acc_template);

                ierr = H5Pclose(acc_template);

                rank = 2;

		int nro_blocks = dim/nb;
		int nro_tile = (nro_blocks+1)*nro_blocks/2;

                dimens_2d[0] = ((nro_tile+nro_blocks-1)/nro_blocks)*nx;
                dimens_2d[1] = mx;

                dataspace = H5Screate_simple(rank, dimens_2d, NULL);

                acc_template = H5Pcreate(H5P_DATASET_CREATE);
                float fill_value = 0; 
                H5Pset_fill_value(acc_template, H5T_NATIVE_FLOAT, &fill_value);
                chunk_dims[0] = ny;
                chunk_dims[1] = nx;
                H5Pset_chunk(acc_template, rank, chunk_dims);

                dataset = H5Dcreate(file_identifier, "data", H5T_NATIVE_UCHAR,dataspace, H5P_DEFAULT,acc_template,H5P_DEFAULT);
                
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
 * each matrix block, compresses it using `nvcomp`, and writes it to the
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
void GenerateBlocks(const char *tmp,long dim, long nb, int ** dev, int p, int q,int r, size_t* nro_size){
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
	        int  ierr;
	        hid_t acc_template;
                MPI_Info FILE_INFO_TEMPLATE;

                ierr = MPI_Info_create(&FILE_INFO_TEMPLATE);

                acc_template = H5Pcreate(OMP_FILE_ACCESS);

                ierr = H5Pset_fapl_mpio(acc_template, MPI_COMM_WORLD, FILE_INFO_TEMPLATE);
		MPI_Info info;
		MPI_Info_create(&info);
		MPI_Info_set(info, "romio_cb_write", "enable");
		MPI_Info_set(info, "cb_buffer_size", "536870912"); 
		MPI_Info_set(info, "striping_unit", "67108864");
		MPI_Info_set(info, "romio_ds_write", "disable"); 
		H5Pset_fapl_mpio(acc_template, MPI_COMM_WORLD, info);

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
		            map(from:nro_size[i*nro_blocks+j:1])\
                    depend(inout:vector[node_local]) \
                    device(node_local)
		{
		    int  ierr;
		    int r, nx=nb*PORC, ny=nb*PORC, mx=dim*PORC, my=dim*PORC, ng=0;

		    int num_devices;
		    cudaGetDeviceCount(&num_devices);
		    int world_rank;
                    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
                    cudaSetDevice(world_rank%num_devices);

		    float *h_uncompressed;
		    size_t uncompressed_size = sizeof(float)*nb*nb;
		    cudaMallocHost(&h_uncompressed, uncompressed_size);
		    FillBlock(nb, nb, i, j, dim, nb, h_uncompressed);

                    void *d_uncompressed;
                    cudaMalloc(&d_uncompressed, uncompressed_size);
                    cudaMemcpy(d_uncompressed, h_uncompressed, uncompressed_size, cudaMemcpyHostToDevice);
                    cudaFreeHost(h_uncompressed);

                    // Variables
                    size_t real_compressed_size = 0;
                    uint8_t* device_compressed_data = nullptr;
                    size_t* device_compressed_size = nullptr;

                    // Compress configuration
                    size_t uncomp_chunk_size_elements = 1024 * 1024;
                    size_t uncomp_chunk_size_bytes = uncomp_chunk_size_elements * sizeof(float);
                    
                    cudaStream_t stream = nullptr; 
                    cudaStreamCreate(&stream);

                    nvcompBatchedSnappyOpts_t snappy_opts;
                    memset(&snappy_opts, 0, sizeof(nvcompBatchedSnappyOpts_t));
                    
                    
                    {
                        nvcomp::SnappyManager snappy_manager(
                            uncomp_chunk_size_bytes,
                            snappy_opts,
                            stream
                        );

                        nvcomp::CompressionConfig comp_config = snappy_manager.configure_compression(uncompressed_size);

                        cudaMalloc(&device_compressed_data, comp_config.max_compressed_buffer_size);
                        cudaMalloc(&device_compressed_size, sizeof(size_t));

                        snappy_manager.compress(
                            reinterpret_cast<const uint8_t*>(d_uncompressed),
                            device_compressed_data,
                            comp_config,
                            device_compressed_size
                        );

                        cudaDeviceSynchronize();
                        cudaMemcpy(&real_compressed_size, device_compressed_size, sizeof(size_t), cudaMemcpyDeviceToHost);

                    }

                    uint8_t* h_compressed = (uint8_t*)malloc(real_compressed_size);
                    cudaMemcpy(h_compressed, device_compressed_data, real_compressed_size, cudaMemcpyDeviceToHost);

                    // Free memory
                    if(device_compressed_data) cudaFree(device_compressed_data);
                    if(device_compressed_size) cudaFree(device_compressed_size);
                    cudaFree(d_uncompressed);
                    cudaStreamDestroy(stream);

		    nro_size[i*nro_blocks+j] = real_compressed_size;

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

		    long SUBCHUNK_DIM0 = (real_compressed_size + nx -1)/(nx);
		    long SUBCHUNK_DIM1 = ny;

		    count_2d[0] = SUBCHUNK_DIM0;
		    count_2d[1] = SUBCHUNK_DIM1;

		    err = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start_2d, stride_2d, count_2d, NULL);

		    rank =2;
		    dimens_2d[0] = ny+2*ng;
		    dimens_2d[1] = nx+2*ng;

		    memspace = H5Screate_simple(rank, dimens_2d, NULL);

		    start_2d[0] = ng;
		    start_2d[1] = ng;

		    stride_2d[0] = 1;
		    stride_2d[1] = 1;

		    count_2d[0] = SUBCHUNK_DIM0;
		    count_2d[1] = SUBCHUNK_DIM1;

		    err = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, start_2d, stride_2d, count_2d, NULL);
		    acc_template = H5Pcreate(H5P_DATASET_XFER);
		    ierr = H5Pset_dxpl_mpio(acc_template, H5FD_MPIO_INDEPENDENT);

		    hsize_t size_mem = H5Sget_select_npoints(memspace);
		    if (size_mem != real_compressed_size) {
			h_compressed = static_cast<uint8_t*>(realloc(h_compressed, size_mem));
                        std::fill(
                            h_compressed + real_compressed_size,
                            h_compressed + size_mem,
                            uint8_t{0}
                        );
		    }

		    err = H5Dwrite(dataset, H5T_NATIVE_UCHAR, memspace, dataspace,acc_template, h_compressed);

		    H5Pclose(acc_template);
		    H5Sclose(memspace);
		    H5Sclose(dataspace);
		    H5Dclose(dataset);
                    free(h_compressed);

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
void ReadBlock(long dim, long nb, long i, long j, hid_t file_id, size_t compressed_size,uint8_t **compressed_data_out){
    hid_t acc_template;
    hid_t dataspace, memspace, dataset;

    int rank = 2, ierr;

    dataset = H5Dopen(file_id, "/data", H5P_DEFAULT);

    dataspace = H5Dget_space(dataset);

    size_t size = nb*PORC;
    hsize_t SUBCHUNK_DIM0 = (compressed_size + size -1)/(size);
    hsize_t SUBCHUNK_DIM1 = size;

    int nro_blocks = dim/nb;
    int id_blocks = j*nro_blocks+i-(j*(j+1)/2);
    int i_ = id_blocks / nro_blocks;
    int j_ = id_blocks - i_*nro_blocks;

    hsize_t start_2d[2] = {i_*size, j_*size};
    hsize_t count_2d[2] = {SUBCHUNK_DIM0, SUBCHUNK_DIM1};
    memspace = H5Screate_simple(rank, count_2d, NULL);
    ierr = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start_2d, NULL, count_2d, NULL);

    acc_template = H5Pcreate(H5P_DATASET_XFER);
    ierr = H5Pset_dxpl_mpio(acc_template, H5FD_MPIO_INDEPENDENT);

    uint8_t *compressed_data = (uint8_t *)malloc(SUBCHUNK_DIM0*SUBCHUNK_DIM1 * sizeof(uint8_t));

    H5Dread(dataset, H5T_NATIVE_UCHAR, memspace, dataspace, acc_template, compressed_data);

    H5Pclose(acc_template);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);

    std::memcpy(*compressed_data_out, compressed_data, compressed_size);
    free(compressed_data);
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
void WriteBlock(long dim, long nb, int i, int j, hid_t file_id, uint8_t** compressed_data, size_t compressed_size){

    long size = nb*PORC;
    int rank, ng =0 ,ierr;
    hsize_t start_2d[2];
    hsize_t stride_2d[2];
    hsize_t count_2d[2];

    hsize_t dimens_2d[2];
    hid_t dataspace, memspace, dataset;
    hid_t err;
    hid_t acc_template, plist_id;

    if(size*size < compressed_size){
	   printf("ERROR %zu < %zu\n",size*size,compressed_size);
    }

    // Get identifier
    dataset = H5Dopen(file_id, "/data", H5P_DEFAULT);

    dataspace =  H5Dget_space(dataset);

    int nro_blocks = dim/nb;
    int id_blocks = j*nro_blocks+i-(j*(j+1)/2);
    int i_ = id_blocks / nro_blocks;
    int j_ = id_blocks - i_*nro_blocks;

    start_2d[0] = i_*size;
    start_2d[1] = j_*size;

    stride_2d[0] = 1;
    stride_2d[1] = 1;

    long SUBCHUNK_DIM0 = (compressed_size + size -1)/(size);
    long SUBCHUNK_DIM1 = size;

    count_2d[0] = SUBCHUNK_DIM0;
    count_2d[1] = SUBCHUNK_DIM1;

    err = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start_2d, stride_2d, count_2d, NULL);

    rank =2;
    dimens_2d[0] = size+2*ng;
    dimens_2d[1] = size+2*ng;

    memspace = H5Screate_simple(rank, dimens_2d, NULL);

    start_2d[0] = 0;
    start_2d[1] = 0;

    stride_2d[0] = 1;
    stride_2d[1] = 1;

    count_2d[0] = SUBCHUNK_DIM0;
    count_2d[1] = SUBCHUNK_DIM1;

    err = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, start_2d, stride_2d, count_2d, NULL);

    acc_template = H5Pcreate(H5P_DATASET_XFER);
    ierr = H5Pset_dxpl_mpio(acc_template, H5FD_MPIO_INDEPENDENT);

    hsize_t expected = H5Sget_select_npoints(memspace);
   
    if (expected != compressed_size) {
	*compressed_data = static_cast<uint8_t*>(realloc(*compressed_data, expected));
	if (expected > compressed_size) {
            memset((char *)(*compressed_data) +compressed_size,0,expected-compressed_size);
	}
	
    }

    ierr = H5Dwrite(dataset, H5T_NATIVE_UCHAR, memspace, dataspace, acc_template, * compressed_data);

    H5Pclose(acc_template);
    H5Sclose(memspace);

    H5Sclose(dataspace);
    H5Dclose(dataset);
}

/**
 * @brief Executes the out-of-core Cholesky factorization on a large matrix.
 *
 * This is the main loop of the Cholesky algorithm. It performs the factorization
 * by blocks, reading compressed data from HDF5, decompressing it, performing
 * linear algebra calculations on the GPU, and re-compressing the results
 * to write back to disk.
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
void Cholesky_Factorization(float * a,const char * tmp, long dim, long nb, int p, int q,int r,size_t* nro_size){
    int nro_blocks = dim/nb;
    int len = strlen(tmp);
    int *vector = (int *)malloc(p * q *sizeof(int));
    long BS=1;
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
                map(from:file_id[i:1]) map(to:tmp[:len])\
                depend(inout:vector[i]) device(i)
            {
                int MyPE, NumPEs;
                int  ierr;
                ierr = MPI_Comm_size(MPI_COMM_WORLD, &NumPEs);
                ierr = MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);

                hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
		H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

		std::string filename(tmp, len);

		file_id[i] = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
		H5Pclose(plist_id);
            }
        }
	for (int i = 0; i < nro_blocks; ++i) {
            for (int j = i; j < nro_blocks; ++j) {
                size_t *BlockA = &nro_size[j*(nro_blocks)+i]; //A.GetBlock(j, i);
                #pragma omp target enter data nowait \
                            map(to: BlockA[:BS * BS]) \
                            depend(out: *BlockA)
            }
        }

	for (long k = 0; k < nro_blocks; k++) {
	     size_t nroDiag = k*nro_blocks+k;
            int dev_spotrf = (int)a[nroDiag];
            size_t *Block_Diag = &nro_size[nroDiag];
            #pragma omp target nowait depend(inout:*Block_Diag) \
                depend(inout:vector[dev_spotrf])\
                firstprivate(k,dim,nb,dev_spotrf) \
                map(to:file_id[dev_spotrf:1])\
                device(dev_spotrf)
            {

                size_t uncompressed_size = sizeof(float)*nb*nb;

		int num_devices;
		cudaGetDeviceCount(&num_devices);
		int world_rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
                cudaSetDevice(world_rank%num_devices);

		// Read compressed block
		uint8_t *hDiag_compressed;
		cudaMallocHost(&hDiag_compressed, Block_Diag[0]);
	        ReadBlock(dim,nb,k,k, file_id[dev_spotrf],Block_Diag[0],&hDiag_compressed);

                size_t real_compressed_size = 0;
                size_t uncomp_chunk_size_elements = 1024 * 1024;
		size_t uncomp_chunk_size_bytes = uncomp_chunk_size_elements * sizeof(float);
                uint8_t* device_compressed_data = nullptr;
                size_t* device_compressed_size = nullptr;

                cudaStream_t stream = nullptr; 
                cudaStreamCreate(&stream);

                nvcompBatchedSnappyOpts_t snappy_opts;
                memset(&snappy_opts, 0, sizeof(nvcompBatchedSnappyOpts_t));

                {

                    nvcomp::SnappyManager snappy_manager(
                            uncomp_chunk_size_bytes,
                            snappy_opts,
                            stream
                    );

                    // Send block compressed to gpu
                    uint8_t *dDiag_compressed;
                    cudaMalloc(&dDiag_compressed, Block_Diag[0]);
                    cudaMemcpy(dDiag_compressed, hDiag_compressed, Block_Diag[0], cudaMemcpyHostToDevice);

                    // Configure decompression
		    nvcomp::DecompressionConfig decomp_config = snappy_manager.configure_decompression(dDiag_compressed, &Block_Diag[0]);

                    // Allocate memory for the decompressed data
                    float *dA;
                    cudaMalloc(&dA, decomp_config.decomp_data_size);

                    snappy_manager.decompress(
                                reinterpret_cast<uint8_t*>(dA),
                                dDiag_compressed,
                                decomp_config
                    );

                    cudaFree(dDiag_compressed);

                    // Calculate potrf
		    cusolverDnHandle_t handle;
                    cusolverDnCreate(&handle);
                    int work_size = 0;
                    int *devInfo;
                    cudaMalloc((void**)&devInfo, sizeof(int));
                    cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, nb, dA, nb, &work_size);
                    float *work;
                    cudaMalloc((void**)&work, work_size * sizeof(float));
                    cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, nb, dA, nb, work, work_size, devInfo);
                    cudaFree(work);
                    cudaFree(devInfo);
                    cusolverDnDestroy(handle);

                    cudaDeviceSynchronize(); 

                    // Configure compression
                    nvcomp::CompressionConfig comp_config = snappy_manager.configure_compression(uncompressed_size);
                     
                    // Allocate memory in the GPU to compressed data
                    cudaMalloc(&device_compressed_data, comp_config.max_compressed_buffer_size); 
                    
                    // Allocate memory for the actual output size
                    cudaMalloc(&device_compressed_size, sizeof(size_t));

                    // Compress
                    snappy_manager.compress(
                        reinterpret_cast<const uint8_t*>(dA),
                        device_compressed_data,
                        comp_config,
                        device_compressed_size
                    );

                    cudaDeviceSynchronize();                    
                   
                    cudaMemcpy(&real_compressed_size, device_compressed_size, sizeof(size_t), cudaMemcpyDeviceToHost); 
                    
                    cudaFree(dA);

                }
                cudaFreeHost(hDiag_compressed);

                uint8_t* h_compressed = (uint8_t*)malloc(real_compressed_size);
                cudaMemcpy(h_compressed, device_compressed_data, real_compressed_size, cudaMemcpyDeviceToHost);
               
                cudaStreamDestroy(stream);
 
                if(device_compressed_data) cudaFree(device_compressed_data);
                if(device_compressed_size) cudaFree(device_compressed_size);

                Block_Diag[0] = real_compressed_size;
                WriteBlock(dim,nb,k,k, file_id[dev_spotrf], &h_compressed,Block_Diag[0]);                
                free(h_compressed); 
	    }

	    for (long m = k + 1; m < nro_blocks; m++) {
                size_t nroMK = m*nro_blocks+k;
                int dev_strsm = (int)a[nroMK];
                Block_Diag = &nro_size[nroDiag];
                size_t *Block_mk = &nro_size[nroMK];
                #pragma omp target nowait  depend(in: *Block_Diag) depend(inout: *Block_mk) \
                    depend(inout:vector[dev_strsm])\
                    firstprivate(k,dim,m,nb,dev_strsm) \
                    map(to:file_id[dev_strsm:1])\
                    device(dev_strsm)
                {

                    size_t uncompressed_size = sizeof(float)*nb*nb;

		    int num_devices;
                    cudaGetDeviceCount(&num_devices);
                    int world_rank;
                    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
                    cudaSetDevice(world_rank%num_devices);

		    // Read compressed block 1
		    uint8_t *hDiag_compressed;
                    cudaMallocHost(&hDiag_compressed, Block_Diag[0]);
                    ReadBlock(dim,nb,k,k, file_id[dev_strsm],Block_Diag[0],&hDiag_compressed);

		    // Read compressed block 2
		    uint8_t *hBmk_compressed;
		    cudaMallocHost(&hBmk_compressed,Block_mk[0]);
		    ReadBlock(dim,nb,m,k, file_id[dev_strsm], Block_mk[0], &hBmk_compressed);

                    size_t real_compressed_size = 0;
                    size_t uncomp_chunk_size_elements = 1024 * 1024;
		    size_t uncomp_chunk_size_bytes = uncomp_chunk_size_elements * sizeof(float);

		    uint8_t* device_compressed_data = nullptr;
                    size_t* device_compressed_size = nullptr;

                    cudaStream_t stream = nullptr; 
                    cudaStreamCreate(&stream);

                    nvcompBatchedSnappyOpts_t snappy_opts;
                    memset(&snappy_opts, 0, sizeof(nvcompBatchedSnappyOpts_t));

                    {

                        nvcomp::SnappyManager snappy_manager(
                                uncomp_chunk_size_bytes,
                                snappy_opts,
                                stream
                        );

                        // Descompression A
                        uint8_t *dDiag_compressed;
                        cudaMalloc(&dDiag_compressed, Block_Diag[0]);
                        cudaMemcpyAsync(dDiag_compressed, hDiag_compressed, Block_Diag[0], cudaMemcpyHostToDevice);
                        nvcomp::DecompressionConfig decomp_configA = snappy_manager.configure_decompression(dDiag_compressed, &Block_Diag[0]);
                        float *dA;
                        cudaMalloc(&dA, decomp_configA.decomp_data_size);
                        snappy_manager.decompress(
                                        reinterpret_cast<uint8_t*>(dA),
                                        dDiag_compressed,
                                        decomp_configA
                        );

                        // Descompression B
                        uint8_t *dBmk_compressed;
                        cudaMalloc(&dBmk_compressed, Block_mk[0]);
                        cudaMemcpyAsync(dBmk_compressed, hBmk_compressed, Block_mk[0], cudaMemcpyHostToDevice);
                        nvcomp::DecompressionConfig decomp_configB = snappy_manager.configure_decompression(dBmk_compressed, &Block_mk[0]);
                        float *dB;
                        cudaMalloc(&dB, decomp_configB.decomp_data_size);
                        snappy_manager.decompress(
                                        reinterpret_cast<uint8_t*>(dB),
                                        dBmk_compressed,
                                        decomp_configB
                        );
                        cudaStreamSynchronize(stream);
                        cudaFree(dDiag_compressed);
                        cudaFree(dBmk_compressed);
                        cudaFreeHost(hDiag_compressed);  // Liberar tan pronto como sea posible
                        cudaFreeHost(hBmk_compressed);

		        // Calculate trsm
		        const float alpha = 1.0f;
                        cublasHandle_t handle;
                        cudaEvent_t start, stop;
                        cublasCreate(&handle);
                        cublasStrsm(handle,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,nb,nb, &alpha,dA,nb,dB,nb);
                        cublasDestroy(handle);
                        cudaFree(dA); 

                        // Compression
                        nvcomp::CompressionConfig comp_config = snappy_manager.configure_compression(uncompressed_size);
                        cudaMalloc(&device_compressed_data, comp_config.max_compressed_buffer_size); 
                        cudaMalloc(&device_compressed_size, sizeof(size_t));
                        snappy_manager.compress(
                            reinterpret_cast<const uint8_t*>(dB),
                            device_compressed_data,
                            comp_config,
                            device_compressed_size
                        );
                        cudaStreamSynchronize(stream);  
                        
                        cudaMemcpy(&real_compressed_size, device_compressed_size, sizeof(size_t), cudaMemcpyDeviceToHost); 
                        cudaFree(dB);
                        

                    }

                    cudaFreeHost(hDiag_compressed);
                    cudaFreeHost(hBmk_compressed);
                    uint8_t* h_compressed = (uint8_t*)malloc(real_compressed_size);
                    cudaMemcpy(h_compressed, device_compressed_data, real_compressed_size, cudaMemcpyDeviceToHost);
                    
                    cudaStreamDestroy(stream);

                    if(device_compressed_data) cudaFree(device_compressed_data);
                    if(device_compressed_size) cudaFree(device_compressed_size);
                    
		    // Write Block
                    Block_mk[0] = real_compressed_size;
                    WriteBlock(dim,nb,m,k, file_id[dev_strsm], &h_compressed,Block_mk[0]);
                    free(h_compressed);
		}
	    }

	    for (long m = k + 1; m < nro_blocks; m++) {

	        size_t nroMK = m*nro_blocks+k;
                size_t nroMM = m*nro_blocks+m;
                int dev_ssyrk = (int)a[nroMM];
                size_t *Block_mk = &nro_size[nroMK];//A.GetBlock(m, k);
                size_t *Block_mm = &nro_size[nroMM];//A.GetBlock(m, m);
                #pragma omp target nowait depend(in: *Block_mk) depend(inout: *Block_mm)\
                    depend(inout:vector[dev_ssyrk])\
                    firstprivate(m,k,dim,nb,dev_ssyrk)\
                    map(to:file_id[dev_ssyrk:1]) \
                    device(dev_ssyrk)
                {

                    size_t uncompressed_size = sizeof(float)*nb*nb;

		    int num_devices;
                    cudaGetDeviceCount(&num_devices);
                    int world_rank;
                    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
                    cudaSetDevice(world_rank%num_devices);

		    // Read compressed block 1
                    uint8_t *hBmk_compressed;
                    cudaMallocHost(&hBmk_compressed, Block_mk[0]);
		    ReadBlock(dim,nb,m,k, file_id[dev_ssyrk], Block_mk[0], &hBmk_compressed);

		    // Read compressed block 2
		    uint8_t *hBmm_compressed;
                    cudaMallocHost(&hBmm_compressed, Block_mm[0]);
		    ReadBlock(dim,nb,m,m, file_id[dev_ssyrk], Block_mm[0], &hBmm_compressed);

		    size_t real_compressed_size = 0;
                    size_t uncomp_chunk_size_elements = 1024 * 1024;
		    size_t uncomp_chunk_size_bytes = uncomp_chunk_size_elements * sizeof(float);

		    uint8_t* device_compressed_data = nullptr;
                    size_t* device_compressed_size = nullptr;

                    cudaStream_t stream = nullptr; 
                    cudaStreamCreate(&stream);

                    nvcompBatchedSnappyOpts_t snappy_opts;
                    memset(&snappy_opts, 0, sizeof(nvcompBatchedSnappyOpts_t));

                    {

                        nvcomp::SnappyManager snappy_manager(
                                uncomp_chunk_size_bytes,
                                snappy_opts,
                                stream
                        );

                        // Descompression A
                        uint8_t *dBmk_compressed;
                        cudaMalloc(&dBmk_compressed, Block_mk[0]);
                        cudaMemcpyAsync(dBmk_compressed, hBmk_compressed, Block_mk[0], cudaMemcpyHostToDevice);
                        nvcomp::DecompressionConfig decomp_configA = snappy_manager.configure_decompression(dBmk_compressed, &Block_mk[0]);
                        float *dA;
                        cudaMalloc(&dA, decomp_configA.decomp_data_size);
                        snappy_manager.decompress(
                                        reinterpret_cast<uint8_t*>(dA),
                                        dBmk_compressed,
                                        decomp_configA
                        );

                        // Descompression dC
                        uint8_t *dBmm_compressed;
                        cudaMalloc(&dBmm_compressed, Block_mm[0]);
                        cudaMemcpyAsync(dBmm_compressed, hBmm_compressed, Block_mm[0], cudaMemcpyHostToDevice);
                        nvcomp::DecompressionConfig decomp_configC = snappy_manager.configure_decompression(dBmm_compressed, &Block_mm[0]);
                        float *dC;
                        cudaMalloc(&dC, decomp_configC.decomp_data_size);
                        snappy_manager.decompress(
                                        reinterpret_cast<uint8_t*>(dC),
                                        dBmm_compressed,
                                        decomp_configC
                        );


                        cudaStreamSynchronize(stream);
                        cudaFree(dBmk_compressed);
                        cudaFree(dBmm_compressed);
                        cudaFreeHost(hBmk_compressed);  // Liberar tan pronto como sea posible
                        cudaFreeHost(hBmm_compressed);

                        // Calculate syrk
		        const float alpha = -1.0f;
                        const float beta  = 1.0f;
                        cublasHandle_t handle;
                        cudaEvent_t start, stop;
                        cublasCreate(&handle);
                        cublasSsyrk(handle,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,nb,nb,&alpha,dA,nb,&beta,dC,nb);
                        cublasDestroy(handle);
                        cudaFree(dA);

                        // Compression
                        nvcomp::CompressionConfig comp_config = snappy_manager.configure_compression(uncompressed_size);
                        cudaMalloc(&device_compressed_data, comp_config.max_compressed_buffer_size); 
                        cudaMalloc(&device_compressed_size, sizeof(size_t));
                        snappy_manager.compress(
                            reinterpret_cast<const uint8_t*>(dC),
                            device_compressed_data,
                            comp_config,
                            device_compressed_size
                        );
                        cudaStreamSynchronize(stream);  
                        
                        cudaMemcpy(&real_compressed_size, device_compressed_size, sizeof(size_t), cudaMemcpyDeviceToHost); 
                        cudaFree(dC);

                    } 

                    cudaFreeHost(hBmk_compressed);
                    cudaFreeHost(hBmm_compressed);
                    uint8_t* h_compressed = (uint8_t*)malloc(real_compressed_size);
                    cudaMemcpy(h_compressed, device_compressed_data, real_compressed_size, cudaMemcpyDeviceToHost);
                    
                    cudaStreamDestroy(stream);

                    if(device_compressed_data) cudaFree(device_compressed_data);
                    if(device_compressed_size) cudaFree(device_compressed_size);

		    // Write Block
                    Block_mm[0] = real_compressed_size;
                    WriteBlock(dim,nb,m,m, file_id[dev_ssyrk], &h_compressed,Block_mm[0]);
                    free(h_compressed);
		}

		for (long n = k + 1; n < m; n++) {
		    size_t nroNK = n*nro_blocks+k;
                    size_t nroMN = m*nro_blocks+n;
                    int dev_sgemm = (int)a[nroMN];
                    size_t *Block_nk = &nro_size[nroNK];//A.GetBlock(n, k);
                    size_t *Block_mn = &nro_size[nroMN];//A.GetBlock(m, n);
                    #pragma omp target nowait depend(in: *Block_mk, *Block_nk) depend(inout: *Block_mn)\
                        depend(inout:vector[dev_sgemm])\
                        firstprivate(m,n,k,dim,nb,dev_sgemm) \
                        map(to:file_id[dev_sgemm:1]) \
                        device(dev_sgemm)
                    {

                        size_t uncompressed_size = sizeof(float)*nb*nb;

		        int num_devices;
                        cudaGetDeviceCount(&num_devices);
                        int world_rank;
                        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
                        cudaSetDevice(world_rank%num_devices);

			// Read compressed block
                        uint8_t *hBmk_compressed;
                        cudaMallocHost(&hBmk_compressed, Block_mk[0]);
                        ReadBlock(dim,nb,m,k, file_id[dev_sgemm], Block_mk[0], &hBmk_compressed);

			uint8_t *hBnk_compressed;
                        cudaMallocHost(&hBnk_compressed, Block_nk[0]);
                        ReadBlock(dim,nb,n,k, file_id[dev_sgemm], Block_nk[0], &hBnk_compressed);

			uint8_t *hBmn_compressed;
                        cudaMallocHost(&hBmn_compressed, Block_mn[0]);
	                ReadBlock(dim,nb,m,n, file_id[dev_sgemm], Block_mn[0], &hBmn_compressed);

                        size_t real_compressed_size = 0;
                        size_t uncomp_chunk_size_elements = 1024 * 1024;
		        size_t uncomp_chunk_size_bytes = uncomp_chunk_size_elements * sizeof(float);

		        uint8_t* device_compressed_data = nullptr;
                        size_t* device_compressed_size = nullptr;

                        cudaStream_t stream = nullptr; 
                        cudaStreamCreate(&stream);

                        nvcompBatchedSnappyOpts_t snappy_opts;
                        memset(&snappy_opts, 0, sizeof(nvcompBatchedSnappyOpts_t));

                        {

                            nvcomp::SnappyManager snappy_manager(
                                uncomp_chunk_size_bytes,
                                snappy_opts,
                                stream
                            );

                            // Descompression A
                            uint8_t *dBmk_compressed;
                            cudaMalloc(&dBmk_compressed, Block_mk[0]);
                            cudaMemcpyAsync(dBmk_compressed, hBmk_compressed, Block_mk[0], cudaMemcpyHostToDevice);
                            nvcomp::DecompressionConfig decomp_configA = snappy_manager.configure_decompression(dBmk_compressed, &Block_mk[0]);
                            float *dA;
                            cudaMalloc(&dA, decomp_configA.decomp_data_size);
                            snappy_manager.decompress(
                                        reinterpret_cast<uint8_t*>(dA),
                                        dBmk_compressed,
                                        decomp_configA
                            );

                            // Descompression B
                            uint8_t *dBnk_compressed;
                            cudaMalloc(&dBnk_compressed, Block_nk[0]);
                            cudaMemcpyAsync(dBnk_compressed, hBnk_compressed, Block_nk[0], cudaMemcpyHostToDevice);
                            nvcomp::DecompressionConfig decomp_configB = snappy_manager.configure_decompression(dBnk_compressed, &Block_nk[0]);
                            float *dB;
                            cudaMalloc(&dB, decomp_configB.decomp_data_size);
                            snappy_manager.decompress(
                                        reinterpret_cast<uint8_t*>(dB),
                                        dBnk_compressed,
                                        decomp_configB
                            );

                            // Descompression C
                            uint8_t *dBmn_compressed;
                            cudaMalloc(&dBmn_compressed, Block_mn[0]);
                            cudaMemcpyAsync(dBmn_compressed, hBmn_compressed, Block_mn[0], cudaMemcpyHostToDevice);
                            nvcomp::DecompressionConfig decomp_configC = snappy_manager.configure_decompression(dBmn_compressed, &Block_mn[0]);
                            float *dC;
                            cudaMalloc(&dC, decomp_configC.decomp_data_size);
                            snappy_manager.decompress(
                                        reinterpret_cast<uint8_t*>(dC),
                                        dBmn_compressed,
                                        decomp_configC
                            );

                            cudaStreamSynchronize(stream);
                            cudaFree(dBmk_compressed);
                            cudaFree(dBnk_compressed);
                            cudaFree(dBmn_compressed);
                            cudaFreeHost(hBmk_compressed);
                            cudaFreeHost(hBnk_compressed);
                            cudaFreeHost(hBmn_compressed);


  			    // Calculate sgemm
  			    const float alpha = -1.0f;
                            const float beta  = 1.0f;
                            cublasHandle_t handle;
                            cudaEvent_t start, stop;
                            cublasCreate(&handle);
                            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb, nb, nb, &alpha, dB, nb, dA, nb, &beta, dC, nb);
                            cublasDestroy(handle);
                            cudaFree(dA);
                            cudaFree(dB);

                            // Compression
                            nvcomp::CompressionConfig comp_config = snappy_manager.configure_compression(uncompressed_size);
                            cudaMalloc(&device_compressed_data, comp_config.max_compressed_buffer_size); 
                            cudaMalloc(&device_compressed_size, sizeof(size_t));
                            snappy_manager.compress(
                                reinterpret_cast<const uint8_t*>(dC),
                                device_compressed_data,
                                comp_config,
                                device_compressed_size
                            );
                            cudaStreamSynchronize(stream);  
                        
                            cudaMemcpy(&real_compressed_size, device_compressed_size, sizeof(size_t), cudaMemcpyDeviceToHost); 
                            cudaFree(dC);

                        }

                        cudaFreeHost(hBmk_compressed);
                        cudaFreeHost(hBnk_compressed);
                        cudaFreeHost(hBmn_compressed);
                        uint8_t* h_compressed = (uint8_t*)malloc(real_compressed_size);
                        cudaMemcpy(h_compressed, device_compressed_data, real_compressed_size, cudaMemcpyDeviceToHost);
                    
                        cudaStreamDestroy(stream);

                        if(device_compressed_data) cudaFree(device_compressed_data);
                        if(device_compressed_size) cudaFree(device_compressed_size);

			// Write Block
                        Block_mn[0] = real_compressed_size;
			WriteBlock(dim,nb,m,n, file_id[dev_sgemm],&h_compressed,Block_mn[0]);
			free(h_compressed);
		    }
		}
	    }

	}

	for (int i = 0; i < nro_blocks; ++i) {
            for (int j = i; j < nro_blocks; ++j) {
                size_t *BlockA =  &nro_size[j*(nro_blocks)+i];//A.GetBlock(j, i);
                #pragma omp target exit data nowait \
                            map(from: BlockA[:BS * BS]) depend(inout: *BlockA)
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
        fprintf(stderr, "[USAGE] %s  dim nb direction_data p q\n", argv[0]);
        fprintf(stderr, "  nb must divide dim\n");
        return 1;
    }

    // dev
    int **dev = (int **)malloc(r * sizeof(int *));
    for (int i = 0; i < r; i++) {
        dev[i] = (int *)malloc(r * sizeof(int));
    }

    int p = (r % 2 == 1) ? r*(r-1)/2 : r*r/2;
    int th = p/workers;

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

    if(r%2==1){ //impar
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
        int dev_diag =  dev1;
        for(int i=0;i<r;i++)
        {
            dev[i][i] = dev_diag;
            dev_diag++;
            if (dev_diag >= p) dev_diag = dev1;
        }
    }

    t = omp_get_wtime();
    CreateFile(tmp,workers,th,N,BS);
    t = omp_get_wtime() - t;
    printf("Create file %0.6lf\n",t);
    fflush(stdout);

    long nro_blocks = N/BS;
    size_t* nro_size = (size_t *)malloc(nro_blocks*nro_blocks* sizeof(size_t));

    t = omp_get_wtime();
    GenerateBlocks(tmp,N,BS,dev,workers,th,r,nro_size);
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
    Cholesky_Factorization(a,tmp,N,BS,workers,th,r,nro_size);
    t = omp_get_wtime() - t;
    printf("Cholesky Factorization time %0.6lf\n",t);

    free(dev);
    free(a);

    return 0;
}
