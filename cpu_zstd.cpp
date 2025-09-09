/********************************************************************************
 * Cholesky with only CPU, and with a level of parallelization                 *
 * For this version p                                                      *
*********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <unistd.h>
#include <string>

#include <cblas.h>
#include <lapacke.h>

#pragma omp declare target

#include "mpi.h"
#include <vector>
#include <zstd.h>

extern "C" {
  #include "hdf5.h"
}

#pragma omp end declare target

#define PORC 2

void print_tile(const char* label, float *tile, int64_t m, int64_t n)
{
    printf("%s\n", label);
    for (int64_t i = 0; i < m; i++)
    {
        printf(" ");
        for (int64_t j = 0; j < n; j++)
        {
            /*if (tile[i * n + j] >= 0) printf(" ");
            if (tile[i * n + j] == 0) printf("        ");*/
            printf("%.10f  ", (tile[i * n + j]));
        }
        printf("\n");
    }
}

void print_tile_char(const char* label, void *tile, int64_t m, int64_t n)
{
    unsigned char *data = (unsigned char *)tile;
    printf("%s\n", label);
    for (int64_t i = 0; i < m; i++)
    {
        printf(" ");
        for (int64_t j = 0; j < n; j++)
        {
            /*if (tile[i * n + j] >= 0) printf(" ");
            if (tile[i * n + j] == 0) printf("        ");*/
            printf("%02X  ", data[i * n + j]);
        }
        printf("\n");
    }
}


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

                int MyPE, NumPEs;
                int ierr;
                            
                ierr = MPI_Comm_size(MPI_COMM_WORLD, &NumPEs);
                ierr = MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
                //printf("Is the %d process of %d \n",MyPE,NumPEs);

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
                dimens_2d[1] = my;

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

std::vector<uint8_t> compressData(float* data, size_t numElements) {
    size_t dataSize = numElements * sizeof(float);
    size_t maxCompressedSize = ZSTD_compressBound(dataSize);
    std::vector<uint8_t> compressedData(maxCompressedSize);

    size_t compressedSize = ZSTD_compress(compressedData.data(), maxCompressedSize,
                                          data, dataSize, 9);

    if (ZSTD_isError(compressedSize)) {
        throw std::runtime_error("Error: " + std::string(ZSTD_getErrorName(compressedSize)));
    }

    compressedData.resize(compressedSize);  
    return compressedData;
}


void decompress_data(void* compressed_data, size_t compressed_size, float** decompressed_data, size_t original_size) {
    // Asignar memoria para los datos descomprimidos
    *decompressed_data = (float *)malloc(original_size);

    if (*decompressed_data == NULL) {
        return;
    }

    // Descomprimir los datos
    size_t decompressed_size = ZSTD_decompress(
        *decompressed_data, original_size, // Destino y tamaño original
        compressed_data, compressed_size   // Fuente y tamaño comprimido
    );

    if (ZSTD_isError(decompressed_size)) {
        free(*decompressed_data);
        return;
    }

}


void GenerateBlocks(const char *tmp,long dim, long nb, int ** dev, int p, int q,int r,size_t* nro_size){
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
		    map(from:nro_size[i*nro_blocks+j:1])\
                    depend(inout:vector[node_local]) \
                    device(node_local)
                {
                    int MyPE, NumPEs;
                    int  ierr;
		    ierr = MPI_Comm_size(MPI_COMM_WORLD, &NumPEs);
                    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
                    
                    int nx, ny, mx, my, ng;

                    nx = nb*PORC;
                    ny = nb*PORC;
                    mx = dim*PORC;
                    my = dim*PORC;
                    ng = 0;

                    float *data;
                    data = (float *) malloc(sizeof(float )*nb*nb);

                    FillBlock(nb, nb, i, j, dim, nb, data);

		    std::vector<uint8_t> compressedData = compressData(data, nb*nb);
		    free(data);

		    size_t compressed_size = compressedData.size();
		    nro_size[i*nro_blocks+j] = compressed_size;
                    
		    int rank;
                    hsize_t dimens_2d[2];
                    hsize_t chunk_dims[2];
                    hid_t err;

                    hsize_t start_2d[2];
                    hsize_t stride_2d[2];
                    hsize_t count_2d[2];
		    hsize_t block_2d[2];

                    hsize_t remaining_start_2d[2];
                    hsize_t remaining_stride_2d[2];
                    hsize_t remaining_count_2d[2];
                    hsize_t remaining_block_2d[2];

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

		    hsize_t SUBCHUNK_DIM0 = compressed_size/nx;
		    hsize_t SUBCHUNK_DIM1 = ny;

                    count_2d[0] = SUBCHUNK_DIM0;
                    count_2d[1] = 1;

		    block_2d[0] = 1;
		    block_2d[1] = SUBCHUNK_DIM1;
	
		    err = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start_2d, stride_2d, count_2d, block_2d);

		    remaining_start_2d[0] = i_*ny+SUBCHUNK_DIM0;
                    remaining_start_2d[1] = j_*nx;

                    remaining_stride_2d[0] = 1;
                    remaining_stride_2d[1] = 1;

                    remaining_count_2d[0] = 1;
                    remaining_count_2d[1] = 1;

                    remaining_block_2d[0] = 1;
                    remaining_block_2d[1] = compressed_size-SUBCHUNK_DIM0*SUBCHUNK_DIM1;

		    err = H5Sselect_hyperslab(dataspace, H5S_SELECT_OR, remaining_start_2d, remaining_stride_2d, remaining_count_2d, remaining_block_2d);

		    hsize_t mem_count[1] = {compressed_size};

                    memspace = H5Screate_simple(1, mem_count, NULL);
                   
                    acc_template = H5Pcreate(H5P_DATASET_XFER);
                    ierr = H5Pset_dxpl_mpio(acc_template, H5FD_MPIO_INDEPENDENT);
		    hssize_t num_elements = H5Sget_select_npoints(dataspace);
                    
                    err = H5Dwrite(dataset, H5T_NATIVE_UCHAR, memspace, dataspace, acc_template, compressedData.data());

                    H5Pclose(acc_template);
                    H5Sclose(memspace);

                    H5Sclose(dataspace);
                    H5Dclose(dataset);
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

/*void ReadBlock(long dim, long nb, long i, long j, hid_t file_id, size_t compressed_size, float** data){
    hid_t acc_template;
    hid_t dataspace, memspace, dataset;
                
    int rank = 2, ierr;
                
    dataset = H5Dopen(file_id, "/data", H5P_DEFAULT); 

    dataspace = H5Dget_space(dataset);

    size_t size = nb*PORC;
    hsize_t SUBCHUNK_DIM0 = compressed_size/size;
    hsize_t SUBCHUNK_DIM1 = size;

    int nro_blocks = dim/nb;
    int id_blocks = j*nro_blocks+i-(j*(j+1)/2);
    int i_ = id_blocks / nro_blocks;
    int j_ = id_blocks - i_*nro_blocks;

    hsize_t start_2d[2] = {i_*size, j_*size};  
    hsize_t count_2d[2] = {SUBCHUNK_DIM0, 1};
    hsize_t block_2d[2] = {1,SUBCHUNK_DIM1};
   
    ierr = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start_2d, NULL, count_2d, block_2d);

    hsize_t remaining_start_2d[2] = {i_*size+SUBCHUNK_DIM0, j_*size};
    hsize_t remaining_count_2d[2] = {1, 1};
    hsize_t remaining_block_2d[2] = {1,compressed_size -SUBCHUNK_DIM0*SUBCHUNK_DIM1};

    ierr = H5Sselect_hyperslab(dataspace, H5S_SELECT_OR, remaining_start_2d, NULL, remaining_count_2d, remaining_block_2d);

    hsize_t mem_count[1] = {compressed_size};
    memspace = H5Screate_simple(1, mem_count, NULL);

    acc_template = H5Pcreate(H5P_DATASET_XFER);
    ierr = H5Pset_dxpl_mpio(acc_template, H5FD_MPIO_INDEPENDENT);

    void *compressed_data = (void *)malloc(compressed_size * sizeof(char));

    H5Dread(dataset, H5T_NATIVE_UCHAR, memspace, dataspace, acc_template, compressed_data);

    H5Pclose(acc_template);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);

    decompress_data(compressed_data, compressed_size, data, nb * nb * sizeof(float));

}*/

void ReadBlock(long dim, long nb, long i, long j, hid_t file_id, size_t compressed_size, float** data){
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

    void *read_buffer = (void *)malloc(SUBCHUNK_DIM0*SUBCHUNK_DIM1 * sizeof(char));

    H5Dread(dataset, H5T_NATIVE_UCHAR, memspace, dataspace, acc_template, read_buffer);

    H5Pclose(acc_template);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);

    void* compressed_data = malloc(compressed_size);
    memcpy(compressed_data, read_buffer, compressed_size);

    free(read_buffer);

    decompress_data(compressed_data, compressed_size, data, nb * nb * sizeof(float));

    free(compressed_data);
}

/*size_t WriteBlock(long dim, long nb, int i, int j, hid_t file_id, float* data){

    long size = nb*PORC;
    int rank, ng =0 ,ierr;
    hsize_t start_2d[2];
    hsize_t stride_2d[2];
    hsize_t count_2d[2];
    hsize_t block_2d[2];

    hsize_t remaining_start_2d[2];
    hsize_t remaining_stride_2d[2];
    hsize_t remaining_count_2d[2];
    hsize_t remaining_block_2d[2];
   
    hid_t dataspace, memspace, dataset;
    hid_t err;
    hid_t acc_template, plist_id;

    std::vector<uint8_t> comData = compressData(data, nb*nb);
    size_t compressed_size = comData.size();

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

    hsize_t SUBCHUNK_DIM0 = compressed_size/size;
    hsize_t SUBCHUNK_DIM1 = size;

    count_2d[0] = SUBCHUNK_DIM0;
    count_2d[1] = 1;

    block_2d[0] = 1;
    block_2d[1] = SUBCHUNK_DIM1;

    err = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start_2d, stride_2d, count_2d, block_2d);

    remaining_start_2d[0] = i_*size+SUBCHUNK_DIM0;
    remaining_start_2d[1] = j_*size;

    remaining_stride_2d[0] = 1;
    remaining_stride_2d[1] = 1;

    remaining_count_2d[0] = 1;
    remaining_count_2d[1] = 1;

    remaining_block_2d[0] = 1;
    remaining_block_2d[1] = compressed_size-SUBCHUNK_DIM0*SUBCHUNK_DIM1;

    err = H5Sselect_hyperslab(dataspace, H5S_SELECT_OR, remaining_start_2d, remaining_stride_2d, remaining_count_2d, remaining_block_2d);

    hsize_t mem_count[1] = {compressed_size};
    memspace = H5Screate_simple(1, mem_count, NULL);

    acc_template = H5Pcreate(H5P_DATASET_XFER);
    ierr = H5Pset_dxpl_mpio(acc_template, H5FD_MPIO_INDEPENDENT);

    ierr = H5Dwrite(dataset, H5T_NATIVE_UCHAR, memspace, dataspace, acc_template, comData.data());

    H5Pclose(acc_template);
    H5Sclose(memspace);

    H5Sclose(dataspace);
    H5Dclose(dataset);

    return compressed_size;

}*/

size_t WriteBlock(long dim, long nb, int i, int j, hid_t file_id, float* data){

    long size = nb*PORC;
    int rank, ng =0 ,ierr;
    hsize_t start_2d[2];
    hsize_t stride_2d[2];
    hsize_t count_2d[2];

    hsize_t dimens_2d[2];
    hid_t dataspace, memspace, dataset;
    hid_t err;
    hid_t acc_template, plist_id;

    std::vector<uint8_t> comData = compressData(data, nb*nb);
    size_t compressed_size = comData.size();

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

    hsize_t SUBCHUNK_DIM0 = (compressed_size + size -1)/(size);
    hsize_t SUBCHUNK_DIM1 = size;

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

    comData.resize(SUBCHUNK_DIM0*SUBCHUNK_DIM1,0);

    ierr = H5Dwrite(dataset, H5T_NATIVE_UCHAR, memspace, dataspace, acc_template, comData.data());

    H5Pclose(acc_template);
    H5Sclose(memspace);

    H5Sclose(dataspace);
    H5Dclose(dataset);

    return compressed_size;

}

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
                firstprivate(k,nb,dev_spotrf,dim) \
                map(to:file_id[dev_spotrf:1])\
                device(dev_spotrf)
            {
                //printf("Potrf before inout: A(%ld,%ld) -> %zu\n",k,k,Block_Diag[0]);
                float *BDiag = NULL;
                ReadBlock(dim,nb,k,k, file_id[dev_spotrf],Block_Diag[0] , &BDiag);
                LAPACKE_spotrf_work(LAPACK_COL_MAJOR, 'L', nb, BDiag, nb);
                Block_Diag[0] = WriteBlock(dim,nb,k,k, file_id[dev_spotrf], BDiag);
		//print_tile("A",BDiag,10,10);
                free(BDiag);
                //printf("Potrf after inout: A(%ld,%ld)  -> %zu\n",k,k,Block_Diag[0]);
            }
	    for (long m = k + 1; m < nro_blocks; m++) {
                size_t nroMK = m*nro_blocks+k;
                int dev_strsm = (int)a[nroMK];
                Block_Diag = &nro_size[nroDiag];
                size_t *Block_mk = &nro_size[nroMK];
                #pragma omp target nowait  depend(in: *Block_Diag) depend(inout: *Block_mk) \
                    depend(inout:vector[dev_strsm])\
                    firstprivate(k,m,nb,dev_strsm,dim) \
                    map(to:file_id[dev_strsm:1])\
                    device(dev_strsm)
                {
                    //printf("    Strsm before inout: A(%ld,%ld) -> %zu in: A(%ld,%ld)  -> %zu\n",m,k,Block_mk[0],k,k,Block_Diag[0]);
                    float *BDiag = NULL; //(float*)malloc(nb*nb*sizeof(float));
                    float *Bmk = NULL;//(float*)malloc(nb*nb*sizeof(float));
                    ReadBlock(dim,nb,k,k, file_id[dev_strsm], Block_Diag[0], &BDiag);
                    ReadBlock(dim,nb,m,k, file_id[dev_strsm], Block_mk[0], &Bmk);

                    cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,CblasNonUnit, nb, nb, 1.0, BDiag, nb, Bmk, nb);

                    free(BDiag);
                    Block_mk[0] =  WriteBlock(dim,nb,m,k, file_id[dev_strsm],Bmk);
                    free(Bmk);
                    //printf("    Strsm after inout: A(%ld,%ld) -> %zu in: A(%ld,%ld)  -> %zu\n",m,k,Block_mk[0],k,k,Block_Diag[0]);
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
                    firstprivate(m,k,nb,dev_ssyrk,dim)\
                    map(to:file_id[dev_ssyrk:1]) \
                    device(dev_ssyrk)
                {
                    //printf("    Ssyrk before inout: A(%ld,%ld)-> %zu in: A(%ld,%ld) -> %zu\n",m,m,Block_mm[0],m,k,Block_mk[0]);
                    float *Bmk = NULL;//(float*)malloc(nb*nb*sizeof(float));
                    float *Bmm = NULL;//(float*)malloc(nb*nb*sizeof(float));
                    ReadBlock(dim,nb,m,k, file_id[dev_ssyrk], Block_mk[0], &Bmk);
                    ReadBlock(dim,nb,m,m, file_id[dev_ssyrk], Block_mm[0], &Bmm);
                    cblas_ssyrk(CblasColMajor, CblasLower, CblasNoTrans, nb, nb, -1.0, Bmk, nb, 1.0, Bmm, nb);
                    free(Bmk);
                    Block_mm[0] =  WriteBlock(dim,nb,m,m, file_id[dev_ssyrk], Bmm);
                    free(Bmm);
                    //printf("    Ssyrk after inout: A(%ld,%ld) -> %zu in: A(%ld,%ld) -> %zu\n",m,m,Block_mm[0],m,k,Block_mk[0]);
                }

                for (long n = k + 1; n < m; n++) {
                    size_t nroNK = n*nro_blocks+k;
                    size_t nroMN = m*nro_blocks+n;
                    int dev_sgemm = (int)a[nroMN];
                    size_t *Block_nk = &nro_size[nroNK];//A.GetBlock(n, k);
                    size_t *Block_mn = &nro_size[nroMN];//A.GetBlock(m, n);
                    #pragma omp target nowait depend(in: *Block_mk, *Block_nk) depend(inout: *Block_mn)\
                        depend(inout:vector[dev_sgemm])\
                        firstprivate(m,n,k,nb,dev_sgemm,dim) \
                        map(to:file_id[dev_sgemm:1]) \
                        device(dev_sgemm)
                    {
                        //printf("        Sgemm before inout: A(%ld,%ld) -> %zu in: A(%ld,%ld) -> %zu A(%ld,%ld) -> %zu\n",m,n,Block_mn[0],m,k,Block_mk[0],n,k,Block_nk[0]);
                        float *Bmk = NULL;//(float*)malloc(nb*nb*sizeof(float));
                        float *Bnk = NULL;//(float*)malloc(nb*nb*sizeof(float));
                        float *Bmn = NULL;//(float*)malloc(nb*nb*sizeof(float));
                        ReadBlock(dim,nb,m,k, file_id[dev_sgemm], Block_mk[0], &Bmk);
                        ReadBlock(dim,nb,n,k, file_id[dev_sgemm], Block_nk[0], &Bnk);
                        ReadBlock(dim,nb,m,n, file_id[dev_sgemm], Block_mn[0], &Bmn);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, nb, nb, nb, -1.0, Bmk, nb, Bnk, nb, 1.0, Bmn, nb);
                        free(Bmk);
                        free(Bnk);
                        Block_mn[0] = WriteBlock(dim,nb,m,n, file_id[dev_sgemm],Bmn);
                        free(Bmn);
                        //printf("        Sgemm after inout: A(%ld,%ld) -> %zu in: A(%ld,%ld) -> %zu A(%ld,%ld) -> %zu\n",m,n,Block_mn[0],m,k,Block_mk[0],n,k,Block_nk[0]);
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

    long nro_blocks = N/BS;
    size_t* nro_size = (size_t *)malloc(nro_blocks*nro_blocks* sizeof(size_t));

    t = omp_get_wtime();
    GenerateBlocks(tmp,N,BS,dev,workers,threads,r,nro_size);
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
    Cholesky_Factorization(a,tmp,N,BS,workers,threads,r,nro_size);
    t = omp_get_wtime() - t;
    printf("Cholesky Factorization time %0.6lf\n",t);

    free(dev);
    free(a);

    return 0;
}
