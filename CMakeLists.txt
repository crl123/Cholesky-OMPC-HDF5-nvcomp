cmake_minimum_required(VERSION 3.22)

project(Cholesky_OMPC
    VERSION         1.0
    DESCRIPTION     "Cholesky with OMPC+HDF5+nvcomp"
    LANGUAGES       C CXX CUDA)

set(CMAKE_C_STANDARD   17)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Ensure we are using OMPC Clang
# ------------------------------------------------------------------------------

if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
    execute_process(
        COMMAND ${CMAKE_C_COMPILER} --version
        OUTPUT_VARIABLE CLANG_VERSION)
    if(CLANG_VERSION MATCHES "^OmpCluster")
        message(STATUS
            "Using OMPC Clang version ${CMAKE_C_COMPILER_VERSION}!")
    else()
        message(FATAL_ERROR
            "It seems you are using vanilla Clang instead of OMPC Clang. "
            "Please check your installation and try again!")
    endif()
else()
    message(FATAL_ERROR
        "Not using OMPC Clang as compiler, found: ${CMAKE_C_COMPILER_ID}.\n"
        ""
        "Make sure to export the following variables and call CMake from within"
        " the container.\n"
        ""
        "    export CC=clang\n"
        "    export CXX=clang++")
endif()


add_library(OMPC::Runtime IMPORTED INTERFACE)

set_property(
    TARGET OMPC::Runtime
    PROPERTY
        INTERFACE_COMPILE_OPTIONS -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu)

set_property(
    TARGET OMPC::Runtime
    PROPERTY
        INTERFACE_LINK_OPTIONS -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu)

# BLAS
include(CheckSymbolExists)
find_package( BLAS REQUIRED )
find_package( LAPACK REQUIRED )

message( STATUS "BLAS(${BLAS_FOUND})" ) 
message( STATUS "LAPACK(${LAPACK_FOUND})" )

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/Modules")

cmake_policy(PUSH)
cmake_policy(SET CMP0074 NEW) # allows to use CBLAS_ROOT and LAPACKE_ROOT

find_package( CBLAS REQUIRED )
include_directories(${CBLAS_INCLUDE_DIRS})

find_package( LAPACKE REQUIRED )
include_directories(${LAPACKE_INCLUDE_DIRS})

cmake_policy(POP)

set(LINALG_LIBRARIES ${LAPACKE_LIBRARIES} ${LAPACK_LIBRARIES} ${CBLAS_LIBRARIES} ${BLAS_LIBRARIES})

find_library(MATH_LIBRARY m)
if( MATH_LIBRARY )
  set(PLASMA_LIBRARIES ${PLASMA_LINALG_LIBRARIES} ${MATH_LIBRARY})
endif( MATH_LIBRARY )

find_package(HDF5 REQUIRED COMPONENTS C HL)
if (HDF5_FOUND)
    message(STATUS "HDF5 found: ${HDF5_INCLUDE_DIRS}")
    include_directories(${HDF5_INCLUDE_DIRS})
    add_definitions(${HDF5_DEFINITIONS})
else()
    message(FATAL_ERROR "HDF5 no encontrado")
endif()

include_directories(${HDF5_INCLUDE_DIRS})

find_path(ZSTD_INCLUDE_DIR
  NAMES zstd.h
  PATHS ${ZSTD_ROOT}/include
  NO_DEFAULT_PATH
)
find_library(ZSTD_LIBRARY
  NAMES zstd
  PATHS ${ZSTD_ROOT}/lib
  NO_DEFAULT_PATH
)
if (NOT ZSTD_INCLUDE_DIR OR NOT ZSTD_LIBRARY)
  message(FATAL_ERROR "Not founded Zstandard. Put set ZSTD_ROOT.")
endif()
include_directories(${ZSTD_INCLUDE_DIR})

find_package(CUDA REQUIRED)
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} cusolver)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(/usr/local/cuda/lib64)

set(CMAKE_PREFIX_PATH ${NVCOMP_SRC_DIR}/lib/cmake/nvcomp)

find_package(nvcomp REQUIRED)

# Define as bibliotecas e flags para CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations")

find_package(MPI REQUIRED COMPONENTS CXX)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${CMAKE_SOURCE_DIR}/include)

include_directories(${NVCOMP_PATH}/include/nvcomp)
link_directories(${NVCOMP_PATH}/lib)

add_executable(zstd gpu_zstd.cpp )
target_link_libraries(zstd OMPC::Runtime)
target_link_libraries(zstd ${LINALG_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_CXX_LIBRARIES}  ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})
target_link_libraries(zstd nvcomp -lnvToolsExt -lcuda -lcudart -lcublas )
target_include_directories(zstd PRIVATE ${CUDA_INCLUDE_DIRS})

add_executable(ans gpu_ans.cpp )
target_link_libraries(ans OMPC::Runtime)
target_link_libraries(ans ${LINALG_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_CXX_LIBRARIES}  ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})
target_link_libraries(ans nvcomp -lnvToolsExt -lcuda -lcudart -lcublas )
target_include_directories(ans PRIVATE ${CUDA_INCLUDE_DIRS})

add_executable(lz4 gpu_lz4.cpp )
target_link_libraries(lz4 OMPC::Runtime)
target_link_libraries(lz4 ${LINALG_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_CXX_LIBRARIES}  ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})
target_link_libraries(lz4 nvcomp -lnvToolsExt -lcuda -lcudart -lcublas )
target_include_directories(lz4 PRIVATE ${CUDA_INCLUDE_DIRS})

add_executable(snappy gpu_snappy.cpp )
target_link_libraries(snappy OMPC::Runtime)
target_link_libraries(snappy ${LINALG_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_CXX_LIBRARIES}  ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})
target_link_libraries(snappy nvcomp -lnvToolsExt -lcuda -lcudart -lcublas )
target_include_directories(snappy PRIVATE ${CUDA_INCLUDE_DIRS})

add_executable(bitcomp gpu_bitcomp.cpp )
target_link_libraries(bitcomp OMPC::Runtime)
target_link_libraries(bitcomp ${LINALG_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_CXX_LIBRARIES}  ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})
target_link_libraries(bitcomp nvcomp -lnvToolsExt -lcuda -lcudart -lcublas )
target_include_directories(bitcomp PRIVATE ${CUDA_INCLUDE_DIRS})

add_executable(uncomp gpu.cpp )
target_link_libraries(uncomp OMPC::Runtime)
target_link_libraries(uncomp ${LINALG_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_CXX_LIBRARIES}  ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})
target_link_libraries(uncomp nvcomp -lnvToolsExt -lcuda -lcudart -lcublas )
target_include_directories(uncomp PRIVATE ${CUDA_INCLUDE_DIRS})
