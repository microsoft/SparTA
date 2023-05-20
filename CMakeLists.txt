cmake_minimum_required(VERSION 3.13)

project(NMSPARSE CUDA CXX C)


# Requires C++11.
set(CMAKE_CXX_STANDARD 11)

# #
# ## Project build.
# #
include(cmake/CUDA.cmake)
cuda_find_library(CUDART_LIBRARY cudart_static)
cuda_find_library(CUBLAS_LIBRARY cublas_static)
cuda_find_library(CUSPARSE_LIBRARY cusparse_static)
include_directories(${CUDART_INCLUDE_DIR})

list(APPEND NMSPARSE_LIBS "cudart_static;cublas_static;cusparse_static;culibos;cublasLt_static")

include_directories("${PROJECT_SOURCE_DIR}")
add_subdirectory(nmsparse)
