cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

project(NMSPARSE CUDA CXX C)


# Requires C++14.
set(CMAKE_CXX_STANDARD 14)

# #
# ## Project build.
# #
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/cutlass/include")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/cutlass/examples/common")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/cutlass/examples/43_gemm_block_sparse")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/cutlass/tools/util/include")

include(cmake/FindCuda.cmake)

add_subdirectory(nmsparse)

file(GLOB GTEST_SOURCE thirdparty/gtest/gtest-all.cc)
file(GLOB GTEST_HEADER thirdparty/gtest/gtest.h)
add_library(googletest-gtest ${GTEST_HEADER} ${GTEST_SOURCE})

add_subdirectory(tests)

