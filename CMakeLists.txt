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
include(cmake/FindCuda.cmake)

add_subdirectory(nmsparse)

# build test
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/")
file(GLOB GTEST_SOURCE thirdparty/gtest/gtest-all.cc)
file(GLOB GTEST_HEADER thirdparty/gtest/gtest.h)
add_library(googletest-gtest ${GTEST_HEADER} ${GTEST_SOURCE})

add_subdirectory(tests)

