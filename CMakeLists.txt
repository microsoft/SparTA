cmake_minimum_required(VERSION 3.13)

project(NMSPARSE CUDA CXX C)


# Requires C++11.
set(CMAKE_CXX_STANDARD 11)

##
### Project build.
##

include_directories("${PROJECT_SOURCE_DIR}")
add_subdirectory(nmsparse)
