cmake_minimum_required(VERSION 3.20)
project(spectral_bridges CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build by default for accurate benchmarking
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# Eigen (installed via brew on Apple Silicon)
find_package(Eigen3 REQUIRED)

# Apple Accelerate framework (gives us BLAS for free on macOS)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

# Shared include directory
include_directories(include)

# Core library (kmeans + affinity + spectral)
add_library(sb_core
    src/kmeans.cpp
    src/affinity.cpp
    src/spectral.cpp
)

target_link_libraries(sb_core
    Eigen3::Eigen
    ${ACCELERATE_FRAMEWORK}
)

# Main benchmark executable
add_executable(spectral_bridges main.cpp)
target_link_libraries(spectral_bridges sb_core)

# Tests
add_executable(test_kmeans tests/test_kmeans.cpp)
target_link_libraries(test_kmeans sb_core)

add_executable(test_reference tests/test_reference.cpp)
target_link_libraries(test_reference sb_core)