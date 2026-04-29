cmake_minimum_required(VERSION 3.20)
project(spectral_bridges CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Guard against multi-config generators
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- 1. Compiler Flags ---
# MSVC flags have been completely removed.
add_compile_options(-Wall -Wextra)
add_compile_options("$<$<CONFIG:Release>:-O3;-march=native>")
add_compile_options("$<$<CONFIG:Debug>:-g>")

# --- 2. Find Dependencies ---
find_package(Eigen3 REQUIRED)

if(DEFINED EIGEN3_INCLUDE_DIR)
  message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
endif()

# --- 2a. OpenMP ---
find_package(OpenMP REQUIRED)
message(STATUS "OpenMP: ${OpenMP_CXX_FLAGS}")

# --- 2b. BLAS / LAPACK / LAPACKE ---
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    set(BLAS_LIBS ${ACCELERATE_FRAMEWORK})
    set(HAVE_LAPACKE OFF)
elseif(UNIX AND NOT APPLE)
    if(BLAS_LIBRARIES)
        set(OPENBLAS_LIB ${BLAS_LIBRARIES})
    else()
        # Cleaned up hardcoded HINTS; relying on standard system paths or user-provided variables
        find_library(OPENBLAS_LIB NAMES openblas)
    endif()

    find_path(LAPACKE_INCLUDE_DIR NAMES lapacke.h)

    if(OPENBLAS_LIB AND LAPACKE_INCLUDE_DIR)
        set(BLAS_LIBS ${OPENBLAS_LIB})
        set(HAVE_LAPACKE ON)
        message(STATUS "BLAS/LAPACK/LAPACKE: OpenBLAS at ${OPENBLAS_LIB}")
        message(STATUS "LAPACKE include: ${LAPACKE_INCLUDE_DIR}")
    else()
        set(BLAS_LIBS ${OPENBLAS_LIB})
        set(HAVE_LAPACKE OFF)
        message(WARNING
            "LAPACKE headers or OpenBLAS lib not found. "
            "OPENBLAS_LIB=${OPENBLAS_LIB} "
            "LAPACKE_INCLUDE_DIR=${LAPACKE_INCLUDE_DIR}")
    endif()
endif()

# --- 2c. CUDA ---
find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA target architectures")
endif()

string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr")
string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -O3")
message(STATUS "CUDA: enabled, arch=${CMAKE_CUDA_ARCHITECTURES}")

# --- 3. Core Static Library ---
set(SB_CORE_SOURCES
    src/kmeans.cpp
    src/affinity.cpp
    src/spectral.cpp
    src/kmeans_cuda.cu
    src/affinity.cu
    src/spectral.cu
)

add_library(sb_core STATIC ${SB_CORE_SOURCES})
set_property(TARGET sb_core PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(sb_core PUBLIC include)

if(DEFINED EIGEN3_INCLUDE_DIR)
  target_include_directories(sb_core PUBLIC ${EIGEN3_INCLUDE_DIR})
endif()

# Spectra: header-only partial-spectrum eigensolver
# Replaced hardcoded path with standard find_path
find_path(SPECTRA_INCLUDE_DIR NAMES Spectra/SymEigsSolver.h)
if(NOT SPECTRA_INCLUDE_DIR)
    message(FATAL_ERROR "Spectra not found. Please provide -DSPECTRA_INCLUDE_DIR=/path/to/spectra/include")
endif()
target_include_directories(sb_core PUBLIC ${SPECTRA_INCLUDE_DIR})
message(STATUS "Spectra: ${SPECTRA_INCLUDE_DIR}")

target_link_libraries(sb_core
    PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    ${BLAS_LIBS}
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)

# Route Eigen through LAPACK for parallelized eigensolvers etc.
if(HAVE_LAPACKE)
    target_compile_definitions(sb_core PUBLIC EIGEN_USE_LAPACKE EIGEN_USE_BLAS)
    if(LAPACKE_INCLUDE_DIR)
        target_include_directories(sb_core PUBLIC ${LAPACKE_INCLUDE_DIR})
    endif()
    message(STATUS "Eigen: using LAPACKE backend")
else()
    message(STATUS "Eigen: using built-in routines (no LAPACKE)")
endif()

target_compile_definitions(sb_core PUBLIC SB_HAVE_CUDA)
set_target_properties(sb_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- 4. Python Extension Module ---
find_package(pybind11 REQUIRED)
pybind11_add_module(specbridge MODULE NO_EXTRAS python/bindings.cpp)
target_link_libraries(specbridge PRIVATE sb_core)

# Avoid GCC/NVCC LTO plugin version mismatches at CUDA device-link time.
set_target_properties(specbridge PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

# --- 5. Executables ---
add_executable(spectral_bridges main.cpp)
target_link_libraries(spectral_bridges sb_core)

add_executable(test_kmeans tests/test_kmeans.cpp)
target_link_libraries(test_kmeans sb_core)

add_executable(test_reference tests/test_reference.cpp)
target_link_libraries(test_reference sb_core)

add_executable(test_kmeans_cuda tests/test_kmeans_cuda.cpp)
target_link_libraries(test_kmeans_cuda sb_core)