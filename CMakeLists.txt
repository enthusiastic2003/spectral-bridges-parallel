cmake_minimum_required(VERSION 3.20)
project(spectral_bridges CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Remove -fopenmp from here — OpenMP::OpenMP_CXX handles it properly
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

option(PYMODULE "Build Python extension module (specbridge)" OFF)

# --- 1. Find Dependencies ---
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)  # <-- moved up, needed by core

if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    set(BLAS_LIBS ${ACCELERATE_FRAMEWORK})
elseif(UNIX)
    find_package(BLAS REQUIRED)
    set(BLAS_LIBS ${BLAS_LIBRARIES})
endif()

include_directories(include)

# --- 2. Core C++ Logic (Static Library) ---
set(SB_CORE_SOURCES
    src/kmeans.cpp
    src/affinity.cpp
    src/spectral.cpp
)

add_library(sb_core STATIC ${SB_CORE_SOURCES})
set_property(TARGET sb_core PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(sb_core
    PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX  # <-- OpenMP on the library that actually uses it
    ${BLAS_LIBS}
)

# --- 3. Python Extension Module ---
if(PYMODULE)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(specbridge python/bindings.cpp)
    target_link_libraries(specbridge PRIVATE sb_core)
    # specbridge gets OpenMP transitively via sb_core PUBLIC linkage
endif()

# --- 4. Executables ---
add_executable(spectral_bridges main.cpp)
target_link_libraries(spectral_bridges sb_core)

add_executable(test_kmeans tests/test_kmeans.cpp)
target_link_libraries(test_kmeans sb_core)

add_executable(test_reference tests/test_reference.cpp)
target_link_libraries(test_reference sb_core)