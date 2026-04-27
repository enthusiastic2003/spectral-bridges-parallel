cmake_minimum_required(VERSION 3.20)
project(spectral_bridges CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Guard against multi-config generators (Visual Studio, Ninja Multi-Config)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(PYMODULE "Build Python extension module (specbridge)" OFF)
option(USE_OPENMP_LLVM "Use /openmp:llvm instead of /openmp on MSVC (requires VS 2019 16.9+)" ON)
option(USE_CUDA "Build with CUDA support (GPU acceleration)" OFF)

# --- 1. Compiler Flags ---
if(MSVC)
    add_compile_options(
        /O2 /arch:AVX2 /MP /EHsc /bigobj /W3
        /wd4127 /wd4244 /wd4305
    )
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    add_compile_options(-Wall -Wextra)
    add_compile_options("$<$<CONFIG:Release>:-O3;-march=native>")
    add_compile_options("$<$<CONFIG:Debug>:-g>")
endif()

# --- 2. Find Dependencies ---
find_package(Eigen3 REQUIRED)

# FIX: Just log the Eigen dir here. We will attach it to the target in Section 3.
if(DEFINED EIGEN3_INCLUDE_DIR)
  message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
endif()

# --- 2a. OpenMP ---
find_package(OpenMP REQUIRED)

if(MSVC)
    if(MSVC_VERSION LESS 1929)
        if(USE_OPENMP_LLVM)
            message(WARNING
                "USE_OPENMP_LLVM requires VS 2019 16.9+ (MSVC_VERSION >= 1929). "
                "Falling back to /openmp (OpenMP 2.0).")
            set(USE_OPENMP_LLVM OFF CACHE BOOL "" FORCE)
        endif()
    endif()

    if(USE_OPENMP_LLVM)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
            INTERFACE_COMPILE_OPTIONS "/openmp:llvm"
        )
        message(STATUS "OpenMP: /openmp:llvm (OpenMP 3.1, LLVM runtime)")
    else()
        message(STATUS "OpenMP: /openmp (OpenMP 2.0, legacy MSVC runtime)")
    endif()
else()
    message(STATUS "OpenMP: ${OpenMP_CXX_FLAGS}")
endif()

# --- 2b. BLAS ---
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    set(BLAS_LIBS ${ACCELERATE_FRAMEWORK})
elseif(WIN32)
    find_package(OpenBLAS QUIET)
    if(OpenBLAS_FOUND)
        set(BLAS_LIBS OpenBLAS::OpenBLAS)
        message(STATUS "BLAS: OpenBLAS (${OpenBLAS_DIR})")
    else()
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            set(BLAS_LIBS ${BLAS_LIBRARIES})
            message(STATUS "BLAS: ${BLAS_LIBRARIES}")
        else()
            message(WARNING
                "No BLAS found on Windows. Eigen will use its own fallback "
                "routines. Install OpenBLAS via vcpkg or conda for best "
                "performance.")
            set(BLAS_LIBS "")
        endif()
    endif()
elseif(UNIX)
    find_package(BLAS REQUIRED)
    set(BLAS_LIBS ${BLAS_LIBRARIES})
endif()

# --- 2c. CUDA (optional) ---
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA target architectures")
    endif()

    string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr")
    string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -O3")

    message(STATUS "CUDA: enabled, arch=${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA: disabled")
endif()

# --- 3. Core Static Library ---
set(SB_CORE_SOURCES
    src/kmeans.cpp
    src/affinity.cpp
    src/spectral.cpp
)

if(USE_CUDA)
    list(APPEND SB_CORE_SOURCES
        src/kmeans_cuda.cu
        src/affinity.cu
        src/spectral.cu
    )
endif()

# FIX: Now that SB_CORE_SOURCES is fully populated, we create the library.
add_library(sb_core STATIC ${SB_CORE_SOURCES})

set_property(TARGET sb_core PROPERTY POSITION_INDEPENDENT_CODE ON)

# FIX: Now that sb_core exists, we can attach include directories to it.
target_include_directories(sb_core PUBLIC include)

if(DEFINED EIGEN3_INCLUDE_DIR)
  target_include_directories(sb_core PUBLIC ${EIGEN3_INCLUDE_DIR})
endif()

target_link_libraries(sb_core
    PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    ${BLAS_LIBS}
)

if(USE_CUDA)
    target_link_libraries(sb_core PUBLIC
        CUDA::cudart
        CUDA::cublas
    )
    target_compile_definitions(sb_core PUBLIC SB_HAVE_CUDA)

    set_target_properties(sb_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

if(USE_CUDA)
    target_link_libraries(sb_core PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
    )
    target_compile_definitions(sb_core PUBLIC SB_HAVE_CUDA)

    set_target_properties(sb_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# --- 4. Python Extension Module ---
if(PYMODULE)
    find_package(pybind11 REQUIRED)
    if(USE_CUDA)
        pybind11_add_module(specbridge MODULE NO_EXTRAS python/bindings.cpp)
    else()
        pybind11_add_module(specbridge python/bindings.cpp)
    endif()
    target_link_libraries(specbridge PRIVATE sb_core)

    if(USE_CUDA)
        # Avoid GCC/NVCC LTO plugin version mismatches at CUDA device-link time.
        set_target_properties(specbridge PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
    endif()

    if(WIN32)
        set_target_properties(specbridge PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
    endif()
endif()

# --- 5. Executables ---
add_executable(spectral_bridges main.cpp)
target_link_libraries(spectral_bridges sb_core)

add_executable(test_kmeans tests/test_kmeans.cpp)
target_link_libraries(test_kmeans sb_core)

add_executable(test_reference tests/test_reference.cpp)
target_link_libraries(test_reference sb_core)

if(USE_CUDA)
    add_executable(test_kmeans_cuda tests/test_kmeans_cuda.cpp)
    target_link_libraries(test_kmeans_cuda sb_core)
endif()

# --- 6. Windows: copy runtime DLLs next to each executable ---
if(WIN32)
    set(WIN_EXE_TARGETS spectral_bridges test_kmeans test_reference)
    if(USE_CUDA)
        list(APPEND WIN_EXE_TARGETS test_kmeans_cuda)
    endif()
    foreach(tgt ${WIN_EXE_TARGETS})
        add_custom_command(TARGET ${tgt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:${tgt}>>,copy_if_different,true>
                $<TARGET_RUNTIME_DLLS:${tgt}>
                $<TARGET_FILE_DIR:${tgt}>
            COMMAND_EXPAND_LISTS
        )
    endforeach()
endif()