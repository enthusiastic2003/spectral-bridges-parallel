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

# --- 1. Compiler Flags ---
if(MSVC)
    # /O2        - optimize for speed
    # /arch:AVX2 - vectorization (portable equivalent of -march=native; drop if
    #              targeting older hardware without AVX2)
    # /MP        - parallel compilation across translation units
    # /EHsc      - standard C++ exception handling (required by Eigen & pybind11,
    #              and mandatory when using any OpenMP variant on MSVC)
    # /bigobj    - Eigen's heavy templates exceed the default .obj symbol limit
    # /W3        - reasonable warning level
    # /wd4127    - "conditional expression is constant" (Eigen internals)
    # /wd4244    - narrowing conversions (Eigen internals)
    # /wd4305    - truncation from double (Eigen internals)
    add_compile_options(
        /O2 /arch:AVX2 /MP /EHsc /bigobj /W3
        /wd4127 /wd4244 /wd4305
    )
    # Prevent <windows.h> from defining min/max macros that collide with std::
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    add_compile_options(-Wall -Wextra)
    add_compile_options("$<$<CONFIG:Release>:-O3;-march=native>")
    add_compile_options("$<$<CONFIG:Debug>:-g>")
endif()

# --- 2. Find Dependencies ---
find_package(Eigen3 REQUIRED)

# --- 2a. OpenMP ---
# FindOpenMP handles flag detection per-compiler automatically.
# On MSVC it injects /openmp; we optionally replace that with /openmp:llvm below.
find_package(OpenMP REQUIRED)

if(MSVC)
    if(MSVC_VERSION LESS 1929)
        # VS 2019 < 16.9 doesn't have /openmp:llvm
        if(USE_OPENMP_LLVM)
            message(WARNING
                "USE_OPENMP_LLVM requires VS 2019 16.9+ (MSVC_VERSION >= 1929). "
                "Falling back to /openmp (OpenMP 2.0).")
            set(USE_OPENMP_LLVM OFF CACHE BOOL "" FORCE)
        endif()
    endif()

    if(USE_OPENMP_LLVM)
        # Strip whatever flag CMake chose and replace with /openmp:llvm.
        # /openmp:llvm uses libomp.dll (LLVM runtime) instead of vcomp140.dll
        # and supports OpenMP 3.1 (tasks, unsigned loop indices, etc.).
        # Use /openmp:experimental instead if you need OpenMP 5.x SIMD clauses
        # or GPU offloading.
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
    # Preferred: OpenBLAS via  vcpkg install openblas
    #                      OR  conda install openblas
    # Fallback:  any BLAS CMake can find (e.g. Intel MKL via oneAPI)
    # Last resort: Eigen's built-in slow path — configure succeeds with a warning
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

# --- 3. Core Static Library ---
set(SB_CORE_SOURCES
    src/kmeans.cpp
    src/affinity.cpp
    src/spectral.cpp
)

add_library(sb_core STATIC ${SB_CORE_SOURCES})

# PIC is a no-op on MSVC but required on Linux/macOS when a static lib is
# linked into a shared pybind11 extension (.so / .pyd).
set_property(TARGET sb_core PROPERTY POSITION_INDEPENDENT_CODE ON)

# target_include_directories (not global include_directories) so MSVC's
# stricter include resolution works correctly for all consumers.
target_include_directories(sb_core PUBLIC include)

target_link_libraries(sb_core
    PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX   # consumers (executables, pybind module) inherit this
    ${BLAS_LIBS}
)

# --- 4. Python Extension Module ---
if(PYMODULE)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(specbridge python/bindings.cpp)
    target_link_libraries(specbridge PRIVATE sb_core)
    # OpenMP and BLAS are inherited transitively via sb_core PUBLIC linkage.

    # Place the .pyd / .so next to the sources for easy in-tree importing.
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

# --- 6. Windows: copy runtime DLLs next to each executable ---
# Covers libomp.dll (/openmp:llvm), vcomp140.dll (/openmp), OpenBLAS.dll, etc.
# $<TARGET_RUNTIME_DLLS:...> requires CMake >= 3.21.
# --- 6. Windows: copy runtime DLLs next to each executable ---
if(WIN32)
    foreach(tgt spectral_bridges test_kmeans test_reference)
        add_custom_command(TARGET ${tgt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:${tgt}>>,copy_if_different,true>
                $<TARGET_RUNTIME_DLLS:${tgt}>
                $<TARGET_FILE_DIR:${tgt}>
            COMMAND_EXPAND_LISTS
        )
    endforeach()
endif()