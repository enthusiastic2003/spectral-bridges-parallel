#!/bin/bash
#SBATCH --job-name=specbridge_build_run
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --output=logs_cpu/specbridge_output.log
#SBATCH --error=logs_cpu/specbridge_error.log

# Environment setup

export PATH=/usr/local/lmod:$PATH

echo "Node list: $SLURM_NODELIST"
echo "Current node: $SLURMD_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# Activate virtual environment

source /home/sirjanhansda/projectfolder/.venv/bin/activate

# Paths

PROJECT_DIR=/home/sirjanhansda/projectfolder/spectral-bridges-parallel
EIGEN_DIR=/home/sirjanhansda/projectfolder/eigen-3.4.0
OPENBLAS_LIB=/home/sirjanhansda/projectfolder/openblas/lib/libopenblas.so
OPENBLAS_DIR=/home/sirjanhansda/projectfolder/openblas

# Runtime linking

export LD_LIBRARY_PATH=$OPENBLAS_DIR/lib:$LD_LIBRARY_PATH

# Threads

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Go to project

cd $PROJECT_DIR

# Clean build

rm -rf build
mkdir -p build
cd build

# Configure with CMake

cmake .. \
    -DPYMODULE=ON \
    -DCMAKE_MODULE_PATH=$EIGEN_DIR/cmake \
    -DEIGEN3_INCLUDE_DIR=$EIGEN_DIR \
    -DBLAS_LIBRARIES=$OPENBLAS_LIB \
    -DLAPACKE_INCLUDE_DIR=$OPENBLAS_DIR/include \
    -DSPECTRA_INCLUDE_DIR=/home/sirjanhansda/projectfolder/spectra/include \
    -DUSE_CUDA=ON \
    -Dpybind11_DIR=$(python -m pybind11 --cmakedir)

# Build

cmake --build .

cd ..

# Run (adjust executable if needed)

echo "Running module..."

python -u ./test_suit_speedups_cpu.py --experiment n
# python -u ./mnist_test.py