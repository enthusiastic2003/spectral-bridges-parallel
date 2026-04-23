#!/bin/bash
#SBATCH --job-name=sb_cuda_test
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/cuda_test.out
#SBATCH --error=logs/cuda_test.err

# =============================================================
# Environment setup
# =============================================================
export PATH=/usr/local/lmod:$PATH

echo "======================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "GPU request:   $SLURM_JOB_GPUS"
echo "Start time:    $(date)"
echo "======================================================"
# GPU info — fail fast if no GPU is visible
echo ""
echo "--- GPU info ---"
nvidia-smi || { echo "ERROR: nvidia-smi failed — no GPU on this node"; exit 1; }
echo ""

# =============================================================
# Module / environment loading
# =============================================================
# Adjust these to match `module avail` output on your cluster.
# Common patterns: cuda/12.2, cuda/11.8, cmake/3.27, gcc/11.3
# If your cluster doesn't use modules, comment these out and ensure
# nvcc/cmake/gcc are on PATH by other means.

# module load cuda/12.2
# module load cmake/3.27
# module load gcc/11.3

# Print what we're actually using — useful when debugging toolchain issues
echo "--- Toolchain ---"
which nvcc && nvcc --version | tail -n 2
which cmake && cmake --version | head -n 1
which gcc && gcc --version | head -n 1
echo ""

# Activate Python venv (for pybind11 if PYMODULE=ON)
source /home/sirjanhansda/projectfolder/.venv/bin/activate

# =============================================================
# Paths
# =============================================================
PROJECT_DIR=/home/sirjanhansda/projectfolder/spectral-bridges-parallel
EIGEN_DIR=/home/sirjanhansda/projectfolder/eigen-3.4.0
OPENBLAS_DIR=/home/sirjanhansda/projectfolder/openblas
OPENBLAS_LIB=$OPENBLAS_DIR/lib/libopenblas.so

export LD_LIBRARY_PATH=$OPENBLAS_DIR/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# =============================================================
# CUDA architecture detection
# =============================================================
# Auto-detect compute capability from the first visible GPU.
# Falls back to sm_80 (A100) if detection fails.
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | head -n 1 | tr -d '.')
if [ -z "$CUDA_ARCH" ]; then
    echo "Could not auto-detect CUDA arch, defaulting to 80"
    CUDA_ARCH=80
fi
echo "CUDA architecture: sm_${CUDA_ARCH}"
echo ""

# =============================================================
# Build
# =============================================================
cd $PROJECT_DIR
rm -rf ./build
mkdir build
# Clean rebuild — remove this line if you want incremental builds during dev
cd build

echo "--- Configuring ---"
cmake .. \
    -DUSE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH \
    -DPYMODULE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MODULE_PATH=$EIGEN_DIR/cmake \
    -DEIGEN3_INCLUDE_DIR=$EIGEN_DIR \
    -DBLAS_LIBRARIES=$OPENBLAS_LIB \
    -Dpybind11_DIR=$(python -m pybind11 --cmakedir)

echo ""
echo "--- Building ---"
cmake --build . -j $SLURM_CPUS_PER_TASK --target test_kmeans_cuda specbridge \
    || { echo "Build failed"; exit 1; }

# =============================================================
# Run
# =============================================================
echo ""
echo "--- Running test_kmeans_cuda ---"
./test_kmeans_cuda
TEST_EXIT=$?

echo ""
echo "======================================================"
echo "End time:      $(date)"
echo "Exit code:     $TEST_EXIT"
echo "======================================================"

exit $TEST_EXIT