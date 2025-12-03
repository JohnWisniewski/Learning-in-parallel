#!/bin/bash -l

#SBATCH --job-name=nnp_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=nnp_train-%j.out
#SBATCH --error=nnp_train-%j.err
#SBATCH --partition=gpu-v100
#SBATCH --time=00:10:00

echo "=========================================="
echo "Neural Network Training on Darwin"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Working Directory: $(pwd)"
echo ""

# Set up environment
echo "=== Setting up environment ==="
workgroup -g ea-cisc372-silber
vpkg_require gcc
vpkg_require cuda
echo ""

# Navigate to script directory (pre-parallel)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=== Current directory: $(pwd) ==="
echo ""

# Clean and compile
echo "=== Compiling ==="
make clean
make

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

echo ""
echo "=== Compilation successful ==="
echo ""

# Run training
echo "=== Starting Training ==="
echo "Time: $(date)"
echo ""

srun ./nnp train

TRAIN_EXIT=$?

echo ""
echo "=== Training Complete ==="
echo "Exit code: $TRAIN_EXIT"
echo "Time: $(date)"
echo ""

# Check if model was created
if [ -f "model.bin" ]; then
    echo "✓ Model file 'model.bin' created successfully"
    ls -lh model.bin
else
    echo "✗ Warning: Model file 'model.bin' not found"
fi

echo ""
echo "=========================================="
echo "Job completed"
echo "=========================================="

exit $TRAIN_EXIT

