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
echo "Initial Working Directory: $(pwd)"
echo ""

# Navigate to the directory where sbatch was executed (should be pre-parallel)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
    echo "=== Changed to SLURM submit directory: $(pwd) ==="
else
    # Fallback: try to find the script's directory
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd "$SCRIPT_DIR"
    echo "=== Changed to script directory: $(pwd) ==="
fi

echo "=== Current directory: $(pwd) ==="
echo ""

# Set up environment
echo "=== Setting up environment ==="
# workgroup might already be set or not needed in batch jobs
if command -v workgroup &> /dev/null; then
    workgroup -g ea-cisc372-silber
fi
vpkg_require gcc
vpkg_require cuda
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

