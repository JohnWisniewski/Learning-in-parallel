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
# Try multiple methods to find the correct directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
    echo "=== Changed to SLURM submit directory: $(pwd) ==="
elif [ -n "$HOME" ] && [ -d "$HOME/Learning-in-parallel/pre-parallel" ]; then
    # Use absolute path based on home directory
    cd "$HOME/Learning-in-parallel/pre-parallel"
    echo "=== Changed to home-based path: $(pwd) ==="
else
    # Fallback: try to find the script's directory
    # In SLURM, BASH_SOURCE might point to spool dir, so try to get original location
    if [ -f "${BASH_SOURCE[0]}" ]; then
        SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
        # If we're in spool directory, try to find pre-parallel in common locations
        if [[ "$SCRIPT_DIR" == *"/spool/slurm"* ]]; then
            if [ -d "/home/4255/Learning-in-parallel/pre-parallel" ]; then
                cd "/home/4255/Learning-in-parallel/pre-parallel"
            elif [ -d "$HOME/Learning-in-parallel/pre-parallel" ]; then
                cd "$HOME/Learning-in-parallel/pre-parallel"
            else
                cd "$SCRIPT_DIR"
            fi
        else
            cd "$SCRIPT_DIR"
        fi
        echo "=== Changed to script directory: $(pwd) ==="
    else
        echo "ERROR: Cannot determine correct directory!"
        exit 1
    fi
fi

# Verify we're in the right place
if [ ! -f "makefile" ]; then
    echo "ERROR: makefile not found in $(pwd)"
    echo "Contents of current directory:"
    ls -la
    exit 1
fi

echo "=== Current directory: $(pwd) ==="
echo "=== Verified: makefile found ==="
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

