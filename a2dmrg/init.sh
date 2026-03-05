#!/bin/bash
# A2DMRG Environment Setup
# Initializes development environment and verifies dependencies

set -e  # Exit on error

echo "========================================="
echo "A2DMRG Environment Setup"
echo "========================================="

# Set OpenMPI paths
export PATH=/usr/lib64/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies if needed
if [ "$1" == "--install" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install numpy scipy quimb mpi4py pytest pytest-mpi

    # Install package in editable mode if setup files exist
    if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
        pip install -e .
    fi
fi

# Run sanity check
echo ""
echo "========================================="
echo "Running sanity checks..."
echo "========================================="

python3 -c "
import sys
import numpy as np
import scipy
import quimb
from a2dmrg.mpi_compat import MPI, HAS_MPI
print(f'✓ NumPy version: {np.__version__}')
print(f'✓ SciPy version: {scipy.__version__}')
print(f'✓ Quimb version: {quimb.__version__}')
print(f'✓ MPI4py version: {MPI.Get_version()}')
print('')
print('All dependencies OK!')
"

echo ""
echo "========================================="
echo "Environment ready!"
echo "========================================="
echo ""
echo "To activate the environment in future sessions:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest -v"
echo "  pytest -v a2dmrg/tests/"
echo ""
echo "To run with MPI:"
echo "  mpirun -np 2 pytest -v"
echo "  mpirun -np 4 python -m a2dmrg.dmrg --sites 40 --bond-dim 100"
echo ""
echo "To reinstall dependencies:"
echo "  ./init.sh --install"
echo ""
