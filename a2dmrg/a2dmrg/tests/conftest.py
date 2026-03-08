"""
Pytest configuration for a2dmrg tests.

This file sets up the testing environment, including MPI library paths
and Python 3.13+ compatibility for quimb/numba.
"""

# CRITICAL: Apply numba fix BEFORE any imports that use quimb
import sys
import os
import pytest

# Add project root to path so we can import fix_quimb_python313
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Apply the numba fix for Python 3.13+
try:
    import fix_quimb_python313
except Exception:
    pass  # If fix not needed or not available, continue


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked with @pytest.mark.mpi when running with np < 2."""
    try:
        from mpi4py import MPI
        size = MPI.COMM_WORLD.size
    except Exception:
        size = 1

    if size < 2:
        skip_mpi = pytest.mark.skip(
            reason="A2DMRG requires np>=2; run with: mpirun -np 2 python -m pytest -m mpi"
        )
        for item in items:
            if "mpi" in item.keywords:
                item.add_marker(skip_mpi)


def pytest_configure(config):
    """
    Set up environment variables before running tests.

    This is particularly important for MPI, which needs LD_LIBRARY_PATH
    to be set correctly.
    """
    # Add OpenMPI to library path
    openmpi_lib = "/usr/lib64/openmpi/lib"
    if os.path.exists(openmpi_lib):
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if openmpi_lib not in current_ld_path:
            if current_ld_path:
                os.environ["LD_LIBRARY_PATH"] = f"{openmpi_lib}:{current_ld_path}"
            else:
                os.environ["LD_LIBRARY_PATH"] = openmpi_lib

    # Add OpenMPI bin to PATH
    openmpi_bin = "/usr/lib64/openmpi/bin"
    if os.path.exists(openmpi_bin):
        current_path = os.environ.get("PATH", "")
        if openmpi_bin not in current_path:
            os.environ["PATH"] = f"{openmpi_bin}:{current_path}"
