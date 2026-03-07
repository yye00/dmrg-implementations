#!/usr/bin/env python3
"""
Benchmark Utilities

Provides path detection and metadata helpers for DMRG benchmark scripts.
Eliminates hard-coded absolute paths and ensures semantic honesty in benchmark reporting.
"""

import os
import sys
import socket
import platform
from datetime import datetime
from typing import Dict, Optional, Any


def get_repo_root() -> str:
    """
    Get the repository root directory.

    Returns:
        Absolute path to the repository root
    """
    # This file is in benchmarks/, so parent is repo root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_implementation_paths() -> Dict[str, str]:
    """
    Get paths to all implementation directories.

    Returns:
        Dictionary with keys: 'pdmrg', 'pdmrg2', 'a2dmrg', 'pdmrg_gpu'
        Values are absolute paths to each implementation directory
    """
    root = get_repo_root()
    return {
        'pdmrg': os.path.join(root, 'pdmrg'),
        'pdmrg2': os.path.join(root, 'pdmrg2'),
        'a2dmrg': os.path.join(root, 'a2dmrg'),
        'pdmrg_gpu': os.path.join(root, 'pdmrg-gpu'),
        'benchmarks': os.path.join(root, 'benchmarks'),
    }


def get_venv_python(implementation: str) -> str:
    """
    Get path to the Python interpreter in an implementation's virtual environment.

    Args:
        implementation: One of 'pdmrg', 'pdmrg2', 'a2dmrg'

    Returns:
        Absolute path to the venv Python interpreter

    Raises:
        FileNotFoundError: If the venv doesn't exist
    """
    paths = get_implementation_paths()
    if implementation not in paths:
        raise ValueError(f"Unknown implementation: {implementation}")

    venv_path = os.path.join(paths[implementation], 'venv', 'bin', 'python')

    if not os.path.exists(venv_path):
        raise FileNotFoundError(
            f"Virtual environment not found for {implementation}: {venv_path}\n"
            f"Please create it with: cd {paths[implementation]} && python -m venv venv"
        )

    return venv_path


def get_mpi_env() -> Dict[str, str]:
    """
    Get MPI environment variables for subprocess execution.

    Returns:
        Dictionary of environment variables with MPI paths added
    """
    env = os.environ.copy()

    # Try common MPI installation paths
    mpi_paths = [
        '/usr/lib64/openmpi/bin',
        '/usr/local/bin',
        '/opt/openmpi/bin',
    ]

    mpi_lib_paths = [
        '/usr/lib64/openmpi/lib',
        '/usr/local/lib',
        '/opt/openmpi/lib',
    ]

    # Find first existing MPI path
    mpi_bin = None
    for path in mpi_paths:
        if os.path.exists(path):
            mpi_bin = path
            break

    mpi_lib = None
    for path in mpi_lib_paths:
        if os.path.exists(path):
            mpi_lib = path
            break

    if mpi_bin:
        env['PATH'] = f"{mpi_bin}:{env.get('PATH', '')}"

    if mpi_lib:
        env['LD_LIBRARY_PATH'] = f"{mpi_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    return env


def create_benchmark_metadata(
    method: str,
    energy: float,
    time: float,
    success: bool,
    algorithm_executed: str,
    early_return: bool,
    warmup_used: bool,
    warmup_sweeps: int,
    np_count: int,
    system_size: int,
    bond_dim: int,
    max_sweeps: int,
    tolerance: float,
    dtype: str = "float64",
    early_return_reason: Optional[str] = None,
    warmup_method: Optional[str] = None,
    skip_opt: Optional[bool] = None,
    canonicalization_enabled: Optional[bool] = None,
    random_init: Optional[bool] = None,
    converged: Optional[bool] = None,
    final_sweep: Optional[int] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a benchmark result dictionary with full metadata schema.

    This implements the metadata schema defined in BENCHMARK_METADATA_SCHEMA.md
    to ensure semantic honesty in benchmark reporting.

    Args:
        method: Method name, e.g., "PDMRG_np4", "A2DMRG_np2"
        energy: Ground state energy
        time: Wall-clock time in seconds
        success: Whether the run succeeded
        algorithm_executed: What actually ran (e.g., "PDMRG parallel sweeps")
        early_return: Did the function return early?
        warmup_used: Was warmup initialization used?
        warmup_sweeps: Number of warmup sweeps
        np_count: Number of MPI processes
        system_size: L (chain length)
        bond_dim: Bond dimension
        max_sweeps: Maximum allowed sweeps
        tolerance: Solver convergence tolerance
        dtype: Data type (e.g., "float64", "complex128")
        early_return_reason: Why early return happened (if applicable)
        warmup_method: How warmup was performed (if applicable)
        skip_opt: PDMRG boundary merge optimization status
        canonicalization_enabled: A2DMRG canonicalization status
        random_init: Was random initialization used?
        converged: Did the solver converge?
        final_sweep: Final sweep number
        error: Error message if failed

    Returns:
        Dictionary with full metadata schema
    """
    try:
        import numpy as np
        numpy_version = np.__version__
    except ImportError:
        numpy_version = "unknown"

    try:
        import quimb
        quimb_version = quimb.__version__
    except ImportError:
        quimb_version = None

    result = {
        "method": method,
        "energy": energy if success else None,
        "time": time if success else None,
        "success": success,
        "metadata": {
            # Execution path
            "algorithm_executed": algorithm_executed,
            "early_return": early_return,
            "early_return_reason": early_return_reason,

            # Warmup
            "warmup_used": warmup_used,
            "warmup_sweeps": warmup_sweeps,
            "warmup_method": warmup_method,

            # Algorithm-specific flags
            "skip_opt": skip_opt,
            "canonicalization_enabled": canonicalization_enabled,
            "random_init": random_init,

            # MPI
            "np": np_count,
            "mpi_used": np_count > 1 or method.startswith("PDMRG") or method.startswith("A2DMRG"),

            # Convergence
            "converged": converged,
            "final_sweep": final_sweep,
            "max_sweeps": max_sweeps,

            # System configuration
            "system_size": system_size,
            "bond_dim": bond_dim,
            "dtype": dtype,
            "tolerance": tolerance,

            # Environment
            "hostname": socket.gethostname(),
            "timestamp": datetime.now().isoformat(),
            "python_version": platform.python_version(),
            "numpy_version": numpy_version,
            "quimb_version": quimb_version,
        }
    }

    if error and not success:
        result["error"] = error

    return result


def get_default_output_path(benchmark_name: str) -> str:
    """
    Get default output path for benchmark results.

    Args:
        benchmark_name: Name of the benchmark (e.g., "heisenberg_benchmark")

    Returns:
        Absolute path to results JSON file in benchmarks/ directory
    """
    paths = get_implementation_paths()
    return os.path.join(paths['benchmarks'], f"{benchmark_name}_results.json")


# Convenience function for backward compatibility
def get_pdmrg_path() -> str:
    """Get PDMRG implementation directory."""
    return get_implementation_paths()['pdmrg']


def get_a2dmrg_path() -> str:
    """Get A2DMRG implementation directory."""
    return get_implementation_paths()['a2dmrg']


def get_pdmrg2_path() -> str:
    """Get PDMRG2 implementation directory (if it exists)."""
    return get_implementation_paths()['pdmrg2']


if __name__ == '__main__':
    # Test path detection
    print("Repository Root:", get_repo_root())
    print("\nImplementation Paths:")
    for name, path in get_implementation_paths().items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {name:15} {exists} {path}")

    print("\nVirtual Environments:")
    for impl in ['pdmrg', 'a2dmrg']:
        try:
            venv = get_venv_python(impl)
            exists = "✓" if os.path.exists(venv) else "✗"
            print(f"  {impl:15} {exists} {venv}")
        except FileNotFoundError as e:
            print(f"  {impl:15} ✗ {e}")

    print("\nMPI Environment:")
    mpi_env = get_mpi_env()
    print(f"  PATH: {mpi_env.get('PATH', 'N/A')[:100]}...")
    print(f"  LD_LIBRARY_PATH: {mpi_env.get('LD_LIBRARY_PATH', 'N/A')[:100]}...")

    print("\nExample Metadata:")
    meta = create_benchmark_metadata(
        method="PDMRG_np1",
        energy=-5.373916515211431,
        time=0.42,
        success=True,
        algorithm_executed="quimb DMRG2 warmup (early return)",
        early_return=True,
        early_return_reason="np=1 with warmup enabled",
        warmup_used=True,
        warmup_sweeps=5,
        warmup_method="quimb DMRG2 serial",
        np_count=1,
        system_size=12,
        bond_dim=20,
        max_sweeps=30,
        tolerance=1e-10,
        skip_opt=None
    )
    import json
    print(json.dumps(meta, indent=2))
