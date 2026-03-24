"""
Hardware detection and run configuration.

Detects available CPU cores and generates valid (np, threads) combinations
ensuring np * threads_per_rank <= core_count.
"""

import os
from dataclasses import dataclass, asdict


@dataclass
class RunConfig:
    """Configuration for a single benchmark run."""
    np: int
    threads_per_rank: int
    total_threads: int
    allowed: bool
    reason: str = ""


def detect_core_count():
    """Detect available physical CPU cores."""
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        if physical and physical > 0:
            return physical
    except ImportError:
        pass
    logical = os.cpu_count()
    return logical if logical and logical > 0 else 4


def get_repo_root():
    """Get the repository root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_venv_python(package_name):
    """Get path to venv Python for a given implementation package.

    Args:
        package_name: e.g. 'pdmrg', 'pdmrg-opt', 'a2dmrg', 'pdmrg-cotengra'
    """
    root = get_repo_root()
    venv_path = os.path.join(root, package_name, "venv", "bin", "python")
    if not os.path.exists(venv_path):
        raise FileNotFoundError(
            f"Virtual environment not found for {package_name}: {venv_path}\n"
            f"Create it with: cd {os.path.join(root, package_name)} && python -m venv venv"
        )
    return venv_path


def get_mpi_env(threads=1):
    """Get environment dict for MPI subprocess execution.

    Sets BLAS thread counts and adds MPI paths.
    """
    env = os.environ.copy()

    # Set BLAS threading
    thread_str = str(threads)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = thread_str

    # Add MPI paths
    for bin_path in ("/usr/lib64/openmpi/bin", "/usr/local/bin", "/opt/openmpi/bin"):
        if os.path.exists(bin_path):
            env["PATH"] = f"{bin_path}:{env.get('PATH', '')}"
            break

    for lib_path in ("/usr/lib64/openmpi/lib", "/usr/local/lib", "/opt/openmpi/lib"):
        if os.path.exists(lib_path):
            env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}"
            break

    return env


def get_blas_env(threads=1):
    """Get environment dict with BLAS thread count set."""
    env = os.environ.copy()
    thread_str = str(threads)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = thread_str
    return env


def generate_configs(np_values, thread_values, core_count=None):
    """Generate valid (np, threads) run configurations.

    Returns list of RunConfig, marking invalid ones with allowed=False.
    """
    if core_count is None:
        core_count = detect_core_count()

    configs = []
    for np_val in np_values:
        for threads in thread_values:
            total = np_val * threads
            allowed = total <= core_count
            reason = "" if allowed else (
                f"np={np_val} x threads={threads} = {total} > {core_count} cores"
            )
            configs.append(RunConfig(
                np=np_val, threads_per_rank=threads,
                total_threads=total, allowed=allowed, reason=reason,
            ))
    return configs
