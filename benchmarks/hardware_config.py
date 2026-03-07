"""
Hardware-aware benchmark configuration.

Detects available CPU cores and generates valid thread/rank combinations
for DMRG benchmarks. Ensures that np × threads_per_rank ≤ detected_core_count.
"""

import os
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict


@dataclass
class RunConfig:
    """Configuration for a single benchmark run."""
    np: int
    threads_per_rank: int
    total_threads: int
    allowed: bool
    reason: str = ""


@dataclass
class HardwareConfig:
    """Hardware configuration and constraints."""
    core_count: int
    preferred_thread_counts: List[int]
    preferred_np_values: List[int]

    def __post_init__(self):
        """Validate configuration."""
        if self.core_count <= 0:
            raise ValueError(f"Invalid core count: {self.core_count}")


def detect_core_count() -> int:
    """
    Detect available CPU cores on the system.

    Returns physical core count if available, otherwise logical core count.
    """
    try:
        # Try to get physical core count
        import psutil
        physical = psutil.cpu_count(logical=False)
        if physical and physical > 0:
            return physical
    except ImportError:
        pass

    # Fallback to os.cpu_count() (logical cores)
    logical = os.cpu_count()
    if logical and logical > 0:
        return logical

    # Last resort default
    return 4


def generate_serial_configs(hw_config: HardwareConfig) -> List[RunConfig]:
    """
    Generate thread configurations for serial (non-MPI) methods.

    For quimb DMRG1/DMRG2: threads in [1, 2, 4, 8] up to core_count.
    """
    configs = []
    for threads in hw_config.preferred_thread_counts:
        if threads <= hw_config.core_count:
            configs.append(RunConfig(
                np=1,
                threads_per_rank=threads,
                total_threads=threads,
                allowed=True
            ))
        else:
            configs.append(RunConfig(
                np=1,
                threads_per_rank=threads,
                total_threads=threads,
                allowed=False,
                reason=f"threads={threads} exceeds core_count={hw_config.core_count}"
            ))
    return configs


def generate_mpi_configs(hw_config: HardwareConfig) -> List[RunConfig]:
    """
    Generate (np, threads_per_rank) configurations for MPI methods.

    For PDMRG/PDMRG2/A2DMRG: ensure np × threads_per_rank ≤ core_count.
    """
    configs = []
    for np in hw_config.preferred_np_values:
        for threads in hw_config.preferred_thread_counts:
            total = np * threads
            allowed = total <= hw_config.core_count

            reason = ""
            if not allowed:
                reason = f"np={np} × threads={threads} = {total} exceeds core_count={hw_config.core_count}"

            configs.append(RunConfig(
                np=np,
                threads_per_rank=threads,
                total_threads=total,
                allowed=allowed,
                reason=reason
            ))

    return configs


def get_default_hardware_config() -> HardwareConfig:
    """Get default hardware configuration for current system."""
    core_count = detect_core_count()
    return HardwareConfig(
        core_count=core_count,
        preferred_thread_counts=[1, 2, 4, 8],
        preferred_np_values=[1, 2, 4, 8]
    )


def generate_run_matrix(hw_config: HardwareConfig = None) -> Dict:
    """
    Generate complete run matrix for benchmarks.

    Returns:
        Dictionary with:
        - hardware: detected hardware config
        - serial: configs for quimb DMRG1/DMRG2
        - mpi: configs for PDMRG/PDMRG2/A2DMRG
    """
    if hw_config is None:
        hw_config = get_default_hardware_config()

    serial_configs = generate_serial_configs(hw_config)
    mpi_configs = generate_mpi_configs(hw_config)

    return {
        "hardware": asdict(hw_config),
        "serial": [asdict(c) for c in serial_configs],
        "mpi": [asdict(c) for c in mpi_configs]
    }


def print_run_matrix(run_matrix: Dict = None):
    """Print human-readable run matrix."""
    if run_matrix is None:
        run_matrix = generate_run_matrix()

    hw = run_matrix["hardware"]
    print("=" * 80)
    print("HARDWARE-AWARE RUN MATRIX")
    print("=" * 80)
    print(f"Detected cores: {hw['core_count']}")
    print(f"Preferred thread counts: {hw['preferred_thread_counts']}")
    print(f"Preferred np values: {hw['preferred_np_values']}")
    print()

    print("SERIAL METHODS (quimb DMRG1/DMRG2):")
    print("-" * 80)
    print(f"{'Threads':<10} {'Total':<10} {'Allowed':<10} {'Reason':<40}")
    print("-" * 80)
    for cfg in run_matrix["serial"]:
        status = "✓" if cfg["allowed"] else "✗"
        print(f"{cfg['threads_per_rank']:<10} {cfg['total_threads']:<10} {status:<10} {cfg['reason']:<40}")
    print()

    print("MPI METHODS (PDMRG/PDMRG2/A2DMRG):")
    print("-" * 80)
    print(f"{'np':<6} {'Threads':<10} {'Total':<10} {'Allowed':<10} {'Reason':<40}")
    print("-" * 80)
    for cfg in run_matrix["mpi"]:
        status = "✓" if cfg["allowed"] else "✗"
        print(f"{cfg['np']:<6} {cfg['threads_per_rank']:<10} {cfg['total_threads']:<10} {status:<10} {cfg['reason']:<40}")
    print()


def filter_allowed_configs(configs: List[Dict]) -> List[Dict]:
    """Filter to only allowed configurations."""
    return [c for c in configs if c["allowed"]]


if __name__ == "__main__":
    # Demo: print run matrix for current system
    run_matrix = generate_run_matrix()
    print_run_matrix(run_matrix)

    # Save to file
    output_path = "run_matrix.json"
    with open(output_path, 'w') as f:
        json.dump(run_matrix, f, indent=2)
    print(f"Run matrix saved to: {output_path}")
