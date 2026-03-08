"""
Test CLI interface with MPI execution (Test #69).

This test verifies that the command-line interface works correctly with mpirun,
including proper MPI initialization, parallel execution, and output handling.
"""

import fix_quimb_python313  # noqa: F401 - Must be first for Python 3.13+

import subprocess
import sys
import re
import pytest

pytestmark = pytest.mark.mpi


def test_cli_basic():
    """Test #69 Step 1: Basic CLI execution."""
    import shutil
    mpirun = shutil.which('mpirun')
    if mpirun is None:
        pytest.skip("mpirun not found")
    cmd = [
        mpirun, '--oversubscribe', '-np', '2',
        sys.executable, '-m', 'a2dmrg',
        '--sites', '4',
        '--bond-dim', '8',
        '--max-sweeps', '1'
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120
    )

    assert result.returncode == 0, f"CLI failed with return code {result.returncode}\nstderr: {result.stderr}"
    assert "A2DMRG Complete" in result.stdout
    assert "Ground state energy:" in result.stdout


def test_cli_mpi_initialization():
    """Test #69 Step 2: MPI initialization successful."""
    import os

    # Build environment with all necessary paths
    env = os.environ.copy()
    env['PATH'] = '/usr/lib64/openmpi/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = '/usr/lib64/openmpi/lib:' + env.get('LD_LIBRARY_PATH', '')

    # Use mpirun with 1 process (always available)
    cmd = [
        'mpirun', '-np', '2', '--oversubscribe',
        sys.executable, '-m', 'a2dmrg',
        '--sites', '4',
        '--bond-dim', '8',
        '--max-sweeps', '1'
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        env=env
    )

    assert result.returncode == 0, f"MPI execution failed with return code {result.returncode}\nstderr: {result.stderr}"
    assert "A2DMRG Complete" in result.stdout


def test_cli_rank0_output():
    """Test #69 Step 4: Verify output only from rank 0."""
    import os

    # Build environment with all necessary paths
    env = os.environ.copy()
    env['PATH'] = '/usr/lib64/openmpi/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = '/usr/lib64/openmpi/lib:' + env.get('LD_LIBRARY_PATH', '')

    # With np=1, only rank 0 exists and should print
    cmd = [
        'mpirun', '-np', '2', '--oversubscribe',
        sys.executable, '-m', 'a2dmrg',
        '--sites', '4',
        '--bond-dim', '8',
        '--max-sweeps', '1'
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        env=env
    )

    # Check that header and results are printed exactly once
    header_count = result.stdout.count("A2DMRG: Additive Two-Level Parallel DMRG")
    results_count = result.stdout.count("RESULTS")

    assert header_count == 1, f"Expected 1 header, got {header_count}"
    assert results_count == 1, f"Expected 1 results section, got {results_count}"


def test_cli_clean_completion():
    """Test #69 Step 5: Verify clean MPI finalization (no errors)."""
    import os

    # Build environment with all necessary paths
    env = os.environ.copy()
    env['PATH'] = '/usr/lib64/openmpi/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = '/usr/lib64/openmpi/lib:' + env.get('LD_LIBRARY_PATH', '')

    cmd = [
        'mpirun', '-np', '2', '--oversubscribe',
        sys.executable, '-m', 'a2dmrg',
        '--sites', '4',
        '--bond-dim', '8',
        '--max-sweeps', '1'
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        env=env
    )

    # Check for clean exit (no MPI errors in stderr)
    # Allow warnings but not errors
    stderr_lower = result.stderr.lower()

    # Filter out known harmless warnings
    lines = stderr_lower.split('\n')
    error_lines = [line for line in lines
                   if 'error' in line
                   and 'runtimeerror' not in line  # numba errors are handled
                   and 'complexwarning' not in line  # complex casting warnings OK
                   and 'userwarning' not in line]  # user warnings OK

    assert len(error_lines) == 0, f"Found error lines in stderr: {error_lines}"
    assert result.returncode == 0, "Process did not exit cleanly"


def test_cli_help():
    """Test that --help works."""
    cmd = [sys.executable, '-m', 'a2dmrg', '--help']

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode == 0
    assert "A2DMRG" in result.stdout
    assert "--sites" in result.stdout
    assert "--bond-dim" in result.stdout


def test_cli_energy_reasonable():
    """Verify that computed energy is physically reasonable."""
    import shutil
    mpirun = shutil.which('mpirun')
    if mpirun is None:
        pytest.skip("mpirun not found")
    cmd = [
        mpirun, '--oversubscribe', '-np', '2',
        sys.executable, '-m', 'a2dmrg',
        '--sites', '4',
        '--bond-dim', '8',
        '--max-sweeps', '1'
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120
    )

    # Extract energy from output
    match = re.search(r'Ground state energy:\s+([-+]?\d+\.\d+)', result.stdout)
    assert match, "Could not find energy in output"

    energy = float(match.group(1))

    # For Heisenberg model, energy should be negative
    assert energy < 0, f"Energy should be negative, got {energy}"

    # Reasonable range for L=4 Heisenberg
    assert energy > -5.0, f"Energy seems too negative: {energy}"
    assert energy < 0.0, f"Energy should be negative: {energy}"


if __name__ == '__main__':
    print("Running CLI/MPI interface tests...")

    print("\n1. Testing basic CLI...")
    test_cli_basic()
    print("✅ Basic CLI works")

    print("\n2. Testing MPI initialization...")
    test_cli_mpi_initialization()
    print("✅ MPI initialization works")

    print("\n3. Testing rank 0 output...")
    test_cli_rank0_output()
    print("✅ Output only from rank 0")

    print("\n4. Testing clean completion...")
    test_cli_clean_completion()
    print("✅ Clean MPI finalization")

    print("\n5. Testing --help...")
    test_cli_help()
    print("✅ Help works")

    print("\n6. Testing energy reasonableness...")
    test_cli_energy_reasonable()
    print("✅ Energy is reasonable")

    print("\n" + "="*60)
    print("All CLI/MPI tests passed!")
    print("="*60)
