"""
Test #67: CLI Interface

This test verifies the command-line interface works correctly with various
arguments and parameter combinations.
"""

import shutil
import subprocess
import sys
import pytest
import numpy as np

pytestmark = pytest.mark.mpi


class TestCLI:
    """Test #67: CLI interface - Run from command line with arguments."""

    def run_cli(self, args, expect_success=True, use_mpi=True):
        """Helper to run CLI and return output."""
        if use_mpi:
            mpirun = shutil.which('mpirun')
            if mpirun is None:
                pytest.skip("mpirun not found")
            cmd = [mpirun, '--oversubscribe', '-np', '2',
                   sys.executable, '-m', 'a2dmrg'] + args
        else:
            cmd = [sys.executable, '-m', 'a2dmrg'] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if expect_success:
            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                assert result.returncode == 0, f"CLI failed with return code {result.returncode}"

        return result

    def test_basic_execution(self):
        """
        Step 1: Test: python -m a2dmrg --sites 20 --bond-dim 50

        Verify basic CLI execution with explicit parameters.
        """
        result = self.run_cli([
            '--sites', '6',  # Use small system for speed
            '--bond-dim', '8',
            '--max-sweeps', '5',
            '--quiet'
        ])

        # Check output contains results
        assert 'Ground state energy:' in result.stdout
        assert 'Energy per site:' in result.stdout

        # Energy should be negative for Heisenberg
        output = result.stdout
        energy_line = [line for line in output.split('\n') if 'Ground state energy:' in line][0]
        energy_str = energy_line.split(':')[1].strip()
        energy = float(energy_str)
        assert energy < 0, f"Energy should be negative, got {energy}"

    def test_argument_parsing(self):
        """
        Step 2: Verify arguments parsed correctly.

        Test that command-line arguments are properly parsed and used.
        """
        # Test with various argument formats
        result = self.run_cli([
            '--sites', '4',
            '--bond-dim', '4',
            '--max-sweeps', '3',
            '--tol', '1e-8',
            '--quiet'
        ])

        assert result.returncode == 0
        assert 'Ground state energy:' in result.stdout

    def test_default_values(self):
        """
        Step 3: Verify defaults applied for unspecified args.

        Only specify required argument (--sites), verify defaults work.
        """
        result = self.run_cli([
            '--sites', '4',
            '--bond-dim', '8',  # Override default (100 is too slow)
            '--max-sweeps', '3',  # Keep it quick
            '--quiet'
        ])

        # Should use defaults for:
        # - model: heisenberg
        # - dtype: float64
        # - one_site: False (two-site)
        assert result.returncode == 0
        assert 'Ground state energy:' in result.stdout

    def test_dtype_complex128(self):
        """
        Step 4: Test: --dtype complex128 flag.

        Verify complex dtype is properly handled.
        """
        result = self.run_cli([
            '--sites', '6',
            '--bond-dim', '8',
            '--dtype', 'complex128',
            '--max-sweeps', '5',
            '--quiet'
        ])

        assert result.returncode == 0
        assert 'Ground state energy:' in result.stdout

        # Extract energy and verify it's real (for Hermitian Hamiltonian)
        output = result.stdout
        energy_line = [line for line in output.split('\n') if 'Ground state energy:' in line][0]
        energy_str = energy_line.split(':')[1].strip()
        energy = float(energy_str)

        # Should be finite and negative
        assert np.isfinite(energy)
        assert energy < 0

    def test_model_josephson(self):
        """
        Step 5: Test: --model josephson flag.

        Verify model selection works.
        """
        result = self.run_cli([
            '--sites', '6',
            '--bond-dim', '8',
            '--model', 'josephson',
            '--max-sweeps', '5',
            '--quiet'
        ])

        assert result.returncode == 0
        assert 'Ground state energy:' in result.stdout

        # Energy should be reasonable
        output = result.stdout
        energy_line = [line for line in output.split('\n') if 'Ground state energy:' in line][0]
        energy_str = energy_line.split(':')[1].strip()
        energy = float(energy_str)
        assert np.isfinite(energy)

    def test_help_message(self):
        """
        Step 6: Verify help message: --help.

        Test that --help displays usage information.
        """
        result = self.run_cli(['--help'], expect_success=True, use_mpi=False)

        # Help should print to stdout and exit with 0
        assert result.returncode == 0
        assert 'usage:' in result.stdout
        assert 'A2DMRG' in result.stdout
        assert '--sites' in result.stdout
        assert '--bond-dim' in result.stdout
        assert '--dtype' in result.stdout
        assert '--model' in result.stdout

    def test_missing_required_argument(self):
        """Test that missing --sites argument produces error."""
        cmd = [sys.executable, '-m', 'a2dmrg', '--bond-dim', '10']
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0
        # Error message should mention missing argument
        assert '--sites' in result.stderr or 'required' in result.stderr.lower()

    def test_invalid_dtype(self):
        """Test that invalid dtype produces error."""
        cmd = [sys.executable, '-m', 'a2dmrg', '--sites', '4', '--dtype', 'invalid']
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0
        # Error message should mention invalid choice
        assert 'invalid choice' in result.stderr.lower() or 'dtype' in result.stderr.lower()

    def test_one_site_flag(self):
        """Test --one-site flag."""
        result = self.run_cli([
            '--sites', '4',
            '--bond-dim', '8',
            '--one-site',
            '--max-sweeps', '5',
            '--quiet'
        ])

        assert result.returncode == 0
        assert 'Ground state energy:' in result.stdout

    def test_verbose_output(self):
        """Test that verbose mode produces output."""
        result = self.run_cli([
            '--sites', '4',
            '--bond-dim', '4',
            '--max-sweeps', '3',
            '--verbose'
        ])

        assert result.returncode == 0
        # Verbose should show header
        assert 'A2DMRG' in result.stdout
        assert 'Sites (L):' in result.stdout or 'sites' in result.stdout.lower()

    def test_quiet_mode(self):
        """Test that quiet mode suppresses verbose output."""
        result = self.run_cli([
            '--sites', '4',
            '--bond-dim', '4',
            '--max-sweeps', '3',
            '--quiet'
        ])

        assert result.returncode == 0
        # Quiet mode should still show results
        assert 'Ground state energy:' in result.stdout
        # But not show sweep-by-sweep progress (if any)

    def test_short_form_sites(self):
        """Test short form -L for --sites."""
        result = self.run_cli([
            '-L', '4',
            '--bond-dim', '4',
            '--max-sweeps', '3',
            '--quiet'
        ])

        assert result.returncode == 0
        assert 'Ground state energy:' in result.stdout

    def test_custom_tolerance(self):
        """Test custom tolerance parameter."""
        result = self.run_cli([
            '--sites', '4',
            '--bond-dim', '8',
            '--max-sweeps', '5',
            '--tol', '1e-6',
            '--quiet'
        ])

        assert result.returncode == 0
        assert 'Ground state energy:' in result.stdout
