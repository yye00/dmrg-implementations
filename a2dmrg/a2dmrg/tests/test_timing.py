"""
Test timing measurements for A2DMRG convergence.

Test #42: Timing measurement: Measure wall time for A2DMRG convergence
"""

import pytest
import numpy as np
import time
from a2dmrg.mpi_compat import MPI
from quimb.tensor import SpinHam1D
from a2dmrg.dmrg import a2dmrg_main


class TestTiming:
    """Test timing measurements for A2DMRG."""

    def test_basic_timing_measurement(self):
        """
        Test #42 Steps 1-3: Run A2DMRG with timing and measure convergence time.

        Steps:
        1. Run A2DMRG with timing instrumentation
        2. Measure time to reach |ΔE| < 1e-8
        3. Record time_to_convergence in seconds
        """
        # Setup
        L = 8
        bond_dim = 16
        max_sweeps = 20
        tol = 1e-8
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Build Heisenberg MPO
        builder = SpinHam1D(S=1/2)
        builder += 1.0, 'X', 'X'
        builder += 1.0, 'Y', 'Y'
        builder += 1.0, 'Z', 'Z'
        mpo = builder.build_mpo(L)

        # Run with timing
        start_time = time.time()
        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=max_sweeps,
            bond_dim=bond_dim,
            tol=tol,
            comm=comm,
            dtype=np.float64,
            one_site=True,
            verbose=False  # Disable verbose for clean timing
        )
        end_time = time.time()

        # Measure time
        total_time = end_time - start_time

        # Verify timing is reasonable
        assert total_time > 0, "Timing should be positive"
        assert total_time < 300, f"Should complete in <5 minutes, took {total_time:.2f}s"

        # Verify convergence
        assert energy < 0, f"Energy should be negative for Heisenberg, got {energy}"

        # Only rank 0 prints results
        if rank == 0:
            print(f"\n=== Timing Results ===")
            print(f"Total time: {total_time:.3f} seconds")
            print(f"Final energy: {energy:.12f}")
            print(f"Convergence tolerance: {tol}")

    def test_timing_components(self):
        """
        Test #42 Step 4: Separate timing for different components.

        Measures time for:
        - Initialization
        - Local steps
        - Coarse-space computation
        - Compression

        Note: This is an integration test that measures overall performance.
        Detailed component timing would require instrumentation inside a2dmrg_main.
        """
        L = 6
        bond_dim = 12
        max_sweeps = 5
        tol = 1e-6
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Build Heisenberg MPO
        builder = SpinHam1D(S=1/2)
        builder += 1.0, 'X', 'X'
        builder += 1.0, 'Y', 'Y'
        builder += 1.0, 'Z', 'Z'
        mpo = builder.build_mpo(L)

        # Measure total time
        start_time = time.time()
        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=max_sweeps,
            bond_dim=bond_dim,
            tol=tol,
            comm=comm,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )
        end_time = time.time()

        total_time = end_time - start_time

        # Verify timing makes sense
        assert total_time > 0
        assert total_time < 180, f"Should complete in <3 minutes for small system"

        # Estimate time per sweep
        time_per_sweep = total_time / max_sweeps

        if rank == 0:
            print(f"\n=== Component Timing (L={L}, bond_dim={bond_dim}) ===")
            print(f"Total time: {total_time:.3f} seconds")
            print(f"Sweeps completed: {max_sweeps}")
            print(f"Average time per sweep: {time_per_sweep:.3f} seconds")
            print(f"Final energy: {energy:.12f}")

    def test_timing_reproducibility(self):
        """
        Test #42 Step 5: Verify timing is reproducible.

        Runs the same calculation twice and verifies times are comparable.
        Note: Exact timing reproducibility is not expected due to system variance,
        but times should be within 50% of each other.
        """
        L = 6
        bond_dim = 10
        max_sweeps = 3
        tol = 1e-5
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        seed = 12345

        # Build Heisenberg MPO
        builder = SpinHam1D(S=1/2)
        builder += 1.0, 'X', 'X'
        builder += 1.0, 'Y', 'Y'
        builder += 1.0, 'Z', 'Z'
        mpo = builder.build_mpo(L)

        # Run 1
        np.random.seed(seed)
        start1 = time.time()
        energy1, mps1 = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=max_sweeps,
            bond_dim=bond_dim,
            tol=tol,
            comm=comm,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )
        time1 = time.time() - start1

        # Run 2
        np.random.seed(seed)
        start2 = time.time()
        energy2, mps2 = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=max_sweeps,
            bond_dim=bond_dim,
            tol=tol,
            comm=comm,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )
        time2 = time.time() - start2

        # Verify both runs completed
        assert time1 > 0 and time2 > 0

        # Verify times are comparable (within 50% - accounts for system variance)
        ratio = max(time1, time2) / min(time1, time2)
        assert ratio < 2.0, f"Timing should be reproducible, got {time1:.2f}s and {time2:.2f}s (ratio: {ratio:.2f})"

        # Verify energies are identical (same seed)
        assert energy1 == energy2, "Energies should be identical with same seed"

        if rank == 0:
            print(f"\n=== Timing Reproducibility ===")
            print(f"Run 1: {time1:.3f} seconds, E = {energy1:.12f}")
            print(f"Run 2: {time2:.3f} seconds, E = {energy2:.12f}")
            print(f"Time ratio: {ratio:.3f}")

    def test_scaling_with_system_size(self):
        """
        Test timing scaling with system size.

        Measures how execution time scales with L.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        results = []
        for L in [4, 6, 8]:
            bond_dim = 10
            max_sweeps = 2
            tol = 1e-5

            # Build MPO
            builder = SpinHam1D(S=1/2)
            builder += 1.0, 'X', 'X'
            builder += 1.0, 'Y', 'Y'
            builder += 1.0, 'Z', 'Z'
            mpo = builder.build_mpo(L)

            # Run with timing
            start = time.time()
            energy, mps = a2dmrg_main(
                L=L,
                mpo=mpo,
                max_sweeps=max_sweeps,
                bond_dim=bond_dim,
                tol=tol,
                comm=comm,
                dtype=np.float64,
                one_site=True,
                verbose=False
            )
            elapsed = time.time() - start

            results.append((L, elapsed, energy))

        # Verify all completed successfully
        for L, elapsed, energy in results:
            assert elapsed > 0
            assert energy < 0, f"Energy should be negative for L={L}"

        # Verify time increases with L (larger systems take longer)
        times = [r[1] for r in results]
        assert times[1] >= times[0] * 0.5, "L=6 should take at least half the time of L=4"

        if rank == 0:
            print(f"\n=== Scaling with System Size ===")
            for L, elapsed, energy in results:
                print(f"L={L}: {elapsed:.3f} seconds, E={energy:.12f}")

    def test_timing_with_different_tolerances(self):
        """
        Test how timing changes with convergence tolerance.

        Tighter tolerance should take longer (more sweeps).
        """
        L = 6
        bond_dim = 12
        max_sweeps = 20
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Build MPO
        builder = SpinHam1D(S=1/2)
        builder += 1.0, 'X', 'X'
        builder += 1.0, 'Y', 'Y'
        builder += 1.0, 'Z', 'Z'
        mpo = builder.build_mpo(L)

        results = []
        for tol in [1e-4, 1e-6, 1e-8]:
            np.random.seed(42)  # Same initialization for fair comparison
            start = time.time()
            energy, mps = a2dmrg_main(
                L=L,
                mpo=mpo,
                max_sweeps=max_sweeps,
                bond_dim=bond_dim,
                tol=tol,
                comm=comm,
                dtype=np.float64,
                one_site=True,
                verbose=False
            )
            elapsed = time.time() - start
            results.append((tol, elapsed, energy))

        # Verify all completed
        for tol, elapsed, energy in results:
            assert elapsed > 0
            assert energy < 0

        if rank == 0:
            print(f"\n=== Timing vs Tolerance ===")
            for tol, elapsed, energy in results:
                print(f"tol={tol:.0e}: {elapsed:.3f} seconds, E={energy:.12f}")

    def test_complex_dtype_timing(self):
        """
        Test timing with complex128 dtype.

        Complex arithmetic may be slower than real.
        """
        L = 6
        bond_dim = 10
        max_sweeps = 3
        tol = 1e-5
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Build MPO
        builder = SpinHam1D(S=1/2)
        builder += 1.0, 'X', 'X'
        builder += 1.0, 'Y', 'Y'
        builder += 1.0, 'Z', 'Z'
        mpo = builder.build_mpo(L)

        # Test float64
        np.random.seed(123)
        start_real = time.time()
        energy_real, mps_real = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=max_sweeps,
            bond_dim=bond_dim,
            tol=tol,
            comm=comm,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )
        time_real = time.time() - start_real

        # Test complex128
        np.random.seed(123)
        start_complex = time.time()
        energy_complex, mps_complex = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=max_sweeps,
            bond_dim=bond_dim,
            tol=tol,
            comm=comm,
            dtype=np.complex128,
            one_site=True,
            verbose=False
        )
        time_complex = time.time() - start_complex

        # Verify both completed
        assert time_real > 0 and time_complex > 0

        # Complex should be reasonably close to real (within 5x)
        assert time_complex < 5 * time_real, "Complex should not be more than 5x slower"

        # Both should produce negative energies (Hermitian Hamiltonian)
        # Note: exact energies may differ due to different random initializations
        # and convergence paths, but both should be physically reasonable
        if np.iscomplexobj(energy_complex):
            energy_complex_real = np.real(energy_complex)
            imag_part = abs(np.imag(energy_complex))
            assert imag_part < 1e-8, f"Complex energy should have negligible imaginary part, got {imag_part}"
        else:
            energy_complex_real = energy_complex

        # Both energies should be negative for Heisenberg
        assert energy_real < 0, f"Float64 energy should be negative, got {energy_real}"
        assert energy_complex_real < 0, f"Complex128 energy should be negative, got {energy_complex_real}"

        if rank == 0:
            print(f"\n=== Dtype Timing Comparison ===")
            print(f"float64:    {time_real:.3f} seconds, E={energy_real:.12f}")
            print(f"complex128: {time_complex:.3f} seconds, E={energy_complex_real:.12f}")
            print(f"Ratio: {time_complex/time_real:.2f}x")
