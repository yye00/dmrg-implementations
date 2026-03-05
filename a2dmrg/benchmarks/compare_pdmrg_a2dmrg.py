#!/usr/bin/env python3
"""
Benchmark comparing PDMRG and A2DMRG on the same problems.

Models tested:
1. Heisenberg spin-1/2 chain
2. Random Transverse-Field Ising Model (RTFIM)
3. Bose-Hubbard / Josephson junction (complex128)

Usage:
    # Serial comparison
    python benchmarks/compare_pdmrg_a2dmrg.py
    
    # MPI comparison (for scaling)
    mpirun -np 4 python benchmarks/compare_pdmrg_a2dmrg.py --mpi
"""

import sys
import os
import time
import json
import argparse
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')

import quimb.tensor as qtn
from quimb.tensor import SpinHam1D


# ============================================================
# Hamiltonians (matching PDMRG)
# ============================================================

def build_heisenberg_mpo(L, j=1.0, bz=0.0):
    """Heisenberg XXX chain."""
    return qtn.MPO_ham_heis(L=L, j=j, bz=bz, cyclic=False)


def build_random_tfim_mpo(L, J_mean=1.0, J_std=0.5, h_mean=1.0, h_std=0.5, 
                          seed=42, dtype='float64'):
    """Random Transverse-Field Ising Model."""
    np.random.seed(seed)
    J = J_mean + J_std * np.random.randn(L - 1)
    h = h_mean + h_std * np.random.randn(L)
    
    builder = SpinHam1D(S=1/2)
    for i in range(L - 1):
        builder[i, i+1] += -J[i], 'Z', 'Z'
    for i in range(L):
        builder[i] += -h[i], 'X'
    
    mpo = builder.build_mpo(L)
    
    if dtype == 'complex128':
        for i in range(L):
            mpo[i].modify(data=mpo[i].data.astype('complex128'))
    
    return mpo, {'J': J, 'h': h}


def build_bose_hubbard_mpo(L, t=1.0, U=4.0, mu=2.0, n_max=3, dtype='complex128'):
    """Bose-Hubbard / Josephson junction model."""
    d = n_max + 1
    
    # Build operator matrices for truncated bosons
    a_dag = np.zeros((d, d), dtype=dtype)
    for n in range(d - 1):
        a_dag[n + 1, n] = np.sqrt(n + 1)
    a = a_dag.conj().T
    n_op = a_dag @ a
    
    builder = SpinHam1D(S=(d - 1) / 2)
    builder.add_term(-t, a_dag, a)
    builder.add_term(-t, a, a_dag)
    
    n2 = n_op @ n_op
    onsite = (U / 2.0) * (n2 - n_op) - mu * n_op
    builder.add_term(1.0, onsite)
    
    return builder.build_mpo(L)


# ============================================================
# Run DMRG Methods
# ============================================================

def run_quimb_dmrg2(mpo, L, bond_dim, max_sweeps=20, tol=1e-10):
    """Run quimb DMRG2 reference."""
    t0 = time.time()
    dmrg = qtn.DMRG2(mpo, bond_dims=bond_dim)
    dmrg.solve(max_sweeps=max_sweeps, tol=tol, verbosity=0)
    t1 = time.time()
    
    return {
        'energy': float(np.real(dmrg.energy)),
        'time': t1 - t0,
        'sweeps': len(dmrg.energies),
    }


def run_a2dmrg(mpo, L, bond_dim, max_sweeps=10, warmup_sweeps=2, tol=1e-10, comm=None):
    """Run A2DMRG."""
    from a2dmrg.dmrg import a2dmrg_main
    from a2dmrg.mpi_compat import MPI
    
    if comm is None:
        comm = MPI.COMM_SELF
    
    rank = comm.Get_rank() if hasattr(comm, 'Get_rank') else 0
    
    # Determine dtype from MPO
    dtype = mpo[0].data.dtype
    
    t0 = time.time()
    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=max_sweeps,
        bond_dim=bond_dim,
        tol=tol,
        comm=comm,
        warmup_sweeps=warmup_sweeps,
        one_site=True,
        dtype=dtype,
        verbose=False,
    )
    t1 = time.time()
    
    return {
        'energy': float(np.real(energy)),
        'time': t1 - t0,
        'sweeps': max_sweeps + warmup_sweeps,
    }


def run_pdmrg(mpo, L, bond_dim, max_sweeps=10, warmup_sweeps=2, tol=1e-10, comm=None):
    """Run PDMRG."""
    try:
        from pdmrg.dmrg import pdmrg_main
        from pdmrg.mps.canonical import create_random_mps
        
        t0 = time.time()
        # PDMRG interface may differ - adjust as needed
        mps = create_random_mps(L, bond_dim=bond_dim)
        energy, mps = pdmrg_main(
            mps=mps,
            mpo=mpo,
            max_sweeps=max_sweeps,
            bond_dim=bond_dim,
            tol=tol,
            comm=comm,
            warmup_sweeps=warmup_sweeps,
        )
        t1 = time.time()
        
        return {
            'energy': float(np.real(energy)),
            'time': t1 - t0,
            'sweeps': max_sweeps + warmup_sweeps,
        }
    except Exception as e:
        return {'error': str(e)}


# ============================================================
# Benchmark Suite
# ============================================================

def run_benchmark_suite(use_mpi=False, quick=False):
    """Run complete benchmark suite."""
    from a2dmrg.mpi_compat import MPI
    
    comm = MPI.COMM_WORLD if use_mpi else MPI.COMM_SELF
    rank = comm.Get_rank() if hasattr(comm, 'Get_rank') else 0
    size = comm.Get_size() if hasattr(comm, 'Get_size') else 1
    
    if rank == 0:
        print("=" * 70)
        print("PDMRG vs A2DMRG Comparison Benchmark")
        print(f"MPI: {'Yes (np={})'.format(size) if use_mpi else 'No (serial)'}")
        print("=" * 70)
    
    # Configurations
    if quick:
        configs = [
            {'model': 'heisenberg', 'L': 20, 'bond_dim': 30},
        ]
    else:
        configs = [
            # Heisenberg
            {'model': 'heisenberg', 'L': 20, 'bond_dim': 50},
            {'model': 'heisenberg', 'L': 40, 'bond_dim': 50},
            # Random TFIM
            {'model': 'random_tfim', 'L': 20, 'bond_dim': 50},
            {'model': 'random_tfim', 'L': 40, 'bond_dim': 50},
            # Bose-Hubbard (complex128)
            {'model': 'bose_hubbard', 'L': 10, 'bond_dim': 30},
            {'model': 'bose_hubbard', 'L': 20, 'bond_dim': 30},
        ]
    
    results = []
    
    for cfg in configs:
        model = cfg['model']
        L = cfg['L']
        bond_dim = cfg['bond_dim']
        
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Model: {model}, L={L}, bond_dim={bond_dim}")
            print("=" * 70)
        
        # Build MPO
        if model == 'heisenberg':
            mpo = build_heisenberg_mpo(L)
            dtype = 'float64'
        elif model == 'random_tfim':
            mpo, _ = build_random_tfim_mpo(L)
            dtype = 'float64'
        elif model == 'bose_hubbard':
            mpo = build_bose_hubbard_mpo(L)
            dtype = 'complex128'
        
        result = {'model': model, 'L': L, 'bond_dim': bond_dim, 'dtype': dtype, 'np': size}
        
        # Run quimb DMRG2 (reference, serial only on rank 0)
        if rank == 0:
            print("  Quimb DMRG2...", end=" ", flush=True)
            quimb_res = run_quimb_dmrg2(mpo, L, bond_dim)
            print(f"E={quimb_res['energy']:.10f}, t={quimb_res['time']:.2f}s")
            result['quimb'] = quimb_res
        
        # Run A2DMRG
        if rank == 0:
            print(f"  A2DMRG (np={size})...", end=" ", flush=True)
        
        a2dmrg_res = run_a2dmrg(mpo, L, bond_dim, comm=comm)
        
        if rank == 0:
            print(f"E={a2dmrg_res['energy']:.10f}, t={a2dmrg_res['time']:.2f}s")
            result['a2dmrg'] = a2dmrg_res
            
            # Compare
            if 'quimb' in result:
                energy_diff = abs(result['a2dmrg']['energy'] - result['quimb']['energy'])
                print(f"  Energy difference: {energy_diff:.2e}")
                result['energy_diff'] = energy_diff
        
        results.append(result)
    
    # Save results
    if rank == 0:
        output_file = f'benchmarks/comparison_np{size}.json'
        os.makedirs('benchmarks', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n{'='*70}")
        print(f"Results saved to {output_file}")
        print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='PDMRG vs A2DMRG Comparison')
    parser.add_argument('--mpi', action='store_true', help='Use MPI parallelization')
    parser.add_argument('--quick', action='store_true', help='Quick test run')
    args = parser.parse_args()
    
    run_benchmark_suite(use_mpi=args.mpi, quick=args.quick)


if __name__ == '__main__':
    main()
