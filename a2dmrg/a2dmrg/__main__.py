"""
Command-line interface for A2DMRG.

Usage:
    python -m a2dmrg --sites 20 --bond-dim 50
    python -m a2dmrg --sites 10 --model heisenberg --dtype complex128
    mpirun -np 4 python -m a2dmrg --sites 20 --bond-dim 100
"""

# CRITICAL: Must import this BEFORE any quimb imports (Python 3.13+ compatibility)
import fix_quimb_python313  # noqa: F401

import argparse
import sys
import numpy as np
from a2dmrg.mpi_compat import MPI
from a2dmrg.dmrg import a2dmrg_main


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='A2DMRG: Additive Two-Level Parallel DMRG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Heisenberg model with 20 sites, bond dimension 50
  python -m a2dmrg --sites 20 --bond-dim 50

  # Run with complex dtype
  python -m a2dmrg --sites 10 --dtype complex128

  # Run Bose-Hubbard model
  python -m a2dmrg --sites 12 --model bose-hubbard

  # Run with MPI (4 processes)
  mpirun -np 4 python -m a2dmrg --sites 20 --bond-dim 100

  # Run with custom parameters
  python -m a2dmrg --sites 16 --bond-dim 64 --max-sweeps 30 --tol 1e-12
"""
    )

    # Required arguments
    parser.add_argument(
        '--sites', '-L',
        type=int,
        required=True,
        help='Number of sites in the spin chain'
    )

    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        default='heisenberg',
        choices=['heisenberg', 'bose-hubbard', 'josephson'],
        help='Model to simulate (default: heisenberg)'
    )

    # DMRG parameters
    parser.add_argument(
        '--bond-dim',
        type=int,
        default=100,
        help='Maximum bond dimension (default: 100)'
    )

    parser.add_argument(
        '--max-sweeps',
        type=int,
        default=20,
        help='Maximum number of sweeps (default: 20)'
    )

    parser.add_argument(
        '--tol',
        type=float,
        default=1e-10,
        help='Energy convergence tolerance (default: 1e-10)'
    )

    # Data type
    parser.add_argument(
        '--dtype',
        type=str,
        default='float64',
        choices=['float64', 'complex128'],
        help='Data type for tensors (default: float64)'
    )

    # Algorithm options
    parser.add_argument(
        '--one-site',
        action='store_true',
        help='Use one-site updates (default: two-site)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Enable verbose output (default: True)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Disable verbose output'
    )

    # Model-specific parameters
    parser.add_argument(
        '--J',
        type=float,
        default=1.0,
        help='Coupling strength (model-dependent, default: 1.0)'
    )

    parser.add_argument(
        '--U',
        type=float,
        default=1.0,
        help='On-site interaction (Bose-Hubbard, default: 1.0)'
    )

    parser.add_argument(
        '--hilbert-dim',
        type=int,
        default=3,
        help='Local Hilbert space dimension (Bose-Hubbard, default: 3)'
    )

    return parser.parse_args()


def build_mpo(args):
    """Build MPO based on model selection."""
    if args.model == 'heisenberg':
        from quimb.tensor import SpinHam1D
        builder = SpinHam1D(S=1/2)
        builder += args.J, 'X', 'X'
        builder += args.J, 'Y', 'Y'
        builder += args.J, 'Z', 'Z'
        mpo = builder.build_mpo(args.sites)
        return mpo

    elif args.model == 'bose-hubbard':
        # Build Bose-Hubbard MPO manually
        d = args.hilbert_dim  # Local Hilbert space dimension

        # Build local operators
        # Bosonic creation operator a†
        a_dag = np.zeros((d, d), dtype=complex)
        for i in range(d-1):
            a_dag[i+1, i] = np.sqrt(i+1)

        # Bosonic annihilation operator a
        a = a_dag.T.conj()

        # Number operator n = a† a
        n = np.diag(np.arange(d, dtype=float))

        # On-site interaction term U/2 * n(n-1)
        n_n_minus_1 = np.diag(np.arange(d, dtype=float) * (np.arange(d, dtype=float) - 1))

        # Identity
        I = np.eye(d, dtype=complex)

        # Build MPO using quimb
        # H = -J Σ(a†_i a_{i+1} + h.c.) + U/2 Σ n_i(n_i-1)
        # For simplicity, use a basic implementation
        # This is a simplified version - for production use, need full MPO construction
        raise NotImplementedError("Bose-Hubbard MPO builder not yet fully implemented. Use --model heisenberg for now.")

    elif args.model == 'josephson':
        # Josephson junction array model not implemented
        raise NotImplementedError(
            "Josephson junction model not yet implemented for A2DMRG CLI.\n"
            "The model requires proper Bose-Hubbard operators with local dimension d>2.\n"
            "Use benchmark_data loader with pre-generated Josephson MPOs instead,\n"
            "or use --model heisenberg for testing."
        )

    else:
        raise ValueError(f"Unknown model: {args.model}")


def main():
    """Main entry point for CLI."""
    # Parse arguments
    args = parse_args()

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Convert dtype string to numpy dtype
    if args.dtype == 'float64':
        dtype = np.float64
    elif args.dtype == 'complex128':
        dtype = np.complex128
    else:
        raise ValueError(f"Unknown dtype: {args.dtype}")

    # Handle verbose flag
    verbose = args.verbose and not args.quiet

    # Only rank 0 prints header
    if rank == 0 and verbose:
        print("=" * 80)
        print("A2DMRG: Additive Two-Level Parallel DMRG")
        print("=" * 80)
        print(f"Model:        {args.model}")
        print(f"Sites (L):    {args.sites}")
        print(f"Bond dim:     {args.bond_dim}")
        print(f"Max sweeps:   {args.max_sweeps}")
        print(f"Tolerance:    {args.tol}")
        print(f"Dtype:        {args.dtype}")
        print(f"One-site:     {args.one_site}")
        print(f"MPI ranks:    {size}")
        print("=" * 80)

    # Build MPO
    try:
        mpo = build_mpo(args)
    except Exception as e:
        if rank == 0:
            print(f"Error building MPO: {e}", file=sys.stderr)
        sys.exit(1)

    # Run A2DMRG
    try:
        energy, mps = a2dmrg_main(
            L=args.sites,
            mpo=mpo,
            max_sweeps=args.max_sweeps,
            bond_dim=args.bond_dim,
            tol=args.tol,
            comm=comm,
            dtype=dtype,
            one_site=args.one_site,
            verbose=verbose
        )

        # Only rank 0 prints results
        if rank == 0:
            print("=" * 80)
            print("RESULTS")
            print("=" * 80)
            print(f"Ground state energy: {energy}")
            print(f"Energy per site:     {energy / args.sites}")
            print("=" * 80)

    except Exception as e:
        if rank == 0:
            print(f"Error during A2DMRG execution: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
