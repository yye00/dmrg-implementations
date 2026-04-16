"""Example: optimize an 8-site Josephson junction array with R-LBFGS.

Usage::

    python examples/run_josephson.py
"""

from __future__ import annotations

import numpy as np

from rlbfgs.driver import run_josephson


def main():
    result = run_josephson(
        L=8, chi=20, n_max=2,
        warmup_epochs=800, warmup_history=20,
        polish_epochs=2000, polish_history=30,
        polish_tol=1e-13, polish_ridge=1e-4,
        seed=0, log_every=100,
    )
    print()
    print(f"Final energy:    {result['energy']:.14f}")
    print(f"Gradient norm:   {result['grad_norm']:.3e}")
    print(f"Wall time:       {result['wall_time']:.2f}s")
    print(f"Converged:       {result['converged']}")
    print(f"Warmup epochs:   {result['warmup_epochs']}")
    print(f"Polish epochs:   {result['polish_epochs']}")
    # Reference from quimb single-site DMRG at the same bond dim.
    DMRG1_CHI20 = -2.84379784155192
    print(f"DMRG1 chi=20 reference: {DMRG1_CHI20}")
    print(f"Gap: {result['energy'] - DMRG1_CHI20:.3e}")


if __name__ == "__main__":
    main()
