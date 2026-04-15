"""Example: optimize a 16-site Heisenberg chain with R-Adam.

Usage::

    python examples/run_heisenberg.py
"""

from __future__ import annotations

import numpy as np

from radam.driver import run_heisenberg


def main():
    result = run_heisenberg(
        L=16,
        chi=32,
        j=1.0,
        bz=0.0,
        lr=5e-3,
        max_epochs=500,
        lr_schedule="cosine",
        seed=0,
        log_every=25,
        tol=1e-7,
    )
    print()
    print(f"Final energy: {result['energy']:.10f}")
    print(f"Gradient norm: {result['grad_norm']:.3e}")
    print(f"Wall time: {result['wall_time']:.2f}s")
    print(f"Converged: {result['converged']}")

    # For reference, the exact ground-state energy per bond of the
    # infinite antiferromagnetic Heisenberg chain is  -ln(2) + 1/4
    # ~= -0.4432.  For a finite chain we just print the energy
    # per bond as a sanity check.
    print(f"Energy per bond: {result['energy'] / (16 - 1):.6f}")


if __name__ == "__main__":
    main()
