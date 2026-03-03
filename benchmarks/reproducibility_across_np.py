#!/usr/bin/env python3
"""Reproducibility across np (Feature 4)

Runs the same A2DMRG problem under different `mpirun -np` settings and checks
that the final energies match within a given tolerance.

Usage:
  source ~/.profile
  ./.venv-bench/bin/python benchmarks/reproducibility_across_np.py --nps 1,2,4 --L 20 --bond-dim 50

Notes
-----
Exact (bitwise) agreement across different np is often unrealistic due to:
- floating point non-associativity in distributed reductions
- solver / BLAS / threading nondeterminism

So this script defaults to an energy tolerance check (abs + relative).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class RunResult:
    np: int
    energy: float
    stdout: str
    stderr: str


def _run_one(np_count: int, args: argparse.Namespace) -> RunResult:
    env = os.environ.copy()
    # Reduce nondeterminism from threaded BLAS.
    for k in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        env.setdefault(k, "1")

    code = (
        "import time, numpy as np; import quimb.tensor as qtn; "
        "from a2dmrg.dmrg import a2dmrg_main; from mpi4py import MPI; "
        "comm=MPI.COMM_WORLD; "
        f"L={args.L}; bond_dim={args.bond_dim}; max_sweeps={args.max_sweeps}; warmup={args.warmup_sweeps}; "
        f"seed={args.seed}; tol={args.tol}; "
        "H=qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False); "
        "np.random.seed(seed); "
        "E,_=a2dmrg_main(L=L, mpo=H, max_sweeps=max_sweeps, bond_dim=bond_dim, tol=tol, "
        "             comm=comm, dtype=np.float64, one_site=True, warmup_sweeps=warmup, verbose=False); "
        "import json; "
        "print(json.dumps({'energy': float(np.real(E))})) if comm.Get_rank()==0 else None"
    )

    cmd = [
        "mpirun",
        "-np",
        str(np_count),
        "--oversubscribe",
        "env",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        sys.executable,
        "-c",
        code,
    ]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"np={np_count} failed (rc={p.returncode})\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")

    last = None
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and "energy" in line:
            last = line
    if last is None:
        raise RuntimeError(f"np={np_count} produced no JSON energy. STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    energy = json.loads(last)["energy"]
    return RunResult(np=np_count, energy=float(energy), stdout=p.stdout, stderr=p.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify A2DMRG energy reproducibility across np")
    ap.add_argument("--nps", default="1,2,4", help="Comma-separated np list")
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--bond-dim", type=int, default=8)
    ap.add_argument("--max-sweeps", type=int, default=6)
    ap.add_argument("--warmup-sweeps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--energy-atol", type=float, default=1e-8)
    ap.add_argument("--energy-rtol", type=float, default=0.0)
    args = ap.parse_args()

    nps = [int(x.strip()) for x in args.nps.split(",") if x.strip()]
    results = []
    for n in nps:
        r = _run_one(n, args)
        results.append(r)
        print(f"np={n}: E={r.energy:.15f}")

    e0 = results[0].energy
    ok = True
    for r in results[1:]:
        diff = abs(r.energy - e0)
        thresh = args.energy_atol + args.energy_rtol * abs(e0)
        if diff > thresh:
            ok = False
        print(f"  ΔE(np={r.np} vs np={results[0].np}) = {diff:.3e} (threshold {thresh:.3e})")

    if not ok:
        print("FAIL: reproducibility check did not meet tolerance", file=sys.stderr)
        return 2

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
