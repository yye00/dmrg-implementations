"""CLI entry point: ``python -m rlbfgs ...``.

Examples::

    python -m rlbfgs heisenberg --L 12 --chi 20 \\
        --warmup-epochs 300 --polish-epochs 400

    python -m rlbfgs josephson --L 8 --chi 20 --n-max 2 \\
        --warmup-epochs 800 --polish-epochs 2000 --polish-ridge 1e-4
"""

from __future__ import annotations

import argparse
import sys

from .driver import run_heisenberg, run_tfim, run_josephson


def _common(p):
    p.add_argument("--L", type=int, default=12)
    p.add_argument("--chi", type=int, default=20)
    p.add_argument("--warmup-epochs", type=int, default=500)
    p.add_argument("--warmup-history", type=int, default=20)
    p.add_argument("--polish-epochs", type=int, default=1500)
    p.add_argument("--polish-history", type=int, default=30)
    p.add_argument("--polish-tol", type=float, default=1e-12)
    p.add_argument("--polish-ridge", type=float, default=1e-6)
    p.add_argument("--line-search", choices=["wolfe", "armijo"], default="wolfe")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="rlbfgs",
        description="Riemannian L-BFGS for MPS ground-state optimization (CPU numpy).",
    )
    sub = parser.add_subparsers(dest="model", required=True)

    ph = sub.add_parser("heisenberg")
    _common(ph)
    ph.add_argument("--j", type=float, default=1.0)
    ph.add_argument("--bz", type=float, default=0.0)

    pt = sub.add_parser("tfim")
    _common(pt)
    pt.add_argument("--j", type=float, default=1.0)
    pt.add_argument("--hx", type=float, default=1.0)

    pj = sub.add_parser("josephson")
    _common(pj)
    pj.add_argument("--E_J", type=float, default=1.0)
    pj.add_argument("--E_C", type=float, default=0.5)
    pj.add_argument("--mu", type=float, default=0.0)
    pj.add_argument("--n-max", type=int, default=2)

    args = parser.parse_args(argv)

    kw = dict(
        warmup_epochs=args.warmup_epochs,
        warmup_history=args.warmup_history,
        polish_epochs=args.polish_epochs,
        polish_history=args.polish_history,
        polish_tol=args.polish_tol,
        polish_ridge=args.polish_ridge,
        line_search=args.line_search,
        seed=args.seed,
        log_every=args.log_every,
    )

    if args.model == "heisenberg":
        run_heisenberg(L=args.L, chi=args.chi, j=args.j, bz=args.bz, **kw)
    elif args.model == "tfim":
        run_tfim(L=args.L, chi=args.chi, j=args.j, hx=args.hx, **kw)
    elif args.model == "josephson":
        run_josephson(
            L=args.L, chi=args.chi,
            E_J=args.E_J, E_C=args.E_C, mu=args.mu, n_max=args.n_max,
            **kw,
        )


if __name__ == "__main__":
    main()
