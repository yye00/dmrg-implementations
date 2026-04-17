"""CLI entry point: ``python -m radam ...``.

Usage examples
--------------

::

    python -m radam heisenberg --L 10 --chi 16 --epochs 500 --lr 1e-2
    python -m radam tfim      --L 10 --chi 16 --epochs 500 --lr 1e-2 --hx 1.0
"""

from __future__ import annotations

import argparse
import sys

from .driver import run_heisenberg, run_tfim


def _common_args(p):
    p.add_argument("--L", type=int, default=10, help="number of sites")
    p.add_argument("--chi", type=int, default=16, help="MPS bond dimension")
    p.add_argument("--epochs", type=int, default=500, help="max epochs")
    p.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument(
        "--lr-schedule",
        choices=["cosine", "none"],
        default="cosine",
        help="learning-rate schedule",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(
        prog="radam",
        description="Riemannian Adam for MPS ground-state optimization (CPU numpy).",
    )
    sub = parser.add_subparsers(dest="model", required=True)

    p_h = sub.add_parser("heisenberg", help="Heisenberg XXX chain")
    _common_args(p_h)
    p_h.add_argument("--j", type=float, default=1.0)
    p_h.add_argument("--bz", type=float, default=0.0)

    p_t = sub.add_parser("tfim", help="Transverse-field Ising")
    _common_args(p_t)
    p_t.add_argument("--j", type=float, default=1.0)
    p_t.add_argument("--hx", type=float, default=1.0)

    args = parser.parse_args(argv)
    schedule = None if args.lr_schedule == "none" else args.lr_schedule

    kw = dict(
        chi=args.chi,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        tol=args.tol,
        max_epochs=args.epochs,
        lr_schedule=schedule,
        min_lr=args.min_lr,
        seed=args.seed,
        log_every=args.log_every,
    )

    if args.model == "heisenberg":
        run_heisenberg(L=args.L, j=args.j, bz=args.bz, **kw)
    elif args.model == "tfim":
        run_tfim(L=args.L, j=args.j, hx=args.hx, **kw)


if __name__ == "__main__":
    main()
