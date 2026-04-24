# radam-gpu (experimental)

Riemannian Adam (RAdam) optimizer for MPS ground-state search on AMD MI300X, not benchmarked in this paper.

This directory contains a Riemannian Adam implementation for variational MPS optimization.
It was developed as an alternative to DMRG sweeps but was not evaluated in the paper:

- No results from this variant appear in the paper.
- It is not built by `gpu-rocm/build_all.sh`.
- It is not covered by the reproducibility checklist in `REPRODUCIBILITY.md`.

For the published implementations, see the parent `gpu-rocm/` directory.
