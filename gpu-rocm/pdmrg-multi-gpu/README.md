# pdmrg-multi-gpu (experimental)

Multi-GPU Parallel DMRG prototype, not benchmarked in this paper.

This directory contains a work-in-progress extension of `pdmrg-gpu` to multiple
AMD MI300X GPUs via HIP peer-to-peer transfers. It was developed as a prototype
and is retained here for transparency, but:

- No results from this variant appear in the paper.
- It is not built by `gpu-rocm/build_all.sh`.
- It is not covered by the reproducibility checklist in `REPRODUCIBILITY.md`.

For the published implementations, see the parent `gpu-rocm/` directory.
