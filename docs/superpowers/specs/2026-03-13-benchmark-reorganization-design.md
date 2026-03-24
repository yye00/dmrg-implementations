# Benchmark Suite Reorganization

## Goal

Reorganize the benchmarks/ directory from ~50 accumulated files into a clean, extensible structure with a single CLI entry point. Supports two workflows: quick development validation and full publication-quality benchmarking. All 9 current implementations covered, easy to add more.

## Directory Layout

```
benchmarks/
├── README.md                       # Single comprehensive guide
├── run.py                          # CLI entry point
├── lib/
│   ├── __init__.py
│   ├── registry.py                 # Implementation configs
│   ├── data_loader.py              # MPS/MPO binary loading
│   ├── hardware.py                 # CPU/thread detection
│   ├── report.py                   # Markdown report generation
│   └── runners/
│       ├── __init__.py
│       ├── quimb_runner.py         # quimb DMRG1/2
│       ├── pdmrg_runner.py         # pdmrg, pdmrg-opt, pdmrg-cotengra
│       ├── a2dmrg_runner.py        # a2dmrg
│       └── gpu_runner.py           # dmrg-gpu, dmrg2-gpu, pdmrg-gpu, pdmrg-gpu-opt
├── data/
│   ├── generate.py                 # Binary MPS/MPO generation (seed=42)
│   ├── verify.py                   # Loader verification
│   └── *.bin, *.json               # Data files
├── validation/
│   ├── validate.py                 # Correctness checks
│   └── gold_standard.json          # CPU reference results
├── performance/
│   ├── benchmark.py                # Timing suite
│   └── scaling.py                  # Scaling studies
├── results/
│   └── .gitkeep
└── scripts/
    └── (shell wrappers if needed)
```

## CLI Interface

```
./run.py validate [--impl X] [--model X] [--np 2,4] [--threads 1,4]
./run.py benchmark [--impl X] [--model X] [--size X] [--np 2,4] [--threads 4]
./run.py scale [--impl X] [--np 1,2,4,8] [--threads 1,2,4,8]
./run.py report [--input results/latest.json]
./run.py generate-data
```

Flags:
- `--impl`: filter implementations (comma-separated or repeated)
- `--model`: heisenberg, josephson
- `--size`: small, medium, large
- `--np`: mpirun -np for CPU parallel impls, HIP stream count for GPU impls
- `--threads`: OPENBLAS_NUM_THREADS for CPU implementations
- `--output`: results output path

## Implementation Registry

```python
IMPLEMENTATIONS = {
    "quimb-dmrg1":     {"type": "cpu-serial",   "supports_threads": True},
    "quimb-dmrg2":     {"type": "cpu-serial",   "supports_threads": True},
    "pdmrg":           {"type": "cpu-parallel",  "supports_np": True, "supports_threads": True},
    "pdmrg-opt":          {"type": "cpu-parallel",  "supports_np": True, "supports_threads": True},
    "pdmrg-cotengra":  {"type": "cpu-parallel",  "supports_np": True, "supports_threads": True},
    "a2dmrg":          {"type": "cpu-parallel",  "supports_np": True, "supports_threads": True},
    "dmrg-gpu":        {"type": "gpu-serial"},
    "dmrg2-gpu":       {"type": "gpu-serial"},
    "pdmrg-gpu":       {"type": "gpu-parallel",  "supports_np": True},
    "pdmrg-gpu-opt":      {"type": "gpu-parallel",  "supports_np": True},
}
```

## Models & Sizes

| Model | d | Type | Small | Medium | Large |
|-------|---|------|-------|--------|-------|
| Heisenberg | 2 | Real | L=12 chi=20 | L=20 chi=50 | L=40 chi=100 |
| Josephson | 5 | Complex | L=8 chi=20 | L=12 chi=50 | L=16 chi=100 |

## File Migration

### Kept (relocated)
- benchmark_data_loader.py + load_mps_mpo.py → lib/data_loader.py
- hardware_config.py → lib/hardware.py
- generate_report.py → lib/report.py
- serialize_mps_mpo.py → data/generate.py
- verify_loaders.py → data/verify.py
- cpu_gold_standard_results.json → validation/gold_standard.json
- benchmark_data/*.bin,*.json → data/

### Deleted (preserved in git history)
- All .log files
- Historical result JSONs (heisenberg_benchmark_results.2026-02-20_*.json, etc.)
- One-off test scripts (test_josephson_d5.py, test_quick_timing.py, etc.)
- Redundant docs (QUICKSTART.md, STATUS.md, QUICK_REFERENCE.md, etc.)
- reports/ directory
- Superseded scripts (comprehensive_benchmark.py, cpu_gpu_benchmark.py, etc.)
- benchmark_utils.py (absorbed into lib/)

## README Sections

1. Overview — what and why
2. Quick Start — validate, benchmark, report in 3 commands
3. Implementations — table with type, np/threads support
4. Models — Heisenberg and Josephson physics summary
5. Problem Sizes — small/medium/large with expected runtimes
6. CLI Reference — all subcommands and flags with examples
7. Adding a New Implementation — registry entry + runner
8. Data Format — binary MPS/MPO spec
9. Gold Standard Results — reference energy table
