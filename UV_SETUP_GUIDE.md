# UV-Based Unified Environment Guide

**Created:** 2026-03-07
**Status:** ✅ Complete - Single unified environment using `uv`

## Overview

This repository now uses a **single unified Python environment** managed by [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver. All three DMRG implementations (PDMRG, PDMRG2, A2DMRG) are installed in editable mode in a single virtual environment.

## Quick Start

### First-Time Setup

```bash
# Clone and enter the repository
cd /home/captain/clawd/work/dmrg-implementations

# Create and activate the unified environment
uv sync

# Install all packages in editable mode (already done by uv sync)
# All three packages are available: pdmrg, pdmrg2, a2dmrg
```

### Activating the Environment

```bash
# Activate the unified virtual environment
source .venv/bin/activate

# Verify installation
python -c "import pdmrg, a2dmrg; print('✓ All packages loaded')"
```

### Running with uv (Recommended)

You can run commands directly with `uv run` without activating:

```bash
# Run PDMRG
uv run mpirun -np 2 python -m pdmrg --sites 40 --bond-dim 50 --model heisenberg

# Run A2DMRG
uv run mpirun -np 2 python -m a2dmrg --sites 40 --bond-dim 50 --model heisenberg

# Run tests
uv run pytest pdmrg/tests/
uv run pytest a2dmrg/tests/

# Run benchmarks
uv run python comprehensive_benchmark.py
```

## Project Structure

```
dmrg-implementations/          # Monorepo root
├── pyproject.toml             # Root project config (shared dependencies)
├── uv.lock                    # Lockfile for reproducible installs
├── .venv/                     # Unified virtual environment
│
├── pdmrg/                     # Real-space parallel DMRG
│   ├── pyproject.toml         # Package metadata
│   └── pdmrg/                 # Source code
│
├── pdmrg2/                    # PROTOTYPE with GPU hooks
│   ├── pyproject.toml         # Package metadata
│   └── pdmrg/                 # Source code (shares namespace)
│
└── a2dmrg/                    # Additive two-level DMRG
    ├── pyproject.toml         # Package metadata
    └── a2dmrg/                # Source code
```

## Dependency Management

### Shared Dependencies (Root Level)

All core dependencies are specified in the root `pyproject.toml`:

```toml
dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "quimb>=1.7.0",
    "mpi4py>=3.1.0",
]
```

### Adding New Dependencies

```bash
# Add a dependency to the unified environment
uv add <package-name>

# Add a dev dependency
uv add --dev <package-name>

# Update dependencies
uv lock

# Sync environment after changes
uv sync
```

### Package-Specific Dependencies

Individual packages inherit from the root. If a package needs unique dependencies, add them to its own `pyproject.toml`:

```toml
# Example: pdmrg2/pyproject.toml
[project.optional-dependencies]
gpu = ["cupy>=12.0.0"]
```

## Development Workflow

### Editable Installs

All three packages are installed in **editable mode** (`pip install -e`). Changes to source code are immediately reflected without reinstallation:

```bash
# Edit pdmrg/pdmrg/dmrg.py
vim pdmrg/pdmrg/dmrg.py

# Run immediately - changes are live
uv run mpirun -np 2 python -m pdmrg --sites 40 --bond-dim 50
```

### Testing

```bash
# Run all tests across all packages
uv run pytest

# Run tests for a specific package
uv run pytest pdmrg/tests/
uv run pytest a2dmrg/tests/

# Run specific test file
uv run pytest test_warmup_policy.py

# Run with MPI (for parallel tests)
mpirun -np 2 uv run pytest pdmrg/tests/test_heisenberg.py
```

### Code Formatting and Linting

```bash
# Format code with black
uv run black pdmrg/ pdmrg2/ a2dmrg/

# Lint with ruff
uv run ruff check pdmrg/ pdmrg2/ a2dmrg/

# Type check with mypy
uv run mypy pdmrg/pdmrg/ a2dmrg/a2dmrg/
```

## Advantages of UV

1. **Speed**: 10-100× faster than pip for dependency resolution
2. **Reproducibility**: `uv.lock` ensures identical environments across machines
3. **Monorepo Support**: Native workspace support for multiple packages
4. **No Activation Required**: `uv run` works without activating the venv
5. **Dependency Conflict Detection**: Better resolver than pip

## Migration from Old Setup

### Before (Separate Environments)

```bash
# Old workflow with 3 separate virtualenvs
cd pdmrg && python -m venv venv && source venv/bin/activate
pip install -e .
deactivate

cd ../pdmrg2 && python -m venv venv && source venv/bin/activate
pip install -e .
deactivate

cd ../a2dmrg && python -m venv venv && source venv/bin/activate
pip install -e .
```

### After (Unified Environment)

```bash
# New workflow with single uv environment
uv sync

# All three packages available immediately
uv run python -c "import pdmrg, a2dmrg; print('✓')"
```

### Cleanup Old Environments

```bash
# Remove old per-package virtualenvs (optional)
rm -rf pdmrg/venv pdmrg2/venv a2dmrg/venv .venv-bench

# Keep only the unified .venv/
```

## Common Commands

### Running DMRG Implementations

```bash
# PDMRG (real-space parallel)
uv run mpirun -np 4 python -m pdmrg --sites 80 --bond-dim 100 --model random_tfim

# A2DMRG (two-level parallel)
uv run mpirun -np 4 python -m a2dmrg --sites 80 --bond-dim 100 --model heisenberg

# PDMRG2 (prototype - not recommended)
uv run mpirun -np 2 python -m pdmrg2 --sites 40 --bond-dim 50
```

### Benchmarking

```bash
# Run comprehensive benchmarks
uv run python comprehensive_benchmark.py

# Run publication benchmarks
uv run python publication_benchmark.py
```

### Installing Optional Dependencies

```bash
# Install GPU support (for pdmrg2 future work)
uv sync --extra gpu

# Install benchmarking tools
uv sync --extra bench

# Install all optional dependencies
uv sync --all-extras
```

## Troubleshooting

### "Module not found" errors

```bash
# Resync the environment
uv sync

# Verify packages are installed
uv pip list | grep -E "pdmrg|a2dmrg"
```

### Dependency conflicts

```bash
# Update lockfile
uv lock --upgrade

# Force reinstall
rm -rf .venv uv.lock
uv sync
```

### MPI issues

```bash
# Check MPI installation
which mpirun
mpirun --version

# Test MPI with uv
uv run mpirun -np 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
```

## Next Steps

After setting up the environment:

1. ✅ Unified environment created
2. **Next: Enforce exact SVD for boundary merge** (CRITICAL)
3. Refactor shared pdmrg/pdmrg2 components
4. Update A2DMRG warmup policy
5. Add comprehensive tests
6. Update documentation

## References

- **uv Documentation**: https://github.com/astral-sh/uv
- **Python Packaging Guide**: https://packaging.python.org/
- **Monorepo Best Practices**: https://monorepo.tools/
