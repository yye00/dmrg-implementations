# Contributing to DMRG Implementations

Thank you for your interest in contributing to this project!

## Setup Instructions

### Prerequisites

- Python 3.13+ (tested with 3.13 and 3.14)
- MPI implementation (OpenMPI or MPICH)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yye00/dmrg-implementations.git
   cd dmrg-implementations
   ```

2. **Set up virtual environments for each implementation:**

   Each implementation (a2dmrg, pdmrg, pdmrg2) has its own virtual environment with specific dependencies.

   ```bash
   # For PDMRG
   cd pdmrg
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt  # if available
   deactivate
   cd ..

   # For A2DMRG
   cd a2dmrg
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt  # if available
   deactivate
   cd ..

   # For PDMRG2
   cd pdmrg2
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt  # if available
   deactivate
   cd ..
   ```

3. **Verify installation:**
   ```bash
   # Run a quick benchmark
   python benchmarks/heisenberg_benchmark.py
   ```

### Running Benchmarks

All benchmarks should be run from the repository root directory:

```bash
# Short correctness test (L=12)
python benchmarks/heisenberg_benchmark.py

# Long benchmark (L=48, 50 sweeps)
python benchmarks/heisenberg_long_benchmark.py

# Josephson junction test (complex128)
python a2dmrg/benchmarks/josephson_correctness_benchmark.py
```

### Running DMRG Simulations

The helper scripts `run_pdmrg_np1.py` and `run_a2dmrg_np1.py` use relative paths and can be run from the repository root:

```bash
# PDMRG with np=1
pdmrg/venv/bin/python run_pdmrg_np1.py

# A2DMRG with np=1
a2dmrg/venv/bin/python run_a2dmrg_np1.py

# PDMRG with MPI (np=4)
mpirun -np 4 pdmrg/venv/bin/python run_pdmrg_np1.py

# A2DMRG with MPI (np=4)
mpirun -np 4 a2dmrg/venv/bin/python run_a2dmrg_np1.py
```

### Development Guidelines

1. **Code Style:**
   - Follow PEP 8 conventions
   - Use type hints where appropriate
   - Add docstrings to all public functions

2. **Testing:**
   - Run the short benchmark suite before submitting changes
   - Ensure machine precision accuracy (ΔE < 1e-14) on correctness tests
   - Test with multiple MPI process counts (np=1,2,4)

3. **Path Handling:**
   - Always use relative paths or `os.path` for file operations
   - Never hardcode absolute paths in scripts
   - Use `os.path.dirname(os.path.abspath(__file__))` to get script directory

4. **Documentation:**
   - Update README.md if adding new features
   - Document benchmark results in BENCHMARK_MANIFEST.md
   - Keep HEARTBEAT.md updated with latest test results

### Common Issues

**Problem:** Import errors when running scripts
**Solution:** Ensure you're running from the repository root directory and the correct virtual environment is activated

**Problem:** MPI errors
**Solution:** Verify MPI is installed: `mpirun --version`

**Problem:** Slow performance with A2DMRG
**Solution:** Install `optuna` in the a2dmrg virtual environment for cotengra optimization:
```bash
cd a2dmrg
source venv/bin/activate
pip install optuna
deactivate
```

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests to verify correctness
5. Commit with descriptive messages
6. Push to your fork
7. Submit a pull request

### Questions?

Open an issue on GitHub or contact the maintainers.
