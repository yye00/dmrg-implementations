# A2DMRG Task Orientation

## Current State (as of 2026-02-16)
- 58/73 features passing
- 15 features pending (mostly require MPI parallel testing)
- MPI compatibility layer added: can run serial without MPI installed
- Serial validation passes: L=10, L=20 match quimb DMRG within 1e-8

## Environment
- **No MPI installed** - parallel tests (np>1) cannot run
- Use `a2dmrg.mpi_compat` instead of `mpi4py` for imports
- Serial tests work with fake MPI communicator

## Remaining Features (15 total)

### Require MPI (cannot run currently):
1. A2DMRG np=2,4,8 matches serial
2. Complex128 parallel np=2,4
3. Parallel speedup verification
4. Reproducibility across np
5. Full workflow parallel tests

### Can complete without MPI:
1. **Heisenberg L=40 serial** - add test, verify E/L ≈ -0.443
2. **Scalability report framework** - create structure, fill serial data
3. **Memory usage tracking** - add memory profiling
4. **Documentation** - README, docstrings, examples

## Priority Tasks

### Task 1: L=40 Serial Validation
Add to `a2dmrg/tests/test_validation.py`:
```python
def test_heisenberg_serial_l40():
    L, bond_dim = 40, 50
    E_quimb = _run_quimb_dmrg(L, bond_dim)
    E_a2dmrg, _ = _run_a2dmrg(L, bond_dim, warmup_sweeps=2)
    # Verify E/L ≈ -0.443 (infinite chain limit: -ln(2) + 1/4)
    assert abs(E_a2dmrg/L - (-0.443)) < 0.01
    assert abs(E_a2dmrg - E_quimb) < 1e-8
```

### Task 2: Scalability Report Structure
Create `a2dmrg/tests/test_scaling.py`:
- Framework for timing tests
- JSON output format
- Placeholder for parallel data (to fill when MPI available)

### Task 3: Update feature_list.json
Mark completed features as `passes: true`

## Code Locations
- Main algorithm: `a2dmrg/dmrg.py`
- MPI compat: `a2dmrg/mpi_compat.py`
- Tests: `a2dmrg/tests/`
- Features: `feature_list.json`

## Running Tests
```bash
# All serial tests
python -m pytest a2dmrg/tests/test_validation.py -v

# Quick check
python -c "from a2dmrg.dmrg import a2dmrg_main; print('OK')"
```

## Success Criteria
- L=40 serial test passes
- Scalability framework ready
- feature_list.json updated
- Clean commit with updated README
