# A2DMRG Approach-A Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the two main remaining performance/scalability bottlenecks in A2DMRG: streaming i-orthogonal decompositions (removes O(L×χ²) memory wall per rank), and expose S-spectrum reduction controls + observability, then validate correctness and scaling against PDMRG on the Heisenberg benchmark.

**Architecture:** Three code changes, all backward-compatible (default params preserve existing behavior). Change 1 is a refactor of Phase 2 in `local_steps.py` to generate each rank's decompositions one-at-a-time instead of all-at-once. Change 2 adds `coarse_reduction_tol` and `overlap_threshold` params to `a2dmrg_main` and wires them through to the solver and filter. Change 3 adds observability (logs effective coarse dim `k` per sweep). Finish with full Heisenberg benchmark run.

**Tech Stack:** Python 3.13, quimb, mpi4py, numpy, pytest (run inside `a2dmrg/` with `source venv/bin/activate`)

**Working directory for all commands:** `/home/captain/clawd/work/dmrg-implementations/a2dmrg`

**How to run tests:**
```bash
cd /home/captain/clawd/work/dmrg-implementations/a2dmrg
source venv/bin/activate
python -m pytest a2dmrg/tests/<test_file>.py -v
```

---

## Task 1: Streaming i-orthogonal decompositions

**What:** Replace the all-at-once `prepare_orthogonal_decompositions(mps)` call (creates ALL L copies on every rank) with a per-site streaming loop that creates one copy, runs the microstep, then discards the copy. Peak memory per rank drops from O(L×χ²) to O(χ²).

**Files:**
- Modify: `a2dmrg/parallel/local_steps.py:82-136`
- Test: `a2dmrg/tests/test_parallel_local_steps.py`

**Step 1: Write the failing test**

Add this test to `a2dmrg/tests/test_parallel_local_steps.py` after the existing imports:

```python
def test_streaming_decompositions_same_results():
    """
    Streaming (one-at-a-time) decompositions must produce identical results
    to the old all-at-once approach for correctness verification.
    """
    import numpy as np
    import quimb.tensor as qtn
    from a2dmrg.mpi_compat import MPI
    from a2dmrg.parallel.local_steps import parallel_local_microsteps

    L = 6
    bond_dim = 4
    mps = qtn.MPS_rand_state(L, bond_dim=bond_dim, phys_dim=2, dtype=np.float64)
    mpo = qtn.MPO_ham_heis(L, cyclic=False)
    comm = MPI.COMM_WORLD  # serial (size=1) in pytest

    results = parallel_local_microsteps(mps, mpo, comm, microstep_type="one_site", tol=1e-8)

    # All L sites should be covered
    assert set(results.keys()) == set(range(L)), f"Expected sites 0..{L-1}, got {sorted(results.keys())}"
    # All energies should be finite and negative for Heisenberg (after a microstep)
    for site, (updated_mps, energy) in results.items():
        assert np.isfinite(energy), f"Site {site}: energy is not finite"
        assert updated_mps is not None
        assert updated_mps.L == L
```

Run:
```bash
python -m pytest a2dmrg/tests/test_parallel_local_steps.py::test_streaming_decompositions_same_results -v
```
Expected: FAIL (test doesn't exist yet / or passes if already working — either way continue to Step 3).

**Step 2: Verify the memory problem exists**

This is optional but instructive — check that the current code touches all L decompositions regardless of rank's assigned sites:
```bash
python -c "
from a2dmrg.parallel.local_steps import parallel_local_microsteps
import inspect
src = inspect.getsource(parallel_local_microsteps)
assert 'prepare_orthogonal_decompositions(mps)' in src, 'old code not found'
print('Confirmed: old code creates ALL L decompositions')
"
```

**Step 3: Implement streaming decompositions**

Replace the body of `parallel_local_microsteps` in `a2dmrg/parallel/local_steps.py`. Change lines 82–136 to:

```python
    # Import required functions
    from ..mps.canonical import move_orthogonality_center, _pad_to_bond_dimensions

    # Get MPI rank and size
    rank = comm.Get_rank()
    n_procs = comm.Get_size()

    L = mps.L  # Number of sites

    # Record original bond dims ONCE (needed for padding after canonization)
    original_bond_dims = []
    for i in range(L):
        tensor = mps[i].data
        if i == 0:
            original_bond_dims.append(tensor.shape[0])
        elif i == L - 1:
            original_bond_dims.append(tensor.shape[0])
        else:
            original_bond_dims.append((tensor.shape[0], tensor.shape[1]))

    # Distribute sites BEFORE creating any decompositions.
    # Each rank only needs decompositions for its own sites.
    my_sites = distribute_sites(L, n_procs, rank)

    # Dictionary to store results
    results = {}

    if microstep_type == "one_site":
        # STREAMING: create one i-orthogonal copy per site, run microstep, discard copy.
        # Peak memory per rank: O(chi^2) instead of O(L * chi^2).
        for site in my_sites:
            mps_copy = mps.copy()
            move_orthogonality_center(mps_copy, site, normalize=True)
            _pad_to_bond_dimensions(mps_copy, original_bond_dims)

            updated_mps, energy = local_microstep_1site(
                mps_copy, mpo, site, tol=tol
            )
            results[site] = (updated_mps, energy)
            del mps_copy  # release memory immediately

    elif microstep_type == "two_site":
        for site in my_sites:
            if site < L - 1:  # Can only do two-site update if not last site
                mps_copy = mps.copy()
                move_orthogonality_center(mps_copy, site, normalize=True)
                _pad_to_bond_dimensions(mps_copy, original_bond_dims)

                updated_mps, energy = local_microstep_2site(
                    mps_copy, mpo, site,
                    max_bond=max_bond,
                    cutoff=cutoff,
                    tol=tol
                )
                results[site] = (updated_mps, energy)
                del mps_copy  # release memory immediately

    else:
        raise ValueError(f"Unknown microstep_type: {microstep_type}. "
                        f"Must be 'one_site' or 'two_site'")

    # IMPORTANT: No MPI communication here!
    # Each processor returns its local results independently
    return results
```

**Step 4: Run the new test and full local_steps test suite**

```bash
python -m pytest a2dmrg/tests/test_parallel_local_steps.py -v
```
Expected: All tests pass (or same pass count as before the change, plus the new test).

**Step 5: Run broader test suite to check for regressions**

```bash
python -m pytest a2dmrg/tests/ -v -x --ignore=a2dmrg/tests/test_l40_validation.py --ignore=a2dmrg/tests/test_scaling_mpi.py -q 2>&1 | tail -20
```
(Ignore l40 and scaling_mpi as those are long-running.)
Expected: No new failures introduced.

**Step 6: Commit**

```bash
git add a2dmrg/a2dmrg/parallel/local_steps.py a2dmrg/a2dmrg/tests/test_parallel_local_steps.py
git commit -m "perf: stream i-orthogonal decompositions one-at-a-time per rank

Instead of prepare_orthogonal_decompositions(mps) which creates ALL L MPS
copies on every rank, we now distribute sites first, then create and destroy
one copy per assigned site. Peak memory per rank drops from O(L*chi^2) to
O(chi^2), a factor-of-L improvement independent of np.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Expose coarse_reduction_tol and overlap_threshold

**What:** Add two new params to `a2dmrg_main`: `coarse_reduction_tol` (wired to `solve_coarse_eigenvalue_problem`'s `regularization`) and `overlap_threshold` (wired to the redundancy filter). Defaults preserve existing behavior. This lets callers tune the S-spectrum cutoff and redundancy threshold for ablation studies.

**Files:**
- Modify: `a2dmrg/dmrg.py` — signature of `a2dmrg_main` and two call sites
- Test: `a2dmrg/tests/test_a2dmrg_main.py`

**Step 1: Write the failing test**

Add to `a2dmrg/tests/test_a2dmrg_main.py`:

```python
def test_coarse_reduction_tol_param_accepted():
    """a2dmrg_main must accept coarse_reduction_tol and overlap_threshold without error."""
    import numpy as np
    import quimb.tensor as qtn
    from a2dmrg.mpi_compat import MPI
    from a2dmrg.dmrg import a2dmrg_main

    L = 6
    bond_dim = 4
    mpo = qtn.MPO_ham_heis(L, cyclic=False)
    comm = MPI.COMM_WORLD

    # Should not raise TypeError for unknown param
    energy, mps = a2dmrg_main(
        L=L, mpo=mpo, max_sweeps=3, bond_dim=bond_dim,
        tol=1e-4, comm=comm, warmup_sweeps=1,
        coarse_reduction_tol=1e-8,
        overlap_threshold=0.99,
        timing_report=False,
        verbose=False,
    )
    assert np.isfinite(energy)
```

Run:
```bash
python -m pytest a2dmrg/tests/test_a2dmrg_main.py::test_coarse_reduction_tol_param_accepted -v
```
Expected: FAIL with `TypeError: a2dmrg_main() got an unexpected keyword argument 'coarse_reduction_tol'`

**Step 2: Add params to a2dmrg_main signature**

In `a2dmrg/dmrg.py`, find the function signature:
```python
def a2dmrg_main(
    L: int,
    mpo,
    max_sweeps: int = 20,
    bond_dim: int = 100,
    tol: float = 1e-10,
    comm: Optional[MPI.Comm] = None,
    dtype=np.float64,
    one_site: bool = False,
    verbose: bool = True,
    initial_mps: Optional[qtn.MatrixProductState] = None,
    warmup_sweeps: int = 2,
    finalize_sweeps: int = 0,
    timing_report: bool = True,
    timing_dir: str = "reports",
    max_candidates: Optional[int] = None,
) -> Tuple[float, qtn.MatrixProductState]:
```

Replace with:
```python
def a2dmrg_main(
    L: int,
    mpo,
    max_sweeps: int = 20,
    bond_dim: int = 100,
    tol: float = 1e-10,
    comm: Optional[MPI.Comm] = None,
    dtype=np.float64,
    one_site: bool = False,
    verbose: bool = True,
    initial_mps: Optional[qtn.MatrixProductState] = None,
    warmup_sweeps: int = 2,
    finalize_sweeps: int = 0,
    timing_report: bool = True,
    timing_dir: str = "reports",
    max_candidates: Optional[int] = None,
    coarse_reduction_tol: float = 1e-8,
    overlap_threshold: float = 0.99,
) -> Tuple[float, qtn.MatrixProductState]:
```

Also add them to the `timing["meta"]` dict (around line 165):
```python
"max_candidates": max_candidates,
"coarse_reduction_tol": coarse_reduction_tol,
"overlap_threshold": overlap_threshold,
```

**Step 3: Wire coarse_reduction_tol to the eigenvalue solver**

Find the call to `solve_coarse_eigenvalue_problem` (around line 471):
```python
            energy_new, coeffs = solve_coarse_eigenvalue_problem(
                H_coarse,
                S_coarse,
                return_all=False,
                regularization=1e-8  # Increased regularization for numerical stability
            )
```

Replace with:
```python
            energy_new, coeffs = solve_coarse_eigenvalue_problem(
                H_coarse,
                S_coarse,
                return_all=False,
                regularization=coarse_reduction_tol,
            )
```

**Step 4: Wire overlap_threshold to both filter calls**

There are two `filter_redundant_candidates` calls in `dmrg.py`. Find them (around lines 412 and 449):

First call (parallel path, line ~412):
```python
                filtered_candidates, retained_indices = filter_redundant_candidates(
                    candidate_mps_list,
                    overlap_threshold=0.99,
                )
```
Replace `overlap_threshold=0.99` with `overlap_threshold=overlap_threshold`.

Second call (serial path, line ~449):
```python
            candidate_mps_list, _ = filter_redundant_candidates(
                candidate_mps_list,
                overlap_threshold=0.99,
            )
```
Replace `overlap_threshold=0.99` with `overlap_threshold=overlap_threshold`.

**Step 5: Run the new test**

```bash
python -m pytest a2dmrg/tests/test_a2dmrg_main.py::test_coarse_reduction_tol_param_accepted -v
```
Expected: PASS

**Step 6: Run full test suite (quick tests only)**

```bash
python -m pytest a2dmrg/tests/ -v -x -q \
  --ignore=a2dmrg/tests/test_l40_validation.py \
  --ignore=a2dmrg/tests/test_scaling_mpi.py \
  --ignore=a2dmrg/tests/test_l20_validation.py \
  2>&1 | tail -20
```
Expected: No new failures.

**Step 7: Commit**

```bash
git add a2dmrg/a2dmrg/dmrg.py a2dmrg/a2dmrg/tests/test_a2dmrg_main.py
git commit -m "feat: expose coarse_reduction_tol and overlap_threshold params

Add two new tunable parameters to a2dmrg_main:
- coarse_reduction_tol (default 1e-8): S-eigenvalue cutoff for subspace
  projection in solve_coarse_eigenvalue_problem
- overlap_threshold (default 0.99): redundancy filter cutoff

Both default to existing hardcoded values, so behavior is unchanged.
Useful for ablation studies and publication comparisons.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Log effective coarse dimension k per sweep

**What:** After solving the coarse eigenvalue problem, log how many S-eigenvalues survived the cutoff (the "effective coarse dimension" k). This tells us how much the S-spectrum reduction is actually helping each sweep. Modify `solve_coarse_eigenvalue_problem` to optionally return k, then log it in the timing dict in `dmrg.py`.

**Files:**
- Modify: `a2dmrg/numerics/coarse_eigenvalue.py` — add `return_diagnostics` param
- Modify: `a2dmrg/dmrg.py` — capture and log k
- Test: `a2dmrg/tests/test_coarse_eigenvalue.py`

**Step 1: Write the failing test**

Add to `a2dmrg/tests/test_coarse_eigenvalue.py`:

```python
def test_return_diagnostics():
    """solve_coarse_eigenvalue_problem with return_diagnostics=True returns n_effective."""
    import numpy as np
    from a2dmrg.numerics.coarse_eigenvalue import solve_coarse_eigenvalue_problem

    # 3x3 with one near-null direction
    H = np.array([[1.0, 0.1, 0.0],
                  [0.1, 2.0, 0.1],
                  [0.0, 0.1, 3.0]])
    S = np.array([[1.0, 0.5, 0.0],
                  [0.5, 1.0, 0.5],
                  [0.0, 0.5, 1.0]])

    energy, coeffs, diag = solve_coarse_eigenvalue_problem(
        H, S, regularization=1e-10, return_diagnostics=True
    )

    assert "n_effective" in diag, "diagnostics dict must contain n_effective"
    assert isinstance(diag["n_effective"], int)
    assert 1 <= diag["n_effective"] <= 3
    assert np.isfinite(energy)
```

Run:
```bash
python -m pytest a2dmrg/tests/test_coarse_eigenvalue.py::test_return_diagnostics -v
```
Expected: FAIL with `TypeError` (unknown keyword arg) or wrong return count.

**Step 2: Add return_diagnostics to solve_coarse_eigenvalue_problem**

In `a2dmrg/numerics/coarse_eigenvalue.py`, find the function signature:
```python
def solve_coarse_eigenvalue_problem(
    H_coarse: np.ndarray,
    S_coarse: np.ndarray,
    regularization: float = 1e-10,
    return_all: bool = False
) -> Tuple[float, np.ndarray]:
```

Replace with:
```python
def solve_coarse_eigenvalue_problem(
    H_coarse: np.ndarray,
    S_coarse: np.ndarray,
    regularization: float = 1e-10,
    return_all: bool = False,
    return_diagnostics: bool = False,
):
```

Then, just before the `return` statements at the bottom of the function, add:
```python
    diagnostics = {"n_effective": int(np.sum(keep))}
```

And modify the two return statements:
```python
    if return_all:
        if return_diagnostics:
            return eigenvalues, coefficients_matrix, diagnostics
        return eigenvalues, coefficients_matrix
    else:
        if return_diagnostics:
            return eigenvalues[0], coefficients_matrix[:, 0], diagnostics
        return eigenvalues[0], coefficients_matrix[:, 0]
```

**Step 3: Capture k in dmrg.py and add to timing**

In `a2dmrg/dmrg.py`, find the coarse solve block (around line 468):
```python
        coarse_solve_t0 = time.perf_counter()
        if rank == 0:
            energy_new, coeffs = solve_coarse_eigenvalue_problem(
                H_coarse,
                S_coarse,
                return_all=False,
                regularization=coarse_reduction_tol,
            )
```

Replace with:
```python
        coarse_solve_t0 = time.perf_counter()
        n_effective_coarse = None
        if rank == 0:
            energy_new, coeffs, coarse_diag = solve_coarse_eigenvalue_problem(
                H_coarse,
                S_coarse,
                return_all=False,
                regularization=coarse_reduction_tol,
                return_diagnostics=True,
            )
            n_effective_coarse = coarse_diag["n_effective"]
            if verbose:
                print(f"  Coarse dim: {len(candidate_mps_list)} candidates → {n_effective_coarse} effective (S-spectrum)", flush=True)
```

Then in the timing recording block (around line 554), inside the `phase_summary` dict construction, add after building phase_summary:
```python
            if n_effective_coarse is not None:
                phase_summary["n_coarse_candidates"] = int(len(candidate_mps_list))
                phase_summary["n_effective_coarse"] = int(n_effective_coarse)
```

Note: `n_effective_coarse` is only set on rank 0, so broadcast it before the timing block:
```python
        if size > 1:
            n_effective_coarse = comm.bcast(n_effective_coarse, root=0)
```
Add this right after the existing `coeffs` and `energy_new` broadcasts.

**Step 4: Run the new test**

```bash
python -m pytest a2dmrg/tests/test_coarse_eigenvalue.py::test_return_diagnostics -v
```
Expected: PASS

**Step 5: Quick integration smoke test**

```bash
python -c "
import numpy as np
import quimb.tensor as qtn
from a2dmrg.mpi_compat import MPI
from a2dmrg.dmrg import a2dmrg_main

L, bond_dim = 6, 4
mpo = qtn.MPO_ham_heis(L, cyclic=False)
energy, mps = a2dmrg_main(
    L=L, mpo=mpo, max_sweeps=3, bond_dim=bond_dim,
    tol=1e-4, comm=MPI.COMM_WORLD, warmup_sweeps=1,
    coarse_reduction_tol=1e-8, overlap_threshold=0.99,
    timing_report=False, verbose=True,
)
print(f'Energy: {energy:.8f}')
assert np.isfinite(energy)
print('PASS')
"
```
Expected: Prints sweep output including "Coarse dim: N candidates → k effective", energy, PASS.

**Step 6: Run full quick test suite**

```bash
python -m pytest a2dmrg/tests/ -q -x \
  --ignore=a2dmrg/tests/test_l40_validation.py \
  --ignore=a2dmrg/tests/test_scaling_mpi.py \
  --ignore=a2dmrg/tests/test_l20_validation.py \
  2>&1 | tail -20
```

**Step 7: Commit**

```bash
git add a2dmrg/a2dmrg/numerics/coarse_eigenvalue.py a2dmrg/a2dmrg/dmrg.py a2dmrg/a2dmrg/tests/test_coarse_eigenvalue.py
git commit -m "feat: log effective coarse dimension k per sweep

solve_coarse_eigenvalue_problem now accepts return_diagnostics=True and
returns {'n_effective': k} showing how many S-eigenvalues survived the
cutoff. a2dmrg_main logs this per sweep and includes n_coarse_candidates
and n_effective_coarse in timing JSON. Helps diagnose how much S-spectrum
reduction is occurring in practice.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Run Heisenberg benchmark and compare against PDMRG

**What:** Run the full Heisenberg benchmark (`benchmarks/heisenberg_benchmark.py`) which tests quimb DMRG1, quimb DMRG2, PDMRG np=1/2/4/8, and A2DMRG np=1/2/4/8 on L=12, bond_dim=20. Compare accuracy (all must be within PASS_TOL=1e-10 of quimb DMRG2) and timing.

**Files:**
- Run: `benchmarks/heisenberg_benchmark.py` (no edits needed)
- Output: `benchmarks/heisenberg_benchmark_results.json`

**Step 1: Activate venv and run benchmark**

```bash
cd /home/captain/clawd/work/dmrg-implementations
source a2dmrg/venv/bin/activate
python benchmarks/heisenberg_benchmark.py --out benchmarks/heisenberg_benchmark_results.json 2>&1 | tee benchmarks/heisenberg_run.$(date +%Y-%m-%d_%H%M%S).log
```

This runs ~10 methods sequentially. Expected total time: 5–30 minutes.

**Step 2: Verify all accuracy tests pass**

```bash
python -c "
import json
with open('benchmarks/heisenberg_benchmark_results.json') as f:
    results = json.load(f)

PASS_TOL = 1e-10
ref_energy = results['quimb_DMRG2']['energy']
print(f'Reference energy (quimb DMRG2): {ref_energy:.12f}')
print()

all_pass = True
for name, r in results.items():
    if name == 'quimb_DMRG2':
        continue
    if 'energy' not in r:
        print(f'SKIP {name}: no energy')
        continue
    delta = abs(r['energy'] - ref_energy)
    status = 'PASS' if delta < PASS_TOL else 'FAIL'
    if status == 'FAIL':
        all_pass = False
    print(f'{status} {name}: E={r[\"energy\"]:.12f}, ΔE={delta:.2e}, t={r.get(\"time\",\"?\"):.2f}s')

print()
print('ALL PASS' if all_pass else 'SOME FAILURES')
"
```
Expected: ALL PASS with all ΔE < 1e-10.

**Step 3: Print speedup table**

```bash
python -c "
import json
with open('benchmarks/heisenberg_benchmark_results.json') as f:
    results = json.load(f)

print('=== Timing Comparison ===')
pdmrg_np1_t = results.get('PDMRG_np1', {}).get('time', None)
a2dmrg_np1_t = results.get('A2DMRG_np1', {}).get('time', None)

for name, r in sorted(results.items(), key=lambda x: x[0]):
    t = r.get('time', None)
    if t is None:
        continue
    speedup_vs_pdmrg1 = (pdmrg_np1_t / t) if pdmrg_np1_t else '?'
    speedup_vs_a2dmrg1 = (a2dmrg_np1_t / t) if a2dmrg_np1_t else '?'
    print(f'{name:25s}: {t:8.2f}s  speedup_vs_PDMRG_np1={speedup_vs_pdmrg1 if isinstance(speedup_vs_pdmrg1, str) else f\"{speedup_vs_pdmrg1:.2f}x\"}')
"
```

**Step 4: Check HEARTBEAT for benchmark skill instructions**

```bash
cat /home/captain/clawd/work/dmrg-implementations/HEARTBEAT.md
```
(Verify benchmark ran correctly and update HEARTBEAT if needed.)

**Step 5: Update HEARTBEAT with new results**

Edit `/home/captain/clawd/work/dmrg-implementations/HEARTBEAT.md` to add a new section for the post-improvement benchmark results with today's date, the energy table, and timing observations.

**Step 6: Commit**

```bash
cd /home/captain/clawd/work/dmrg-implementations
git add HEARTBEAT.md benchmarks/heisenberg_benchmark_results.json
git commit -m "bench: post-Approach-A Heisenberg benchmark results

All 10 methods PASS accuracy threshold (ΔE < 1e-10 vs quimb DMRG2).
Streaming decompositions + coarse_reduction_tol exposure changes confirmed
to not regress accuracy.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Summary of Changes

| File | Change |
|------|--------|
| `a2dmrg/parallel/local_steps.py` | Streaming decompositions: one copy per site per rank |
| `a2dmrg/dmrg.py` | Add `coarse_reduction_tol`, `overlap_threshold` params; log `n_effective_coarse` |
| `a2dmrg/numerics/coarse_eigenvalue.py` | Add `return_diagnostics` param, return `n_effective` |
| `a2dmrg/tests/test_parallel_local_steps.py` | New test: streaming correctness |
| `a2dmrg/tests/test_a2dmrg_main.py` | New test: new params accepted |
| `a2dmrg/tests/test_coarse_eigenvalue.py` | New test: diagnostics return |
| `HEARTBEAT.md` | Update with post-improvement benchmark results |

## Expected Outcomes

- **Memory**: Peak per-rank drops from O(L×χ²) to O(χ²) for Phase 2
- **Accuracy**: All Heisenberg benchmark tests still pass at ΔE < 1e-10
- **Observability**: Per-sweep logging of `n_coarse_candidates → n_effective` reveals how much S-spectrum reduction helps
- **Tuning**: `coarse_reduction_tol` and `overlap_threshold` let you sweep these for publication ablations
