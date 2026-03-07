# CPU-AUDIT BRANCH COMPREHENSIVE AUDIT REPORT
**Generated**: 2026-03-07  
**Branch**: cpu-audit  
**Commit**: 5dd0e28

---

## CRITICAL ISSUES (Require Immediate Attention)

### 🔴 CRITICAL #1: Boundary Optimization Permanently Disabled
**Location**: `pdmrg/pdmrg/dmrg.py:835`, `pdmrg2/pdmrg/dmrg.py:839`

```python
# Skip optimization due to spurious H_eff eigenvalues (TODO: fix H_eff bug)
skip_opt = True  # Always skip until H_eff bug is fixed
```

**Impact**: The boundary merge between ranks NEVER optimizes the shared bond tensor. This is the key step that allows independently evolved wavefunctions to join properly. Without it, multi-rank PDMRG/PDMRG2 can only measure energy, not improve it during merges.

**Evidence**: Metadata shows `"skip_opt": True` for all `np>1` tests.

**Recommendation**: Fix the H_eff construction bug or remove boundary optimization entirely and document the limitation.

---

### 🔴 CRITICAL #2: V Matrix Shape Mismatch Bug
**Location**: `pdmrg/pdmrg/dmrg.py:704-707`

**Problem**: V matrices initialized as 2D identity:
```python
V_right = np.eye(chi_R, dtype=...)  # shape (chi_R, chi_R)
```

But used as 1D vectors in merge:
```python
V_psi_right = V[:, None, None] * psi_right  # expects shape (chi,)
```

**Why It Doesn't Crash**: `recompute_boundary_v` overwrites V to 1D before first merge. But this is fragile—changing call order would cause silent shape corruption or broadcast errors.

**Recommendation**: Initialize V as 1D vectors from the start or add explicit shape assertions.

---

### 🔴 CRITICAL #3: A2DMRG CLI Completely Non-Functional
**Location**: `a2dmrg/a2dmrg/dmrg.py:687-756`

The entire `main()` function body is commented out. All model paths lead to either:
- Bose-Hubbard: `NotImplementedError`
- Josephson: Silently uses Heisenberg physics instead
- Code reaches `sys.exit(1)` without running

**Impact**: `python -m a2dmrg` does nothing useful.

**Recommendation**: Either implement the CLI properly or remove it and document that A2DMRG is library-only.

---

### 🔴 CRITICAL #4: i-Orthogonal Transformation Not Implemented
**Location**: `a2dmrg/a2dmrg/numerics/local_microstep.py:70,253`

```python
# TODO: Implement proper i-orthogonal transformation without bond compression
mps_updated = mps.copy()
```

**Impact**: The core theoretical requirement of A2DMRG (Grigori & Hassan Section 2) — that local micro-steps operate on i-orthogonally gauged tensors — is completely skipped. The algorithm copies the MPS and proceeds without the required gauge transformation.

**Recommendation**: This explains A2DMRG's poor performance. The algorithm is not actually A2DMRG without i-orthogonal form.

---

## HIGH SEVERITY ISSUES

### 🟠 HIGH #1: Hardcoded OpenMPI Path (Non-Portable)
**Location**: `benchmarks/correctness_suite.py:136,245`

```python
'/usr/lib64/openmpi/bin/mpirun'
```

**Impact**: Only works on Fedora/RHEL. Debian/Ubuntu use `/usr/bin/mpirun`. Fails silently on other distributions.

**Recommendation**: Use `shutil.which('mpirun')` or make path configurable.

---

### 🟠 HIGH #2: Josephson Model Silently Uses Wrong Physics
**Location**: `a2dmrg/a2dmrg/__main__.py:176-186`

```python
elif args.model == 'josephson':
    # For now, use Heisenberg as placeholder
    # TODO: Implement full Josephson model
    from quimb.tensor import SpinHam1D
    builder = SpinHam1D(S=1/2)
```

**Impact**: Users requesting Josephson get Heisenberg results with no error. Silent correctness failure.

**Recommendation**: Raise `NotImplementedError` with clear message.

---

### 🟠 HIGH #3: Parallel Warmup Ignores Its Own Parameter
**Location**: `pdmrg/pdmrg/dmrg.py:67-143`

```python
def parallel_warmup(..., n_warmup_sweeps=3, ...):
```

`n_warmup_sweeps` is accepted but never used. Function only does one QR canonization.

**Impact**: Misleading API. Users setting `n_warmup_sweeps=10` get same result as `n_warmup_sweeps=1`.

**Recommendation**: Either implement sweeps or remove parameter and update docstring.

---

## MEDIUM SEVERITY ISSUES

### 🟡 MEDIUM #1: Hardcoded `/tmp/` for Benchmark Scripts
**Location**: `benchmarks/correctness_suite.py:128,238`

**Problems**:
- Scripts never cleaned up
- Race condition if two benchmark runs overlap
- Fails in restricted containers

**Recommendation**: Use `tempfile.mkdtemp()` with cleanup.

---

### 🟡 MEDIUM #2: CLI Default Tolerance Differs Between pdmrg/pdmrg2
**Location**: `pdmrg/dmrg.py:952` vs `pdmrg2/dmrg.py:955`

- pdmrg CLI: `--tol` default = `1e-10`
- pdmrg2 CLI: `--tol` default = `1e-8`

**Impact**: Benchmarks using CLI defaults compare different convergence criteria.

**Recommendation**: Unify to `1e-8` or make the difference explicit in documentation.

---

### 🟡 MEDIUM #3: rSVD Uses Fixed Seed on Every Call
**Location**: `pdmrg2/pdmrg/numerics/linalg_utils.py:221`

```python
rng = np.random.default_rng(seed=0)
```

**Impact**: Every call uses identical random sketch, reducing randomness benefit and potentially causing systematic bias.

**Recommendation**: Create RNG once per module or accept seed as parameter.

---

### 🟡 MEDIUM #4: Zero-Energy Sentinel Conflicts with Valid Energies
**Location**: `pdmrg/pdmrg/parallel/communication.py:75`

```python
merge_energies = [e for e in all_E if e != 0.0]
```

**Impact**: If true ground state energy is ≈0 (frustrated systems), valid estimates are filtered out.

**Recommendation**: Use `None` or `np.nan` as sentinel.

---

### 🟡 MEDIUM #5: A2DMRG np=1 Missing from Test Suite
**Location**: `benchmarks/correctness_suite.py:396`

```python
for idx, np_count in enumerate([2, 4, 8], start=11):
```

**Impact**: The most reliable A2DMRG code path (np=1 early return) is never tested.

**Recommendation**: Add `np=1` to test matrix or document why it's excluded.

---

### 🟡 MEDIUM #6: `rebuild_boundary_r_env` Skips Update for `n_local==1`
**Location**: `pdmrg/pdmrg/dmrg.py:436-438`

**Impact**: For small L with many ranks, right environment becomes stale, causing incorrect energies.

**Recommendation**: Update environment even for single-site blocks.

---

## LOW SEVERITY ISSUES (Polish/Documentation)

### ⚪ LOW #1: Test Count Label Is Wrong
`[N/14]` but only 13 tests exist (correctness_suite.py:340-397)

### ⚪ LOW #2: Duplicate Parameters Section
pdmrg_main docstring has two `Parameters ----------` headers (pdmrg/dmrg.py:587-600)

### ⚪ LOW #3: Dead Code in `set_tensor_data`
Line marked `# NO` then immediately overwritten (mps/canonical.py:94)

### ⚪ LOW #4: Variable `L` Shadows Chain Length
Inside `rebuild_boundary_r_env` (pdmrg/dmrg.py:448)

### ⚪ LOW #5: Dead Imports in pdmrg2
`get_initial_direction`, `truncated_svd`, `compute_v_from_svd` imported but unused (pdmrg2/dmrg.py:27,32)

### ⚪ LOW #6: Stray `=4` File
Zero-byte file in repository root (likely bash typo)

### ⚪ LOW #7: `np` as Loop Variable
Shadows numpy convention (hardware_config.py:94)

---

## IMPLEMENTATION DEVIATIONS FROM REFERENCE

### pdmrg2 vs pdmrg (Intentional GPU Optimizations)

| Component | pdmrg | pdmrg2 |
|-----------|-------|--------|
| **Eigensolver** | Lanczos (scipy `eigsh`) | Block-Davidson |
| **Canonicalization** | QR | Newton-Schulz polar |
| **Interior SVD** | Full SVD (`truncated_svd`) | Randomized SVD (`rsvd_cholesky`) |
| **Boundary SVD** | Full SVD (`accurate_svd`) | Full SVD (`accurate_svd`) ✓ Same |
| **CLI `--tol` default** | `1e-10` | `1e-8` |
| **`final_sweep_num` init** | `max_sweeps` | `0` |

**Notes**:
- `block_size=4` hardcoded, no way to configure
- Newton-Schulz falls back to QR for wide matrices
- rSVD not used for boundary merges (correct)
- Both implementations share `skip_opt = True` bug

### A2DMRG Deviations from refs/a2dmrg

**Cannot verify** — `refs/a2dmrg` branch not accessible in current session. Key observations:
- i-orthogonal transformation stubbed out (core requirement missing)
- Josephson model placeholder (uses Heisenberg)
- CLI non-functional
- Parallel scaling tests all `None`

---

## VERIFICATION STATUS BY COMPONENT

| Component | pdmrg | pdmrg2 | a2dmrg | Status |
|-----------|-------|--------|--------|--------|
| **Core algorithm** | ⚠️ | ⚠️ | ❌ | skip_opt bug (pdmrg/2), i-orthogonal missing (a2dmrg) |
| **Parallel communication** | ✓ | ✓ | ✓ | Correct (but 0.0 sentinel fragile) |
| **MPS operations** | ⚠️ | ⚠️ | ❌ | V shape bug (pdmrg/2), gauge stub (a2dmrg) |
| **Benchmarking** | ⚠️ | ⚠️ | ⚠️ | Hardcoded paths, np=1 missing |
| **CLI** | ✓ | ✓ | ❌ | pdmrg/2 work, a2dmrg broken |
| **Tests** | ⚠️ | ⚠️ | ❌ | Correctness tests work, scaling tests stubbed |

**Legend**: ✓ Good | ⚠️ Issues but functional | ❌ Broken/incomplete

---

## RECOMMENDATIONS BY PRIORITY

### Must Fix (Blocks Production Use)
1. ✅ **Fix or document `skip_opt` bug** — Boundary optimization disabled
2. ✅ **Fix V matrix initialization** — Shape mismatch latent bug
3. ✅ **Make benchmarks portable** — Remove hardcoded MPI path
4. ✅ **Implement or remove A2DMRG i-orthogonal** — Core algorithm incomplete

### Should Fix (Quality/Correctness)
5. ⚠️ Fix `parallel_warmup` to actually use `n_warmup_sweeps` or remove parameter
6. ⚠️ Change 0.0 sentinel to `None`/`np.nan` in `check_convergence`
7. ⚠️ Unify CLI `--tol` defaults between pdmrg/pdmrg2
8. ⚠️ Fix rSVD seed to be truly random per-call
9. ⚠️ Add A2DMRG np=1 to test suite

### Nice to Have (Polish)
10. 📝 Clean up docstring duplication
11. 📝 Remove dead code (`# NO` lines, unused imports)
12. 📝 Fix test count labels (13 not 14)
13. 📝 Remove stray `=4` file

---

## PERFORMANCE IMPACT SUMMARY

Based on correctness suite results:

**PDMRG/PDMRG2 Accuracy**: ✅ Good (1e-11 to 1e-14 on Heisenberg)
- Validated up to L=48 with acceptance threshold
- 12-15× faster than quimb on complex systems (Josephson)
- Weak parallel scaling (10-30% speedup at np=8)

**A2DMRG Performance**: ❌ Poor
- Negative scaling (gets slower with more processes)
- 5-7e-09 errors on L=48 (fails acceptance)
- Likely due to missing i-orthogonal transformation

**Root Cause**: The `skip_opt` bug limits PDMRG/PDMRG2 parallel efficiency. The missing i-orthogonal transformation breaks A2DMRG's theoretical foundation.

---

## FILES REQUIRING ATTENTION

### Critical Path
- `pdmrg/pdmrg/dmrg.py` — skip_opt, V shape, parallel_warmup
- `pdmrg2/pdmrg/dmrg.py` — same issues as pdmrg
- `a2dmrg/a2dmrg/numerics/local_microstep.py` — i-orthogonal stub
- `a2dmrg/a2dmrg/dmrg.py` — CLI commented out
- `benchmarks/correctness_suite.py` — hardcoded paths

### Reference Comparison Needed
- Compare against `refs/pdmrg` branch for pdmrg/pdmrg2
- Compare against `refs/a2dmrg` branch for a2dmrg

---

## NEXT STEPS

1. **Immediate**: Fix benchmark portability (hardcoded mpirun path)
2. **Short-term**: Address V matrix shape bug and skip_opt issue
3. **Medium-term**: Implement A2DMRG i-orthogonal or deprecate A2DMRG
4. **Long-term**: Comprehensive comparison with reference branches

