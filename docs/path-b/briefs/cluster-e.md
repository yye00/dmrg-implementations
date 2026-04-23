# Cluster E: quimb baseline tuning (graduated bonds + cotengra), drop false claims

Ground truth: docs/PATH_B_GROUND_TRUTH.md (locked at commit 6f45533)
Planner output SHA (source): a2df6e358ea3845e1.output
Date planned: 2026-04-23

---

I have the ground truth. Now I'll produce the planning document for Defect Cluster E. Note: I'm acknowledging the deferred tools and skills notices but neither are needed here — this is a pure planning/writing task.

---

# DEFECT CLUSTER E — Remediation Plan: quimb baseline + cotengra + thread-sweep + CPU spec + OpenBLAS

## 1. Quimb baseline tuning experiment

**Spec for new `quimb_runner_v2.py`:**

- **(a) Graduated bond schedule.** Replace `bond_dims=chi` (flat) with `bond_dims=[10, 20, 50, min(100,chi), chi, chi, chi, chi]`, truncated to `n_sweeps`. This is the idiomatic quimb pattern and avoids the cold-start chi=512 collapse that inflates the baseline wall-time.
- **(b) cotengra path optimization.** Wire it via `quimb.tensor.set_contract_path_optimizer`:
  ```python
  import cotengra as ctg
  opt = ctg.HyperOptimizer(methods=['greedy','kahypar'],
                           max_repeats=64, max_time=30, parallel='ray')
  qtn.contract_strategy(opt)  # global
  ```
  Cache paths to `~/.cache/cotengra/` keyed by (L, chi, model) so repeated cells don't re-optimize.
- **(c) Explicit cutoff_mode.** `cutoff_mode='rel'` with `cutoff=1e-10` (was implicit absolute 1e-14 — far stricter than C++ baseline uses, biases quimb slow).
- **(d) canonical_form policy.** Set `compress_opts=dict(method='svd', renorm=False)` and call `psi.canonize(orthog=...)` only at sweep boundaries, not per-bond.

**Pseudocode skeleton:**
```python
def run_quimb_tuned(L, chi, model, n_sweeps, threads):
    set_threads(threads)                        # OMP/MKL/OPENBLAS_NUM_THREADS
    opt = ctg.HyperOptimizer(...)
    H   = build_mpo(model, L)
    psi = qtn.MPS_rand_state(L, bond_dim=10, dtype='float64')
    sched = graduated_schedule(chi, n_sweeps)   # [10,20,50,chi,...]
    dmrg = qtn.DMRG2(H, bond_dims=sched,
                     cutoffs=1e-10, cutoff_mode='rel',
                     compress_opts={'method':'svd'})
    with qtn.contract_strategy(opt):
        E = dmrg.solve(tol=1e-9, verbosity=0)
    return {'E':E, 'wall_s':..., 'sched':sched, 'opt':'cotengra-hyper'}
```

**Re-bench scope.** Every CPU baseline cell in Tables 4, 5, 7, 10 plus the headline CPU-vs-GPU speedup figure — count is ~32 cells (8 (L,chi) points × 2 models × 1 thread setting, since thread sweep is a separate axis). MI300X host CPU time estimate: ~6 hr wall, dominated by chi=512 cells; cotengra path-search adds ~30 min one-shot (cached after).

## 2. Thread-sweep recovery

**Option A — reconstruct & rerun.** Add `--threads {1,2,4,8,12}` to runner, write `run_cpu_thread_sweep.py` (NOT in repo today), invoke via `OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T numactl --cpunodebind=0 --membind=0 python run_cpu_thread_sweep.py`. Bench 5 thread counts × 32 cells = 160 runs ≈ 18-22 hr CPU wall. **Pro:** restores the 93% claim if it survives; defensible. **Con:** the highlight may not reproduce on Xeon Platinum (different cache/NUMA than EPYC the paper imagined).

**Option B — drop the claim.** Delete the "93% of CPU wins use 1 thread" highlight and any §3.1 thread-sweep narrative. Replace with a single-sentence note: "CPU baseline pinned to 1 thread per process; multi-thread scaling left to future work." **Pro:** zero compute cost, immediate. **Con:** lose a rhetorical hook; reviewers may ask.

**Recommendation:** Option A is mandatory if highlight stays in abstract; Option B is acceptable if abstract is rewritten anyway (and it is — see §6).

## 3. CPU spec correction

Capture on each host:
```bash
lscpu > cpu_<host>.txt
numactl --hardware > numa_<host>.txt
cat /proc/cpuinfo | grep -E 'model name|cpu MHz' | head -2
```

§3.1 text edit:

> ~~The host CPU is an AMD EPYC processor.~~
> **The MI300X host CPU is an Intel Xeon Platinum 8470 (52 cores / 104 threads per socket, 2 sockets, 2 NUMA nodes, base 2.0 GHz). The H100 host CPU is an Intel Xeon Platinum 8480C (56 cores / 112 threads per socket, 2 sockets, 2 NUMA nodes, base 2.0 GHz). All CPU baselines pinned to NUMA node 0 via `numactl --cpunodebind=0 --membind=0`.**

Add a footnote disclosing prior incorrect "EPYC" attribution if this is a revision.

## 4. OpenBLAS truth

**Option A — actually build 0.3.28.**
```bash
git clone -b v0.3.28 https://github.com/OpenMathLib/OpenBLAS
make USE_OPENMP=1 NUM_THREADS=128 DYNAMIC_ARCH=1 -j
make install PREFIX=$HOME/openblas-0.3.28
LD_LIBRARY_PATH=$HOME/openblas-0.3.28/lib python -c "import numpy; numpy.show_config()"
```
Verify `numpy.show_config()` reports `openblas_info` from the custom path. Rerun all CPU cells. Effort: 2 hr build + 6 hr re-bench = 8 hr wall.

**Option B — drop the claim, report reality.**
```bash
python -c "import numpy; numpy.show_config()" > numpy_blas.txt
ldd $(python -c 'import numpy.linalg._umath_linalg; print(numpy.linalg._umath_linalg.__file__)')
```
Most likely shows pip-wheel `libopenblas` or `libscipy_openblas`. Report the exact lib + version. Effort: 15 min, no re-bench needed. **Recommended** unless §3.1 specifically depends on OpenBLAS version pinning for reproducibility.

## 5. cotengra reality

**Option A — delete the claim** from §3.1 and abstract. Effort: 5 min text edit, no re-bench. **Con:** loses a "we tried hard" signal.

**Option B — wire it (preferred, bundled with §1).** cotengra installation already done in option 1 above; no marginal cost beyond the §1 re-bench. **Pro:** the claim becomes true and likely improves the baseline (more honest comparison, reviewers trust paper more).

**Recommendation:** Option B, bundled with §1 — single re-bench covers both fixes.

## 6. Concrete text-change plan

**Abstract.** Strike "tuned with cotengra path optimization" and "across 1/2/4/8/12 thread sweeps" if §2 Option B chosen. Keep both if §1+§2-A executed.

**§3.1 (CPU baseline section).**
- Replace AMD EPYC sentence (see §3 above).
- Replace "OpenBLAS 0.3.28 source-built" with whichever §4 option chosen — either pin & describe build, or report shipped-with-numpy version with `numpy.show_config()` excerpt in appendix.
- Replace flat-bond schedule description with graduated-schedule description; cite cotengra usage if §1 executed.
- Replace "thread counts of 1, 2, 4, 8, 12" with either Option-A description ("scanned T ∈ {1,2,4,8,12}, NUMA-pinned") or Option-B disclaimer ("single-thread baseline; multi-thread left to future work").

**§5.1 (CPU-vs-GPU comparison).**
- Refresh all wall-time numbers from the new tuned-baseline JSON.
- Recompute every speedup ratio (likely lower — baseline got faster).
- Update the "93% single-thread" highlight or delete.

**Highlights box.** Drop or rewrite "93% of CPU wins use 1 thread" depending on §2 path.

## 7. Effort + dependencies

| Task | Wall hr | Depends on | Compute |
|---|---|---|---|
| §3 CPU spec capture | 0.25 | none | local + remote ssh |
| §4-B numpy BLAS audit | 0.25 | none | local |
| §4-A OpenBLAS source build | 8 | host root or `--prefix` install | MI300X + H100 hosts |
| §1 + §5-B quimb tuned runner | 6-8 | cotengra install | MI300X CPU |
| §2-A thread sweep | 18-22 | §1 (use tuned runner) | MI300X CPU |
| §6 text edits | 1 | §1, §2, §3, §4 results in hand | local |

**Critical path: §1 → §2-A → §6,** total ~28 hr if Option-A everywhere; **~9 hr** if §2-B and §4-B chosen. **Recommended path:** §1 + §2-B + §3 + §4-B + §5-B = ~9 hr total, drops two indefensible claims, keeps the tuned-baseline (which strengthens GPU-speedup credibility by raising the bar). Dependencies: cotengra pip install, ray (for parallel optimizer) optional, root not required for any path.
