# Cluster B: CheFSI retraction + rocprof re-measurement

Ground truth: docs/PATH_B_GROUND_TRUTH.md (locked at commit 6f45533)
Planner output SHA (source): a4a4b064d84d56a90.output
Date planned: 2026-04-23

---

I have full context. Now drafting the plan.

---

# DEFECT CLUSTER B — Remediation Plan

## 1. SVD Path Reconciliation Plan

**Current contradiction.** §2.6 (line 225) asserts SVD runs on CPU LAPACK uniformly, citing §5.6 profiling as justification. Reality (confirmed in `gpu-rocm/{dmrg-gpu,dmrg2-gpu,pdmrg-gpu}/src/*_impl.h`): primary `-gpu` variants call `Traits::rocsolver_gesvd_auto` on device; only `-opt` variants and `pdmrg-gpu`'s boundary `accurate_svd` use `lapack_gesvd`. §5.6 then claims a 97-98% CPU-LAPACK SVD fraction *for `dmrg2-gpu`* — the GPU-SVD variant — and immediately after (line 859) says GPU rocsolver gives "13% improvement" on the same problem. These two sentences cannot both describe the same code path.

**Rewrite strategy.** Restructure §2.6 to explicitly enumerate the two paths:
- `-gpu` (primary): GPU rocsolver `gesvd_auto`, with optional RSVD via `opts_.rsvd`.
- `-opt`: CPU LAPACK `dgesvd` host roundtrip, motivated by the empirical observation in §5.6 that GPU SVD is only marginally better at $\chi=256$ and worse at $\chi \leq 128$.
- `pdmrg-gpu`: GPU SVD interior, CPU `accurate_svd` at segment boundaries (correctness, not performance).

Move the "13% improvement" measurement out of §5.6 into a new §2.6 footnote or §5.6 sub-paragraph titled "SVD path choice rationale," and present it as the *justification* for the `-opt` design choice rather than a contradictory aside. Then §5.6 Table 13 must be relabeled to make clear which variant was profiled and which SVD path the 97-98% applies to. If the profile was actually run on a `-opt` variant (CPU SVD), the cell label must change from "dmrg2-gpu" to "dmrg2-gpu-opt." If it was run on `dmrg2-gpu` (GPU SVD), the row label "SVD (CPU LAPACK)" is wrong and must read "SVD (GPU rocsolver)."

## 2. rocprof Experiment Spec (MI300X)

**Goal.** Produce a defensible per-component wall-time breakdown for both SVD paths, on two problem sizes the paper actually defends.

**Matrix.**
- Variants: `dmrg2-gpu` (GPU SVD), `dmrg2-gpu-opt` (CPU SVD), `pdmrg-gpu` (GPU SVD interior), `pdmrg-gpu-opt` (CPU SVD).
- Models: Heisenberg L=64 χ=256; Josephson L=32 χ=256.
- Reps: N=5 per (variant, model) cell; report median and IQR.
- Sweeps: 3 measured sweeps after 1 untimed warm sweep; CLAUDE.md compliant `--warmup 1 --polish 0`.

**Instrumentation.** Use `omnitrace` (preferred over raw `rocprof` because it captures both HIP and host-side LAPACK ranges):

```
omnitrace-instrument -o ./inst.dmrg2-gpu -- ./build/dmrg2-gpu
omnitrace-run --use-roctracer ON --use-rocm-smi ON \
              --use-mpi OFF --time-output OFF \
              --collapse-threads ON --sampling-freq 1000 \
              -o run_${VARIANT}_${MODEL}_rep${R} -- \
              ./inst.dmrg2-gpu --L 64 --chi 256 --warmup 1 --polish 0 \
                               --sweeps 3 --model heisenberg
```

For CPU SVD path, also wrap `lapack_gesvd` call sites with manual `roctxRangePush/Pop` (add a `SVD_PROFILE` compile flag in `dmrg2_gpu_opt_impl.h` around lines 210 and 1148; `dmrg_gpu_opt_impl.h` lines 216 and 1113). This is the only way to attribute host-side SVD to a labeled bucket since rocprof sees only HIP traffic.

For GPU SVD path, wrap the `Traits::rocsolver_gesvd_auto` calls at `dmrg2_gpu_impl.h:1324`, `:1375` and `dmrg_gpu_impl.h:1263`, `:1324` with `roctxRangePush("SVD_GPU")`.

**Output.** `omnitrace-merge` → JSON with per-range wall fractions. Compute: SVD%, Lanczos%, env%, other%. Commit JSONs to `benchmarks/data/svd_profile/<UTC>/<variant>/`.

## 3. CheFSI Investigation

**Step 3a — git archaeology.** Already done: `git log --all --oneline --diff-filter=D --name-only | grep -iE "chefsi|chebyshev"` returns empty. `git log --all -S "CheFSI" / "Chebyshev" / "chefsi"` returns nothing. **No CheFSI source has ever been committed to this repository.**

**Step 3b — CPU tree.** `grep -rni "chefsi\|chebyshev" cpu/` (run on remote) — confirm absence. Inventory I-3 already confirmed empty.

**Step 3c — references.** Only `Zhou2006` is cited; no implementation citation (no SLEPc, no ChASE, no own package). The paper text describes implementation parameters ($p=15$, $n_\text{outer}=20$) without any "implementation in package X" sentence — implying it claims an in-house implementation that does not exist.

**Step 3d — Table 12 backing data.** I-3 confirms Table 12 (CheFSI L=20-50 results) has no findable backing JSON. Search `benchmarks/data/` for any file mentioning chefsi/chebyshev — expect empty.

**Decision tree.**
- If git archaeology surfaces a deleted branch/file → recover, re-run with proper provenance, retain §4.3 + §5.4 with corrected numbers.
- If no source ever existed (current evidence) → **retract**: delete §4.3 entirely (lines 295-309), delete §5.4 CheFSI subsection (around line 720+, the L=20-50 table), delete the conclusion bullet phrase "Chebyshev filtering is 1.9-11x slower" at line 948, and remove "Chebyshev filtering" from the BLAS-3 enumeration at line 880 and 946. Add a one-line acknowledgment in §4 that polynomial-filter methods were considered but no implementation was completed.

**Recommended path: retraction.** No reproducible evidence + algorithm/problem mismatch reasoning is sound on its own merits without numbers.

## 4. Concrete Text-Change Plan

- **§2.6 (line 225):** Replace single sentence with a paragraph distinguishing the two SVD paths per variant family, citing §5.6 only as justification for the `-opt` choice.
- **§5.6 Table 13 (lines 834-847):** Relabel to either "dmrg2-gpu-opt, Heisenberg L=64 χ=256, CPU LAPACK SVD path" (if actually run on -opt) or "dmrg2-gpu, … GPU rocsolver SVD path" with the row renamed to "SVD (GPU rocsolver)." Numbers must come from the §2 rocprof run, not the current unverified 38-41s. Add caption footnote: "Profile from omnitrace, N=5 reps, median ± IQR, run hash <commit>, JSON at benchmarks/data/svd_profile/<UTC>/."
- **§5.6 line 859 ("13% improvement"):** Move into a new sub-paragraph "SVD path choice" that explicitly compares CPU LAPACK vs GPU rocsolver at χ=256 and explains why -opt variants chose CPU. Cite both 124s and 142s with provenance.
- **§4.3 (lines 295-309) — DELETE** unless source recovered. Replace with one sentence in §4 omnibus: "Polynomial-filter methods (CheFSI \cite{Zhou2006}) were evaluated as candidates but not implemented in the present codebase; we do not report performance numbers."
- **§5.4 CheFSI subsection — DELETE** Table 12 and surrounding paragraphs.
- **§Conclusion (line 948):** Strike "Chebyshev filtering is 1.9-11x slower."
- **§Discussion line 880, 946:** Remove "Chebyshev filtering" from BLAS-3 enumeration.

## 5. Effort Estimate

| Task | Sandbox | MI300X |
|---|---|---|
| §2.6 + §5.6 + §4.3 + §5.4 LaTeX edits | 2-3 h | 0 |
| CheFSI git archaeology | 0.5 h | 0 |
| Add `roctxRange` instrumentation to 4 SVD call sites + recompile | 0.5 h (edit) | 0.5 h (build) |
| omnitrace runs: 4 variants × 2 models × 5 reps = 40 runs | 0 | ~6-8 h wall (Heisenberg L=64 χ=256 ~ 2 min/sweep × 3 sweeps × 40 = 4 h compute + setup) |
| JSON post-processing + Table 13 regeneration | 1 h | 0 |
| Verification re-render of paper | 0.5 h | 0 |
| **Total** | **~5 h** | **~7-9 h** |

## 6. Dependencies on Other Clusters

- **Cluster A (data provenance):** Table 13's new backing JSON must follow the same provenance schema being designed in Cluster A (commit hash, GPU vendor fix for `rocm-smi`, binary SHA pinning). Do this work *after* the provenance fix lands or the new SVD profile JSONs will inherit the same `vendor=unknown` defect.
- **Cluster C (CLAUDE.md compliance):** rocprof runs must use `--warmup 1 --polish 0`, identical to the compliance-fix Cluster C is enforcing on `pdmrg-gpu-opt` (currently hardcoded n_warmup=3, n_polish=10). If Cluster C unblocks CLI overrides on `pdmrg-gpu-opt` first, this saves a rebuild cycle.
- **Cluster D (refs.bib):** CheFSI retraction removes the only use of `Zhou2006`. If retraction is chosen, also drop `Zhou2006` from `refs.bib` to keep the uncited-references list shrinking rather than growing.
- **Cluster E (README/§3.1 baseline):** Independent — no shared edits.

The CheFSI retraction is the highest-value, lowest-risk action; the rocprof re-measurement is the only new MI300X work, and it cleanly justifies whatever language replaces the current §2.6/§5.6 contradiction.
