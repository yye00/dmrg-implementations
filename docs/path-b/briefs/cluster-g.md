# Cluster G: provenance.py patches, clock-pin attempts on VF, timer-scope unification

Ground truth: docs/PATH_B_GROUND_TRUTH.md (locked at commit 6f45533)
Planner output SHA (source): a34ba27e11bbabf49.output
Date planned: 2026-04-23

---

I have enough to draft. Let me confirm the timer scope mismatch lines precisely.

The picture is clear:
- **Primary -gpu** variants (dmrg-gpu, dmrg2-gpu, pdmrg-gpu): `build_initial_environments();` THEN `auto t_start = ...` → env build EXCLUDED
- **-opt** variants (dmrg-gpu-opt, dmrg2-gpu-opt, pdmrg-gpu-opt): `auto t_start = ...` THEN `build_initial_environments();` → env build INCLUDED

---

# Defect Cluster G — Hardware Controls + Provenance Instrumentation

## 1. rocm-smi Pinning Protocol

VF (virtualized GPU) on the Hot Aisle VM cripples most setperf knobs, but the protocol below is the maximum we can extract and is the right harness shape regardless of bare-metal vs VM. Wrap around every benchmark rep; abort the rep if the post-state doesn't match the pre-state.

Pre-rep (run once per benchmark invocation, log to `provenance.gpu.pre`):
```
rocm-smi --showallinfo --json                    # full state
rocm-smi --showclocks --json                     # SCLK/MCLK
rocm-smi --showpids --json                       # CO-TENANT DETECTION (VM-critical)
rocm-smi --showtemp --showpower --showuse --json # baseline thermals
```
Attempt clock pin (will silently no-op on VF, log the attempted command and the rc):
```
rocm-smi --setperfdeterminism 1900               # MI300X peak SCLK
rocm-smi --setperflevel manual
rocm-smi --setpowercap 750                       # MI300X TDP
```
Post-rep (log to `provenance.gpu.post`): repeat `--showclocks --showtemp --showpower --showpids`. Compute `throttled = (post.SCLK_avg < 0.97 * pinned)` and `cotenant_changed = (pre.pids != post.pids)`. Both flags belong in the JSON.

## 2. provenance.py Patches

The "unknown" failure is `gpu_info()` requiring the literal substring `"GPU"` in `--showproductname --csv` output; on VF the header line is just `device,Card series,...` with no `GPU` token, so it falls through to NVIDIA, which fails, returning `{"vendor":"unknown"}`.

Patches to `/home/user/dmrg-implementations/benchmarks/lib/provenance.py`:

**Fix `gpu_info()` (lines 72–96):** drop the `"GPU" in rocm_out` guard; treat any non-empty rocm-smi stdout as AMD. Add `--showproductname --showmeminfo vram --showdriverversion --json` and parse JSON, not CSV. On parse failure record the raw stdout and `vendor="AMD"` rather than "unknown" — VF identification needs the raw bytes more than tidy fields.

**Fix `host_info()` (lines 60–69):** `platform.processor()` returns `x86_64` on Linux — useless for the EPYC/Xeon distinction. Add:
```python
"cpu_model":  _run(["sh","-c","lscpu | awk -F: '/Model name/ {print $2; exit}'"]).strip(),
"lscpu":      _run(["lscpu"]),
"numactl":    _run(["numactl","--hardware"]),
"meminfo":    _run(["sh","-c","grep -E 'MemTotal|HugePages' /proc/meminfo"]),
```

**New `rocm_versions()`:** capture `cat /opt/rocm/.info/version`, `dpkg -l | grep -E 'rocblas|rocsolver|hipblas|hipsolver'`, `hipconfig --version`, `ldd <binary> | grep -E 'rocblas|rocsolver'`. Embed under `provenance.rocm`.

**Fix `env_snapshot()` (line 122):** the `prefixes=("DMRG_GPU_",)` filter is why `provenance.env = {}` everywhere — none of our benchmark scripts set `DMRG_GPU_*` vars. Add the actually-used prefixes (`HIP_`, `HSA_`, `ROCM_`, `OMP_`, `OPENBLAS_`, `MKL_`, `BLIS_`, `GOMP_`, `KMP_`, `LD_`, `PATH`) and explicit keys `OMP_PLACES`, `OMP_PROC_BIND`, `OMP_NUM_THREADS`, `GOMP_CPU_AFFINITY`, `HIP_FORCE_DEV_KERNARG`, `HSA_ENABLE_SDMA`, `MIOPEN_FIND_MODE`. Add NUMA: shell out to `numactl --show`.

## 3. VM vs Bare-Metal

Ask Hot Aisle for bare-metal **only if** they offer it on a billable SKU we can afford for two weeks (rerun all six variants × full grid ≈ 40 GPU-hours). My read: unlikely to be granted free, and rerunning the entire campaign on a different host invalidates the cross-architecture comparisons we already have unless H100 also moves.

**Defensible alternative for the paper:** keep the VM, instrument it honestly. Add §3.2 paragraph: "Benchmarks ran on a Hot Aisle MI300X virtual function (VM `enc1-gpuvm002`); SR-IOV partitioning prevents `rocm-smi --setperfdeterminism` from taking effect, and clock state is governed by the host hypervisor. We mitigate via (i) `rocm-smi --showpids` co-tenant snapshots before/after each rep with reps discarded if PID set changes, (ii) 5 reps with median reporting, (iii) interquartile width disclosed in every table." This is the path of least resistance and is reviewer-defensible.

## 4. Timer Scope Unification

Six variants split 3-3 on whether `build_initial_environments()` is inside `total_time`:
- **EXCLUDES env build** (env build runs, *then* `t_start`): dmrg-gpu (`dmrg_gpu_impl.h:1565,1568`), dmrg2-gpu (`dmrg2_gpu_impl.h:1508,1511`), pdmrg-gpu (`pdmrg_gpu_impl.h:2617,2620`)
- **INCLUDES env build** (`t_start`, *then* env build): dmrg-gpu-opt (`dmrg_gpu_opt_impl.h:1618,1621`), dmrg2-gpu-opt (`dmrg2_gpu_opt_impl.h:1600,1603`), pdmrg-gpu-opt (`pdmrg_gpu_opt_impl.h:3316,3319`)

**Standardize on INCLUDE-env-build.** Reasoning: env build is a real cost; excluding it lets `-gpu` variants under-report by 5–15% at chi=512, biasing the paper's "primary vs opt" comparisons. The `-opt` variants already report it; `-opt` already records `env_time` separately so per-table breakdowns are recoverable.

Patches (3 files, ~3 lines each — swap order of `build_initial_environments();` and `auto t_start = ...`, then add `double env_time` accounting like the `-opt` variants already do):
- `gpu-rocm/dmrg-gpu/src/dmrg_gpu_impl.h:1565–1568`
- `gpu-rocm/dmrg2-gpu/src/dmrg2_gpu_impl.h:1508–1511`
- `gpu-rocm/pdmrg-gpu/src/pdmrg_gpu_impl.h:2617–2620`

Also fix the base classes for parity even though they aren't the published binaries: `dmrg-gpu-base/src/dmrg_gpu_base_impl.h:791–793`, `dmrg2-gpu-base/...:799–801`, `pdmrg-gpu-base/...:1248–1250`, `pdmrg-multi-gpu/...:2405–2407`. Emit `env_build_sec` as a separate JSON field so existing tables are recomputable.

Add to `report_json`: `timer_scope: "include_env_build"`. Force-write it; older JSONs lacking it must be re-run, not back-patched.

## 5. CPU Host Correction Edits (main.tex)

§3.1 line 386 currently: "host CPU is an AMD EPYC processor". Replace with:
> "Host CPU on the MI300X system is an Intel Xeon Platinum 8470 (52 cores / 104 threads, 2 NUMA nodes, dual-socket). Host CPU on the H100 system is an Intel Xeon Platinum 8480C (56 cores / 112 threads, 2 NUMA nodes, dual-socket). Core counts and NUMA topology were captured by `lscpu` and `numactl --hardware` and are embedded in every result JSON's `provenance.host`."

§3.1 line 397: drop "thread counts of 1, 2, 4, 8, 12" (never ran — `run_mi300x_challenge.py:382` hardcodes threads=1). Replace with "single-thread quimb baseline (`OMP_NUM_THREADS=1`)". Drop "with cotengra for path optimization" entirely (cotengra never imported).

Drop the "OpenBLAS 0.3.28 source-built" sentence; replace with the actual `ldd quimb_runner | grep blas` output captured by the provenance patch.

Update Tables 1 and the appendix system table caption with the real Xeon SKUs and NUMA topology.

## 6. H100 Reruns

**Defer.** Rationale: H100 numbers are flagged P2 (FP64 caveat already documented), the env-build timer-scope fix changes only relative ordering within a column, not cross-architecture conclusions, and we haven't paid down the FP64 caveat — re-running now means re-running again post-FP64-fix. Add a footnote: "H100 figures use the prior timer scope; will be regenerated jointly with the FP64 precision audit (Cluster D)." If reviewers push back, that's one weekend to redo.

## 7. Effort + Dependencies

- provenance.py patches: 0.5 day, no deps.
- rocm-smi protocol + harness wrapper in `run_mi300x_challenge.py`: 1 day, depends on provenance.py.
- Timer-scope unification (3 binaries × edit + rebuild): 0.5 day code, **2 days GPU time** to rerun primary `-gpu` cells of Tables 4/6/7/10 on the remote.
- §3.1 text + table edits: 0.5 day, blocks on rerun (table numbers shift).
- VM disclosure paragraph: 0.5 day, parallelizable.
- **Total: ~5 working days, ~2 days GPU wall time**, single owner. Critical-path dep: timer fix → rerun → table edits.
