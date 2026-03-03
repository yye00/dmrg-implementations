# DMRG Benchmark Manifest

## Overview

This document catalogs all benchmarking scripts and results for the PDMRG and A2DMRG implementations.

## Current Benchmark Scripts

### 1. `a2dmrg/benchmarks/josephson_correctness_benchmark.py`
**Purpose:** Full correctness validation (10 runs)  
**Tests:** quimb DMRG1, DMRG2, PDMRG np=1,2,4,8, A2DMRG np=1,2,4,8  
**Model:** Josephson Junction Array (complex128)  
**Parameters:** L=20, D=50, n_max=2  
**Target:** All energies within 1e-10 of reference  
**Status:** ⚠️ Needs cleanup - too slow, not well-structured for publication

### 2. `a2dmrg/benchmarks/josephson_benchmark.py`
**Purpose:** PDMRG vs A2DMRG timing comparison  
**Tests:** quimb DMRG2 (serial reference), PDMRG np=1,2,4,8, A2DMRG np=1,2,4,8  
**Model:** Josephson Junction Array (complex128)  
**Status:** ⚠️ Needs tightening of convergence criteria

### 3. `a2dmrg/benchmarks/scaling_comparison.py`
**Purpose:** Scaling analysis with plots  
**Tests:** A2DMRG and PDMRG weak/strong scaling  
**Model:** Heisenberg spin-1/2  
**Status:** ⚠️ Generates plot but not rigorous enough for publication

### 4. `a2dmrg/benchmarks/compare_pdmrg_a2dmrg.py`
**Purpose:** Direct comparison on multiple models  
**Tests:** Heisenberg, RTFIM, Bose-Hubbard  
**Status:** ⚠️ Incomplete, uses placeholder models

### 5. `a2dmrg/benchmarks/josephson_junction.py`
**Purpose:** MPO builder for Josephson Junction model  
**Status:** ✅ Complete - core utility

### 6. `pdmrg/benchmarks/run_all_benchmarks.py`
**Purpose:** PDMRG benchmark suite  
**Tests:** Multiple models, sizes, bond dimensions  
**Status:** ⚠️ PDMRG only, no A2DMRG comparison

## Results Files

| File | Description |
|------|-------------|
| `a2dmrg/benchmarks/comparison_np1.json` | np=1 comparison results |
| `a2dmrg/benchmarks/comparison_results.json` | General comparison results |
| `a2dmrg/benchmarks/josephson_benchmark_results.json` | Josephson benchmark results |
| `a2dmrg/benchmarks/josephson_run.log` | Run log |
| `a2dmrg/benchmarks/pdmrg_vs_a2dmrg_scaling.png` | Scaling comparison plot |
| `a2dmrg/benchmarks/JOSEPHSON_JUNCTION_BENCHMARK.md` | Benchmark documentation |

## Last Known Results (from HEARTBEAT.md)

### PDMRG (all np values working)
| np | Energy | Time | ΔE from ref |
|----|--------|------|-------------|
| 1 | -7.8390664 | 100s | 3.3e-07 |
| 2 | -7.8390665 | 1138s | 3.8e-07 |
| 4 | -7.8390665 | 1321s | 3.4e-07 |
| 8 | -7.8390665 | 1514s | 3.3e-07 |

**Reference:** E = -7.839066116622 (quimb DMRG2)

## Issues Identified

1. **cotengra optimization:** Warning about missing optuna/nevergrad causing random sampling → extremely slow tensor contractions
2. **A2DMRG stuck:** np=1 was killed after 12+ hours in sweep 2 phase 2
3. **Timing inconsistency:** PDMRG np=1 should be comparable to quimb, but was 100s vs quimb's ~350s
4. **No clean publication-ready script:** Current benchmarks are scattered and incomplete

## Recommended Actions

1. **Fix cotengra:** Ensure `optuna` is properly imported by quimb/cotengra
2. **Create single publication benchmark:** One script that runs all methods cleanly
3. **Tighten convergence:** Use tol=1e-12 or tighter
4. **Add timing breakdown:** Warmup time vs sweep time
5. **Validate correctness first:** Before timing, ensure all methods converge to same energy

## Publication Benchmark Requirements

For a compelling benchmark we need:

### Table 1: Correctness Validation
- Reference: quimb DMRG2 (two-site, serial)
- All implementations must match to machine precision (~1e-12)
- Parameters: L=20, D=50, complex128

### Table 2: Performance Scaling
- quimb DMRG1 (single-site, serial) - baseline
- quimb DMRG2 (two-site, serial) - baseline  
- PDMRG np=1,2,4,8
- A2DMRG np=1,2,4,8
- Report: Time to solution, speedup relative to serial

### Figure 1: Strong Scaling
- Fixed problem size (L=40, D=100)
- Plot time vs number of processors
- Show ideal scaling line for comparison
