#!/usr/bin/env bash
# Run J1-J2 + 2-leg ladder challenge MPOs on MI300X across dmrg-gpu, dmrg2-gpu, pdmrg-gpu.
# Captures raw logs into benchmarks/results/mi300x/challenge_mpos/<timestamp>/.
#
# USAGE: ./run_challenge_mpos_mi300x.sh [run_tag]
#
# Follows PDMRG rules: single-site warmup+polish only, n_warmup <= 2, n_polish <= 2.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="${1:-$(date -u +%Y%m%dT%H%MZ)}"
OUTDIR="${REPO_ROOT}/benchmarks/results/mi300x/challenge_mpos/${RUN_TAG}"
mkdir -p "${OUTDIR}"

DMRG1_BIN="${REPO_ROOT}/gpu-rocm/dmrg-gpu/build/dmrg_gpu"
DMRG2_BIN="${REPO_ROOT}/gpu-rocm/dmrg2-gpu/build/dmrg2_gpu"
PDMRG_BIN="${REPO_ROOT}/gpu-rocm/pdmrg-gpu/build/pdmrg_gpu"

run_one() {
  local binname="$1"; shift
  local label="$1"; shift
  local logfile="${OUTDIR}/${binname}_${label}.log"
  echo "=== ${binname} ${label} ==="
  echo "CMD: $*" | tee "${logfile}"
  "$@" 2>&1 | tee -a "${logfile}" | tail -30
  echo
}

# ===== J1-J2 (J2=0.5, maximally frustrated Majumdar-Ghosh point) =====
for L in 50 100 200; do
  for chi in 64 128; do
    run_one dmrg1 "j1j2_L${L}_chi${chi}" \
      "${DMRG1_BIN}" --j1j2 --L "${L}" --chi "${chi}" --sweeps 10 --j1 1.0 --j2 0.5
    run_one dmrg2 "j1j2_L${L}_chi${chi}" \
      "${DMRG2_BIN}" --j1j2 --L "${L}" --chi "${chi}" --sweeps 5 --j1 1.0 --j2 0.5
    # PDMRG: warmup=2 single-site, outer=6 parallel segments, polish=0 (per PDMRG rules)
    run_one pdmrg "j1j2_L${L}_chi${chi}" \
      "${PDMRG_BIN}" --j1j2 --L "${L}" --chi "${chi}" \
      --outer 6 --segments 4 --local 2 --warmup 2 --polish 0 \
      --j1 1.0 --j2 0.5
  done
done

# ===== 2-leg Heisenberg ladder (J_leg = J_rung = 1) =====
for LRUNG in 50 100; do
  for chi in 64 128; do
    run_one dmrg1 "ladder_L${LRUNG}_chi${chi}" \
      "${DMRG1_BIN}" --ladder --L "${LRUNG}" --chi "${chi}" --sweeps 10 --jleg 1.0 --jrung 1.0
    run_one dmrg2 "ladder_L${LRUNG}_chi${chi}" \
      "${DMRG2_BIN}" --ladder --L "${LRUNG}" --chi "${chi}" --sweeps 5 --jleg 1.0 --jrung 1.0
    run_one pdmrg "ladder_L${LRUNG}_chi${chi}" \
      "${PDMRG_BIN}" --ladder --L "${LRUNG}" --chi "${chi}" \
      --outer 6 --segments 4 --local 2 --warmup 2 --polish 0 \
      --jleg 1.0 --jrung 1.0
  done
done

echo
echo "All runs complete. Logs in: ${OUTDIR}"
ls -1 "${OUTDIR}"
