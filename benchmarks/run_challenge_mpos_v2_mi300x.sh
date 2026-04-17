#!/usr/bin/env bash
# V2 challenge benchmark grid on MI300X — targets harder regimes than v1
# (which hit the Majumdar-Ghosh point at J2=0.5, too easy).
#
# Goals:
#   1. Non-MG J1-J2 (J2 in {0.3, 0.4, 0.6, 0.7}) — real frustration, needs more sweeps.
#   2. Large-χ (χ=256) — amortize pdmrg segment overhead.
#   3. J1-J2-J3 — extended-range frustration (new MPO, D=20).
#   4. Ladder at χ=256 — stress d=4 supersites further.
#
# USAGE: ./run_challenge_mpos_v2_mi300x.sh [run_tag]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="${1:-$(date -u +%Y%m%dT%H%MZ)}"
OUTDIR="${REPO_ROOT}/benchmarks/results/mi300x/challenge_mpos_v2/${RUN_TAG}"
mkdir -p "${OUTDIR}"

DMRG1_BIN="${REPO_ROOT}/gpu-rocm/dmrg-gpu/build/dmrg_gpu"
DMRG2_BIN="${REPO_ROOT}/gpu-rocm/dmrg2-gpu/build/dmrg2_gpu"
PDMRG_BIN="${REPO_ROOT}/gpu-rocm/pdmrg-gpu/build/pdmrg_gpu"

# Uniform pdmrg flags per PDMRG rules (single-site warmup/polish, n_warmup<=2).
PDMRG_COMMON="--outer 10 --segments 4 --local 2 --warmup 2 --polish 0"

run_one() {
  local binname="$1"; shift
  local label="$1"; shift
  local logfile="${OUTDIR}/${binname}_${label}.log"
  echo "=== ${binname} ${label} ==="
  echo "CMD: $*" | tee "${logfile}"
  "$@" 2>&1 | tee -a "${logfile}" | tail -30
  echo
}

# ============================================================================
# 1. Non-MG J1-J2 at L=100, χ=128
# ============================================================================
for J2 in 0.3 0.4 0.6 0.7; do
  label_j2="j1j2_L100_chi128_J2${J2/./p}"
  run_one dmrg1 "${label_j2}" \
    "${DMRG1_BIN}" --j1j2 --L 100 --chi 128 --sweeps 10 --j1 1.0 --j2 "${J2}"
  run_one dmrg2 "${label_j2}" \
    "${DMRG2_BIN}" --j1j2 --L 100 --chi 128 --sweeps 5  --j1 1.0 --j2 "${J2}"
  run_one pdmrg "${label_j2}" \
    "${PDMRG_BIN}" --j1j2 --L 100 --chi 128 ${PDMRG_COMMON} --j1 1.0 --j2 "${J2}"
done

# ============================================================================
# 2. Large-χ (χ=256) on J1-J2 at J2=0.4 (non-MG)
# ============================================================================
for L in 50 100; do
  label_chi="j1j2_L${L}_chi256_J2p4"
  run_one dmrg1 "${label_chi}" \
    "${DMRG1_BIN}" --j1j2 --L "${L}" --chi 256 --sweeps 10 --j1 1.0 --j2 0.4
  run_one dmrg2 "${label_chi}" \
    "${DMRG2_BIN}" --j1j2 --L "${L}" --chi 256 --sweeps 5  --j1 1.0 --j2 0.4
  run_one pdmrg "${label_chi}" \
    "${PDMRG_BIN}" --j1j2 --L "${L}" --chi 256 ${PDMRG_COMMON} --j1 1.0 --j2 0.4
done

# ============================================================================
# 3. J1-J2-J3 (extended-range frustration) at J2=0.4, J3=0.2
# ============================================================================
for L in 50 100 200; do
  for chi in 64 128; do
    label_j3="j1j2j3_L${L}_chi${chi}"
    run_one dmrg1 "${label_j3}" \
      "${DMRG1_BIN}" --j1j2j3 --L "${L}" --chi "${chi}" --sweeps 10 --j1 1.0 --j2 0.4 --j3 0.2
    run_one dmrg2 "${label_j3}" \
      "${DMRG2_BIN}" --j1j2j3 --L "${L}" --chi "${chi}" --sweeps 5  --j1 1.0 --j2 0.4 --j3 0.2
    run_one pdmrg "${label_j3}" \
      "${PDMRG_BIN}" --j1j2j3 --L "${L}" --chi "${chi}" ${PDMRG_COMMON} --j1 1.0 --j2 0.4 --j3 0.2
  done
done

# ============================================================================
# 4. Ladder at χ=256 (stress pdmrg vs dmrg2 on d=4)
# ============================================================================
for LRUNG in 50 100; do
  label_lad="ladder_L${LRUNG}_chi256"
  run_one dmrg1 "${label_lad}" \
    "${DMRG1_BIN}" --ladder --L "${LRUNG}" --chi 256 --sweeps 10 --jleg 1.0 --jrung 1.0
  run_one dmrg2 "${label_lad}" \
    "${DMRG2_BIN}" --ladder --L "${LRUNG}" --chi 256 --sweeps 5  --jleg 1.0 --jrung 1.0
  run_one pdmrg "${label_lad}" \
    "${PDMRG_BIN}" --ladder --L "${LRUNG}" --chi 256 ${PDMRG_COMMON} --jleg 1.0 --jrung 1.0
done

echo
echo "All runs complete. Logs in: ${OUTDIR}"
ls -1 "${OUTDIR}"
