#!/bin/bash
# =============================================================================
# PDMRG Heisenberg Model Benchmark
# =============================================================================
#
# Spin-1/2 antiferromagnetic Heisenberg chain:
#   H = Σᵢ (Sˣᵢ Sˣᵢ₊₁ + Sʸᵢ Sʸᵢ₊₁ + Sᶻᵢ Sᶻᵢ₊₁)
#
# Reference ground state energy per site (L→∞): E/L = -0.4432...
#
# =============================================================================

set -e

echo "=============================================================="
echo "PDMRG Heisenberg Model Benchmark"
echo "Date: $(date)"
echo "=============================================================="

L=40
M=50

echo ""
echo "Configuration: L=$L, m=$M"
echo "Expected E/L ≈ -0.433 (finite size)"
echo ""

echo "--- Serial DMRG (np=1) ---"
python -m pdmrg --sites $L --bond-dim $M --warmup-dim $M \
    --model heisenberg --sweeps 20 --tol 1e-10 --timing 2>&1 | \
    grep -E "Warmup|Sweep|Final|Total"

echo ""
echo "--- Parallel DMRG (np=2) ---"
mpirun --oversubscribe -np 2 python -m pdmrg --sites $L --bond-dim $M --warmup-dim $M \
    --model heisenberg --sweeps 20 --tol 1e-10 --timing 2>&1 | \
    grep -E "Warmup|Sweep|Final|Total"

echo ""
echo "--- Parallel DMRG (np=4) ---"
mpirun --oversubscribe -np 4 python -m pdmrg --sites $L --bond-dim $M --warmup-dim $M \
    --model heisenberg --sweeps 20 --tol 1e-10 --timing 2>&1 | \
    grep -E "Warmup|Sweep|Final|Total"

echo ""
echo "=============================================================="
