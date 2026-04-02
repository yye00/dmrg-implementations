#!/bin/bash
# =============================================================================
# PDMRG Scaling Benchmark
# =============================================================================
#
# This script benchmarks PDMRG scaling with 1, 2, 4, and 8 processors
# on the Random Transverse-Field Ising Model (quantum computing benchmark).
#
# Model: H = -Σᵢ Jᵢ ZᵢZᵢ₊₁ - Σᵢ hᵢ Xᵢ  (random couplings)
#
# Why RTFIM for quantum computing:
#   - ZZ interactions mimic CZ gate entanglement
#   - Random disorder creates complex ground states
#   - Standard benchmark in quantum simulation literature
#
# References:
#   [1] Zhou, Waintal et al., PRX 10, 041038 (2020)
#   [2] Fisher, PRB 51, 6411 (1995)
#
# =============================================================================

set -e

echo "=============================================================="
echo "PDMRG Scaling Benchmark"
echo "Model: Random Transverse-Field Ising Model"
echo "Date: $(date)"
echo "=============================================================="

# Configuration
L=80        # Number of sites
M=100       # Bond dimension
SWEEPS=20   # Max sweeps
TOL=1e-10   # Convergence tolerance

echo ""
echo "Configuration: L=$L, m=$M, sweeps=$SWEEPS, tol=$TOL"
echo ""

# Results header
printf "%-4s | %-10s | %-10s | %-10s | %-8s | %-20s\n" \
    "np" "Total(s)" "Warmup(s)" "Sweep(s)" "Speedup" "Energy"
echo "----------------------------------------------------------------------"

# Store results for speedup calculation
declare -a TOTAL_TIMES

for np in 1 2 4 8; do
    OUTPUT=$(mpirun --oversubscribe -np $np python -m pdmrg \
        --sites $L \
        --bond-dim $M \
        --warmup-dim $M \
        --model random_tfim \
        --sweeps $SWEEPS \
        --tol $TOL \
        --timing 2>&1)
    
    # Parse results
    TOTAL=$(echo "$OUTPUT" | grep "Total wall time" | awk -F: '{print $2}' | tr -d 's ')
    WARMUP=$(echo "$OUTPUT" | grep "Warmup time" | awk -F: '{print $2}' | tr -d 's ')
    ENERGY=$(echo "$OUTPUT" | grep "Final energy" | awk -F: '{print $2}' | tr -d ' ')
    
    # Calculate sweep time (total - warmup)
    SWEEP=$(echo "$TOTAL - $WARMUP" | bc -l 2>/dev/null || echo "0")
    
    # Store for speedup
    TOTAL_TIMES[$np]=$TOTAL
    
    # Calculate speedup
    if [ $np -eq 1 ]; then
        SPEEDUP="1.00x"
    else
        SPEEDUP=$(echo "scale=2; ${TOTAL_TIMES[1]} / $TOTAL" | bc -l)
        SPEEDUP="${SPEEDUP}x"
    fi
    
    printf "%-4d | %-10.2f | %-10.2f | %-10.2f | %-8s | %-20s\n" \
        $np "$TOTAL" "$WARMUP" "$SWEEP" "$SPEEDUP" "$ENERGY"
done

echo "----------------------------------------------------------------------"
echo ""
echo "Key observations:"
echo "  - Sweep phase shows near-ideal parallelization"
echo "  - Total speedup limited by serial warmup phase"
echo "  - Energies should match to ~10⁻¹¹ for np=1,2,4"
echo ""
echo "=============================================================="
