#!/bin/bash
# Full A2DMRG Test Suite
# Run validation and scalability tests

set -e

export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH
export PATH=/usr/lib64/openmpi/bin:$PATH

cd "$(dirname "$0")"

echo "========================================"
echo "A2DMRG Full Test Suite"
echo "Date: $(date)"
echo "========================================"
echo

# Part 1: Accuracy test with np=2
echo "=== PART 1: Accuracy Test (np=2 vs Quimb) ==="
echo
mpirun -np 2 python test_full_validation.py 2>&1 | tee results_accuracy.txt

echo
echo "=== PART 2: Scalability Study ==="
echo

# Run scalability tests for np=1,2,4,8
RESULTS_FILE="results_scalability.txt"
echo "# Scalability Results" > $RESULTS_FILE
echo "# np L chi E time" >> $RESULTS_FILE

for NP in 1 2 4; do
    echo "--- Testing np=$NP ---"
    mpirun -np $NP python test_scalability.py 2>&1 | tee -a results_np${NP}.txt
    # Extract result line
    grep "^RESULT:" results_np${NP}.txt >> $RESULTS_FILE || true
done

echo
echo "=== SCALABILITY SUMMARY ==="
cat $RESULTS_FILE

echo
echo "=== TEST SUITE COMPLETE ==="
