#!/bin/bash
# Quick GPU benchmark test to verify correctness

set -e
cd ~/dmrg-implementations

echo "=========================================="
echo "GPU QUICK BENCHMARK TEST"
echo "=========================================="
echo ""

# Test 1: PDMRG_GPU Heisenberg
echo "=== Test 1: PDMRG_GPU Heisenberg L=12 D=100 ==="
./gpu-port/build/pdmrg_gpu --model heisenberg --L 12 --max-D 100 --sweeps 10 --streams 1 | grep "Final E:"

# Test 2: PDMRG2_GPU Heisenberg
echo ""
echo "=== Test 2: PDMRG2_GPU Heisenberg L=12 D=100 ==="
./gpu-port/build/pdmrg2_gpu --model heisenberg --L 12 --max-D 100 --sweeps 10 --streams 1 | grep "Final E:"

# Test 3: PDMRG_GPU Josephson
echo ""
echo "=== Test 3: PDMRG_GPU Josephson L=8 D=50 ==="
./gpu-port/build/pdmrg_gpu --model josephson --L 8 --max-D 50 --sweeps 10 --streams 1 --n-max 2 | grep "Final E:"

# Test 4: PDMRG2_GPU Josephson
echo ""
echo "=== Test 4: PDMRG2_GPU Josephson L=8 D=50 ==="
./gpu-port/build/pdmrg2_gpu --model josephson --L 8 --max-D 50 --sweeps 10 --streams 1 --n-max 2 | grep "Final E:"

echo ""
echo "=========================================="
echo "Expected energies:"
echo "  Heisenberg L=12: -5.142090632841"
echo "  Josephson L=8:   -2.843801043139"
echo "=========================================="
