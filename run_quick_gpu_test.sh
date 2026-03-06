#!/bin/bash
# Quick GPU test
cd ~/dmrg-implementations
echo '=== PDMRG_GPU Heisenberg ==='
./gpu-port/build/pdmrg_gpu --model heisenberg --L 12 --max-D 100 --sweeps 10 --streams 1 2>&1 | grep 'Final E:'
echo '=== PDMRG2_GPU Heisenberg ==='
./gpu-port/build/pdmrg2_gpu --model heisenberg --L 12 --max-D 100 --sweeps 10 --streams 1 2>&1 | grep 'Final E:'
echo '=== PDMRG_GPU Josephson ==='
./gpu-port/build/pdmrg_gpu --model josephson --L 8 --max-D 50 --sweeps 10 --streams 1 --n-max 2 2>&1 | grep 'Final E:'
echo '=== PDMRG2_GPU Josephson ==='
./gpu-port/build/pdmrg2_gpu --model josephson --L 8 --max-D 50 --sweeps 10 --streams 1 --n-max 2 2>&1 | grep 'Final E:'
