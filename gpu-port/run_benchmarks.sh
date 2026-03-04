#!/bin/bash
# Run comprehensive DMRG-GPU benchmarks
# Tests PDMRG vs PDMRG2 with stream scalability

set -e

echo "========================================="
echo "GPU DMRG Benchmark Suite"
echo "========================================="
echo ""

# Check if build exists
if [ ! -f "build/dmrg_benchmark" ]; then
    echo "Error: dmrg_benchmark not found. Please build first:"
    echo "  cd build && cmake .. && make -j8"
    exit 1
fi

# Create results directory
mkdir -p benchmark_results
cd benchmark_results

echo "Running benchmarks on MI300X..."
echo ""

# L=12 benchmark (default)
echo "Test 1: L=12, bond=100, sweeps=5"
echo "-----------------------------------"
../build/dmrg_benchmark 12 100 5 | tee l12_results.txt
mv benchmark_results.csv l12_results.csv

echo ""
echo "Test 2: L=12, bond=200, sweeps=10 (larger bond)"
echo "------------------------------------------------"
../build/dmrg_benchmark 12 200 10 | tee l12_bond200_results.txt
mv benchmark_results.csv l12_bond200_results.csv

echo ""
echo "Test 3: L=14, bond=100, sweeps=5 (longer chain)"
echo "------------------------------------------------"
../build/dmrg_benchmark 14 100 5 | tee l14_results.txt
mv benchmark_results.csv l14_results.csv

echo ""
echo "========================================="
echo "All benchmarks complete!"
echo "========================================="
echo ""
echo "Results saved in benchmark_results/:"
ls -lh *.txt *.csv

echo ""
echo "Key metrics:"
echo "- Energy accuracy (should be < 1e-10)"
echo "- PDMRG2 speedup vs PDMRG (expect 2-3x)"
echo "- Stream scaling (expect ~1.5-2x from 1→8 streams)"

# Generate comparison plot if gnuplot is available
if command -v gnuplot &> /dev/null; then
    echo ""
    echo "Generating plots..."

    cat > plot_speedup.gnu << 'EOF'
set terminal png size 1200,800
set output 'speedup_comparison.png'
set title 'DMRG-GPU Stream Scalability'
set xlabel 'Number of Streams'
set ylabel 'Speedup vs Single-Stream PDMRG'
set grid
set key left top

plot 'l12_results.csv' using 2:6 with linespoints title 'PDMRG L=12' lw 2, \
     '' using 2:6 every ::4 with linespoints title 'PDMRG2 L=12' lw 2, \
     x title 'Ideal Linear' dashtype 2
EOF

    gnuplot plot_speedup.gnu 2>/dev/null && echo "Plot saved: speedup_comparison.png"
fi

cd ..
echo ""
echo "Done!"
