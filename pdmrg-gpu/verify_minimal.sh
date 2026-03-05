#!/bin/bash
# Verification script for minimal GPU DMRG

echo "=== Minimal GPU DMRG Verification ==="
echo

echo "1. Checking source code..."
if [ -f "src/dmrg_minimal_gpu.cpp" ]; then
    lines=$(wc -l < src/dmrg_minimal_gpu.cpp)
    echo "   ✓ dmrg_minimal_gpu.cpp found ($lines lines)"
else
    echo "   ✗ dmrg_minimal_gpu.cpp NOT FOUND"
    exit 1
fi

echo
echo "2. Checking documentation..."
docs=(
    "INDEX_MINIMAL.md"
    "QUICK_START_MINIMAL.md"
    "MINIMAL_GPU_IMPLEMENTATION.md"
    "OPTIMIZATION_COMPARISON.md"
    "MINIMAL_VERSION_SUMMARY.md"
)

for doc in "${docs[@]}"; do
    if [ -f "$doc" ]; then
        size=$(du -h "$doc" | cut -f1)
        echo "   ✓ $doc ($size)"
    else
        echo "   ✗ $doc NOT FOUND"
    fi
done

echo
echo "3. Code structure analysis..."
echo "   Functions in dmrg_minimal_gpu.cpp:"
grep -E "^\s*(void|double|int|class)\s+\w+\(" src/dmrg_minimal_gpu.cpp | head -10 | sed 's/^/   - /'

echo
echo "4. Key features check..."
echo "   Checking for removed components:"
if ! grep -q "class HeisenbergMPO" src/dmrg_minimal_gpu.cpp; then
    echo "   ✓ HeisenbergMPO class removed"
fi
if ! grep -q "class Environments" src/dmrg_minimal_gpu.cpp; then
    echo "   ✓ Environments class removed"
fi
if ! grep -q "update_left_env" src/dmrg_minimal_gpu.cpp; then
    echo "   ✓ Environment updates removed"
fi

echo
echo "   Checking for preserved features:"
if grep -q "ldvt, k" src/dmrg_minimal_gpu.cpp; then
    echo "   ✓ SVD fix preserved (ldvt=k)"
fi
if grep -q "apply_local_heisenberg" src/dmrg_minimal_gpu.cpp; then
    echo "   ✓ Local Hamiltonian application present"
fi
if grep -q "PowerIterationSolver" src/dmrg_minimal_gpu.cpp; then
    echo "   ✓ Power iteration solver present"
fi

echo
echo "5. Size comparison..."
if [ -f "src/dmrg_with_environments.cpp" ]; then
    old_lines=$(wc -l < src/dmrg_with_environments.cpp)
    new_lines=$(wc -l < src/dmrg_minimal_gpu.cpp)
    reduction=$(( (old_lines - new_lines) * 100 / old_lines ))
    echo "   Original: $old_lines lines"
    echo "   Minimal:  $new_lines lines"
    echo "   Reduction: $reduction%"
fi

echo
echo "=== Verification Complete ==="
echo
echo "Quick start:"
echo "  hipcc -O3 src/dmrg_minimal_gpu.cpp -lrocblas -lrocsolver -o bin/dmrg_minimal_gpu"
echo
echo "Documentation index:"
echo "  cat INDEX_MINIMAL.md"
