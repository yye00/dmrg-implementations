#!/bin/bash
# Build all 6 GPU variants for MI300X
# Usage: ./gpu-rocm/build_all.sh [--clean]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANTS=(dmrg-gpu dmrg-gpu-opt dmrg2-gpu dmrg2-gpu-opt pdmrg-gpu pdmrg-gpu-opt)

CLEAN=0
if [[ "$1" == "--clean" ]]; then
    CLEAN=1
fi

echo "=========================================="
echo "Building all GPU variants for MI300X"
echo "Variants: ${VARIANTS[*]}"
echo "=========================================="
echo ""

for variant in "${VARIANTS[@]}"; do
    echo "------------------------------------------"
    echo ">> $variant"
    echo "------------------------------------------"
    dir="$SCRIPT_DIR/$variant"
    if [[ ! -f "$dir/build_mi300x.sh" ]]; then
        echo "ERROR: $dir/build_mi300x.sh not found"
        exit 1
    fi
    if [[ "$CLEAN" == "1" && -d "$dir/build" ]]; then
        echo "Cleaning $dir/build"
        rm -rf "$dir/build"
    fi
    (cd "$dir" && ./build_mi300x.sh) || exit 1
    echo ""
done

echo "=========================================="
echo "ALL VARIANTS BUILT SUCCESSFULLY"
echo "=========================================="
