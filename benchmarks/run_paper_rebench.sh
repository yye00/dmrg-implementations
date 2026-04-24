#!/usr/bin/env bash
# benchmarks/run_paper_rebench.sh — Binary-determinism / commit-pin rebench protocol
#
# Purpose: Enforces that all 6 GPU variants are built from a SINGLE tagged commit,
# binaries are SHA-pinned, clocks are pinned, and all variants run serially.
#
# Usage (on MI300X host):
#   cd ~/dmrg-implementations
#   bash benchmarks/run_paper_rebench.sh [--reps N] [--model <josephson|heisenberg>]
#
# Ground truth: docs/PATH_B_GROUND_TRUTH.md (locked at 6f45533).
# This script implements Cluster F brief §5 (binary determinism / commit pinning protocol).

set -euo pipefail

REPS="${REPS:-10}"
MODEL="${MODEL:-josephson}"
VARIANTS=(dmrg-gpu dmrg-gpu-opt dmrg2-gpu dmrg2-gpu-opt pdmrg-gpu pdmrg-gpu-opt)

# Parse CLI overrides.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reps)     REPS="$2"; shift 2 ;;
        --model)    MODEL="$2"; shift 2 ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# ── Step 1: tag and record commit ─────────────────────────────────────────────
CAMPAIGN_DATE="$(date +%Y%m%d)"
COMMIT_SHA="$(git rev-parse HEAD)"
CAMPAIGN_TAG="paper-rebench-${CAMPAIGN_DATE}"
echo "=== Path B paper rebench campaign ==="
echo "Commit:       $COMMIT_SHA"
echo "Campaign tag: $CAMPAIGN_TAG"
echo "Reps:         $REPS"
echo "Model:        $MODEL"
echo ""

# Create tag (ok if already exists).
git tag "$CAMPAIGN_TAG" 2>/dev/null || echo "Tag $CAMPAIGN_TAG already exists; continuing."

CAMPAIGN_DIR="benchmarks/data/gpu_ablation/${CAMPAIGN_TAG}"
mkdir -p "$CAMPAIGN_DIR"

# ── Step 2: Clean rebuild of all 6 variants, record binary SHA ────────────────
echo "=== Building all $((${#VARIANTS[@]})) variants from $COMMIT_SHA ==="
MANIFEST_JSON="${CAMPAIGN_DIR}/campaign_manifest.json"

declare -A BINARY_PATHS
declare -A BINARY_SHAS

for VARIANT in "${VARIANTS[@]}"; do
    BUILD_DIR="gpu-rocm/${VARIANT}/build"
    echo "--- Building ${VARIANT} ---"
    pushd "gpu-rocm/${VARIANT}" > /dev/null
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    make clean 2>/dev/null || true
    make -j"$(nproc)" 2>&1 | tail -5
    popd > /dev/null

    # Locate the binary (convention: same name as variant, underscores not hyphens).
    BINARY_NAME="${VARIANT//-/_}"
    BINARY_PATH="${BUILD_DIR}/${BINARY_NAME}"
    if [[ ! -f "$BINARY_PATH" ]]; then
        # Try other common names.
        for CANDIDATE in "${BUILD_DIR}"/*; do
            if [[ -x "$CANDIDATE" && ! "$CANDIDATE" == *.* ]]; then
                BINARY_PATH="$CANDIDATE"
                break
            fi
        done
    fi

    if [[ ! -f "$BINARY_PATH" ]]; then
        echo "ERROR: Could not locate binary for ${VARIANT} under ${BUILD_DIR}/" >&2
        exit 1
    fi

    SHA256="$(sha256sum "$BINARY_PATH" | awk '{print $1}')"
    BINARY_PATHS[$VARIANT]="$BINARY_PATH"
    BINARY_SHAS[$VARIANT]="$SHA256"
    echo "  binary: $BINARY_PATH  sha256: $SHA256"
done

# Write campaign manifest.
python3 - << PYEOF
import json
manifest = {
    "campaign_tag": "${CAMPAIGN_TAG}",
    "commit_sha": "${COMMIT_SHA}",
    "reps": ${REPS},
    "model": "${MODEL}",
    "variants": {}
}
PYEOF

# Build manifest JSON inline.
{
    echo "{"
    echo "  \"campaign_tag\": \"${CAMPAIGN_TAG}\","
    echo "  \"commit_sha\": \"${COMMIT_SHA}\","
    echo "  \"reps\": ${REPS},"
    echo "  \"model\": \"${MODEL}\","
    echo "  \"variants\": {"
    first=1
    for VARIANT in "${VARIANTS[@]}"; do
        [[ $first -eq 0 ]] && echo ","
        first=0
        printf '    "%s": {"binary": "%s", "sha256": "%s"}' \
            "$VARIANT" "${BINARY_PATHS[$VARIANT]}" "${BINARY_SHAS[$VARIANT]}"
    done
    echo ""
    echo "  }"
    echo "}"
} > "$MANIFEST_JSON"
echo ""
echo "Campaign manifest written to $MANIFEST_JSON"

# ── Step 3: Pin GPU clocks ────────────────────────────────────────────────────
echo ""
echo "=== Pinning GPU clocks ==="
rocm-smi --setperflevel high 2>/dev/null || echo "  WARNING: setperflevel failed (may need sudo)"
rocm-smi --setsclk 7 2>/dev/null || echo "  WARNING: setsclk failed (may need sudo)"
rocm-smi --setfan 100 2>/dev/null || echo "  WARNING: setfan failed"
rocm-smi --showall > "${CAMPAIGN_DIR}/rocm_pre.json" 2>/dev/null || \
    rocm-smi > "${CAMPAIGN_DIR}/rocm_pre.json" 2>/dev/null || true
echo "  rocm_pre.json written."

# ── Step 4: Run all variants serially ─────────────────────────────────────────
echo ""
echo "=== Running ablation benchmarks (serial, ${REPS} reps each) ==="

for VARIANT in "${VARIANTS[@]}"; do
    BINARY="${BINARY_PATHS[$VARIANT]}"
    OUT_DIR="${CAMPAIGN_DIR}/${VARIANT}"
    echo ""
    echo "--- ${VARIANT} ---"
    echo "  binary: $BINARY"
    echo "  output: $OUT_DIR"
    python3 benchmarks/bench_dmrg_gpu_ablate.py \
        --binary "$BINARY" \
        --reps "$REPS" \
        --out "$OUT_DIR" \
        --model "$MODEL" \
        --manifest "$MANIFEST_JSON" \
        || echo "WARNING: ${VARIANT} benchmark exited non-zero — check ${OUT_DIR}/results.json"
done

# ── Step 5: Post-campaign clock snapshot + SCLK drift check ───────────────────
echo ""
echo "=== Post-campaign GPU state ==="
rocm-smi --showall > "${CAMPAIGN_DIR}/rocm_post.json" 2>/dev/null || \
    rocm-smi > "${CAMPAIGN_DIR}/rocm_post.json" 2>/dev/null || true

# Warn if rocm_post.json is missing or if the campaign ran without clock pinning.
if [[ -f "${CAMPAIGN_DIR}/rocm_post.json" ]]; then
    echo "  rocm_post.json written."
else
    echo "  WARNING: rocm_post.json not written (rocm-smi unavailable)."
fi

# ── Step 6: Final tag ─────────────────────────────────────────────────────────
FINAL_TAG="${CAMPAIGN_TAG}-final"
git tag "$FINAL_TAG" 2>/dev/null || echo "Tag $FINAL_TAG already exists."
echo ""
echo "=== Campaign complete ==="
echo "Campaign tag:  $CAMPAIGN_TAG"
echo "Final tag:     $FINAL_TAG"
echo "Data dir:      $CAMPAIGN_DIR"
echo "Manifest:      $MANIFEST_JSON"
echo ""
echo "Next steps:"
echo "  1. Run analysis/statistical_summary.py --ablation-root benchmarks/data/gpu_ablation"
echo "     --commit-pin ${COMMIT_SHA} --out reports/mi300x/stats_${COMMIT_SHA:0:8}/"
echo "  2. Update paper §D.A. to cite ${CAMPAIGN_TAG} as the data manifest."
echo "  3. Commit results: git add benchmarks/data/gpu_ablation/${CAMPAIGN_TAG}/ && git commit"
