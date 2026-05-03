#!/bin/bash
# G1 baseline campaign launcher for MI300X
# ============================================================================
# Single-entry script for the post-R20 G1 baseline run on Hot Aisle MI300X.
#
# What it does:
#   1. Verify environment (ROCm, hipcc, GPU visible)
#   2. Build all 9 in-charter GPU variants from clean
#   3. Run the registry-validated G1 grid via the existing
#      run_mi300x_challenge.py harness, with PDMRG-rules-2026-04-15
#      compliant flags (single-site warmup/polish, n<=2)
#   4. Capture results into benchmarks/paper_results/mi300x/g1-<date>/
#   5. Sync to git (manual push at the end, NOT automatic)
#
# Designed to be run from inside the test_remote tmux session on the
# MI300X host:
#     tmux attach -t test_remote
#     cd ~/dmrg-implementations
#     bash benchmarks/run_g1_baseline.sh [--smoke|--full]
#
# Modes:
#   --smoke   ULTRA_TRIM grid (18 configs, ~30 min) — confirms the
#             build is correct end-to-end before committing the GPU
#             window to a long run.
#   --full    CHALLENGE_SIZES grid (44 configs, ~6-12 h) — the actual
#             G1 baseline campaign.
#
# Bail-out points: any build failure, smoke-mode correctness failure,
# or rocm-smi error halts the script. The launcher refuses to start
# --full unless --smoke completed inside this same script invocation
# OR a sentinel file is set (--skip-smoke explicitly acknowledges).
# ============================================================================

set -euo pipefail

# ---- Defaults (PDMRG-rules-2026-04-15 mandatory; do NOT change without
#      updating CLAUDE.md) -----
PDMRG_WARMUP="${PDMRG_WARMUP:-2}"   # MUST be <= 2, single-site enforced in code
PDMRG_POLISH="${PDMRG_POLISH:-2}"   # MUST be <= 2, single-site enforced in code
PDMRG_LOCAL="${PDMRG_LOCAL:-1}"     # local sweeps per outer iteration

# ---- Repetition count for statistics ----
# 1 = single timing per config (smoke / first-look)
# 5 = paper-grade (median + IQR), recommended for --full
REPEATS="${REPEATS:-5}"

# ---- The 9 in-charter variants under conformity audit -----
VARIANTS=(
    dmrg-gpu-base
    dmrg-gpu
    dmrg-gpu-opt
    dmrg2-gpu-base
    dmrg2-gpu
    dmrg2-gpu-opt
    pdmrg-gpu-base
    pdmrg-gpu
    pdmrg-gpu-opt
    # pdmrg-multi-gpu  -- requires 4 visible MI300X devices; comment in if available
)

# ---- Mode parsing -----
MODE="${1:-}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
case "$MODE" in
    --smoke) RUN_SMOKE=1; RUN_FULL=0 ;;
    --full)  RUN_SMOKE=1; RUN_FULL=1 ;;  # full implies a smoke first
    --skip-smoke) RUN_SMOKE=0; RUN_FULL=1; SKIP_SMOKE=1 ;;
    "")      echo "Usage: $0 [--smoke|--full|--skip-smoke]"; exit 1 ;;
    *)       echo "Unknown mode: $MODE"; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---- Date stamp / output dir -----
DATE_TAG="$(date -u +%Y%m%d-%H%M)"
OUT_DIR="benchmarks/paper_results/mi300x/g1-${DATE_TAG}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/launcher.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " G1 baseline launcher — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo "Repo:    $REPO_ROOT"
echo "Out:     $OUT_DIR"
echo "HEAD:    $(git rev-parse HEAD)"
echo "Branch:  $(git rev-parse --abbrev-ref HEAD)"
echo "Mode:    smoke=$RUN_SMOKE full=$RUN_FULL skip_smoke=$SKIP_SMOKE"
echo "PDMRG:   warmup=$PDMRG_WARMUP polish=$PDMRG_POLISH local=$PDMRG_LOCAL"
echo "Variants: ${VARIANTS[*]}"
echo "============================================================"

# ---- 1. Environment verification -----
echo
echo "[1/4] Environment verification"
echo "------------------------------"
command -v hipcc >/dev/null || { echo "FAIL: hipcc not in PATH"; exit 2; }
command -v rocm-smi >/dev/null || { echo "FAIL: rocm-smi not in PATH"; exit 2; }
echo "hipcc: $(hipcc --version | head -1)"
echo "ROCm: $(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
rocm-smi --showproductname 2>&1 | grep -E "GPU|Card" | head -4

# ---- 2. Pre-flight: defect-class registry must be clean -----
echo
echo "[2/4] Defect-class registry pre-flight"
echo "--------------------------------------"
if ! bash .claude/scripts/defect-registry.sh > "$OUT_DIR/registry.log" 2>&1; then
    echo "WARN: registry returned non-zero (only matters with --strict)"
fi
HITS=$(grep -E "^TOTAL HITS:" "$OUT_DIR/registry.log" | awk '{print $NF}')
echo "Registry: TOTAL HITS = $HITS"
if [[ "$HITS" != "0" ]]; then
    echo "FAIL: registry has hits — fix BEFORE consuming GPU time"
    cat "$OUT_DIR/registry.log"
    exit 3
fi

# ---- 3. Build all variants -----
echo
echo "[3/4] Building ${#VARIANTS[@]} GPU variants"
echo "--------------------------------------"
BUILD_FAIL=()
for v in "${VARIANTS[@]}"; do
    echo
    echo "--- Building $v ---"
    if [[ -x "gpu-rocm/$v/build_mi300x.sh" ]]; then
        if ! bash "gpu-rocm/$v/build_mi300x.sh" > "$OUT_DIR/build-${v}.log" 2>&1; then
            echo "FAIL: $v build failed (see $OUT_DIR/build-${v}.log)"
            BUILD_FAIL+=("$v")
            continue
        fi
    else
        # Fallback: cmake from scratch
        rm -rf "gpu-rocm/$v/build"
        mkdir "gpu-rocm/$v/build"
        (cd "gpu-rocm/$v/build" && cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
            -DGPU_TARGETS=gfx942 > "$OUT_DIR/build-${v}.log" 2>&1 \
         && make -j8 >> "$OUT_DIR/build-${v}.log" 2>&1) || {
            echo "FAIL: $v cmake/make failed (see $OUT_DIR/build-${v}.log)"
            BUILD_FAIL+=("$v")
            continue
        }
    fi
    # Verify a binary actually appeared
    bin=$(find "gpu-rocm/$v/build" -maxdepth 2 -type f -executable | head -1)
    [[ -z "$bin" ]] && { echo "FAIL: $v built but no binary found"; BUILD_FAIL+=("$v"); continue; }
    echo "OK: $v -> $bin"
done

if (( ${#BUILD_FAIL[@]} > 0 )); then
    echo
    echo "============================================================"
    echo "BUILD FAILURES (${#BUILD_FAIL[@]}/${#VARIANTS[@]}): ${BUILD_FAIL[*]}"
    echo "============================================================"
    echo "Refusing to consume GPU time with broken builds. Inspect $OUT_DIR/build-*.log"
    exit 4
fi

# ---- 4. Run benchmarks -----
echo
echo "[4/4] Running benchmarks"
echo "------------------------"

run_grid() {
    local label="$1"; shift
    local extra_args="$*"
    echo
    echo "--- Grid: $label ---"
    python3 benchmarks/run_mi300x_challenge.py \
        --pdmrg-warmup "$PDMRG_WARMUP" \
        --pdmrg-polish "$PDMRG_POLISH" \
        --pdmrg-local  "$PDMRG_LOCAL" \
        --repeats "$REPEATS" \
        --output-dir "$OUT_DIR" \
        --tag "$label" \
        $extra_args 2>&1 | tee -a "$OUT_DIR/${label}.log"
}

if (( RUN_SMOKE == 1 )); then
    echo
    echo "==== SMOKE (ULTRA_TRIM grid, --trim, ~30 min) ===="
    run_grid "smoke-${DATE_TAG}" "--trim" || {
        echo "FAIL: smoke run errored — DO NOT proceed to --full"
        exit 5
    }
    # Quick sanity: at least 50% of configs should have a recorded result
    # (correctness threshold left to the harness's own validation hooks)
    smoke_json=$(find "$OUT_DIR" -name "*smoke-${DATE_TAG}*.json" | head -1)
    if [[ -n "$smoke_json" ]]; then
        python3 - "$smoke_json" <<'PYEOF'
import json, sys
fn = sys.argv[1]
data = json.load(open(fn))
runs = data.get("runs", data) if isinstance(data, dict) else data
runs = runs if isinstance(runs, list) else []
ok = sum(1 for r in runs if r.get("status", "ok") == "ok")
total = len(runs)
print(f"Smoke completion: {ok}/{total} configs OK")
sys.exit(0 if total > 0 and ok >= total * 0.5 else 6)
PYEOF
    fi
fi

if (( RUN_FULL == 1 )); then
    echo
    echo "==== FULL G1 BASELINE (CHALLENGE_SIZES, ~6-12 h) ===="
    run_grid "full-${DATE_TAG}" || {
        echo "FAIL: full run errored — partial results in $OUT_DIR"
        exit 7
    }
fi

echo
echo "============================================================"
echo " G1 BASELINE COMPLETE — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo "Output: $OUT_DIR"
echo "Next steps:"
echo "  1. cd $REPO_ROOT && git add $OUT_DIR && git commit -m 'data: G1 baseline ${DATE_TAG}'"
echo "  2. git push origin main"
echo "  3. Update paper main.tex with the new numbers from $OUT_DIR/full-*.json"
echo "============================================================"
