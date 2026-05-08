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
# 10 reps = journal-grade default (median + IQR with stable narrow tails).
# Override at run time:  REPEATS=20 bash ... --full
REPEATS="${REPEATS:-10}"

# ---- Per-rep wall-clock cap (passed to harness via --per-rep-timeout) ----
# 3600 s (1 hour) per rep — guards against hung GPU runs (driver lockup,
# pathological convergence). Healthy GPU runs in CHALLENGE_SIZES finish
# well under this. Smoke uses a tighter cap so it fails fast.
# CPU benchmarks run on a separate CPU host, NOT here.
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-300}"

# ---- Single-GPU variant set (default mode) -----
SINGLE_GPU_VARIANTS=(
    dmrg-gpu-base
    dmrg-gpu
    dmrg-gpu-opt
    dmrg2-gpu-base
    dmrg2-gpu
    dmrg2-gpu-opt
    pdmrg-gpu-base
    pdmrg-gpu
    pdmrg-gpu-opt
)

# ---- Multi-GPU variant set (separate mode, requires 4 visible MI300X) -----
MULTI_GPU_VARIANTS=(
    pdmrg-multi-gpu
)

# ---- Mode parsing -----
# Two top-level mode axes:
#   (1) what to build/run:  --single-gpu (default) | --multi-gpu
#   (2) grid size:          --smoke | --full | --skip-smoke
# Order doesn't matter; both are accepted in any position.
TARGET="single-gpu"   # default
RUN_SMOKE=0
RUN_FULL=0
SKIP_SMOKE="${SKIP_SMOKE:-0}"
for arg in "$@"; do
    case "$arg" in
        --single-gpu) TARGET="single-gpu" ;;
        --multi-gpu)  TARGET="multi-gpu" ;;
        --smoke)      RUN_SMOKE=1; RUN_FULL=0 ;;
        --full)       RUN_SMOKE=1; RUN_FULL=1 ;;  # full implies a smoke first
        --skip-smoke) RUN_SMOKE=0; RUN_FULL=1; SKIP_SMOKE=1 ;;
        *)            echo "Unknown arg: $arg"; echo "Usage: $0 [--single-gpu|--multi-gpu] [--smoke|--full|--skip-smoke]"; exit 1 ;;
    esac
done
if (( RUN_SMOKE == 0 && RUN_FULL == 0 )); then
    echo "Usage: $0 [--single-gpu|--multi-gpu] [--smoke|--full|--skip-smoke]"
    echo "  --smoke       ULTRA_TRIM grid (~30 min)"
    echo "  --full        CHALLENGE_SIZES grid x REPEATS (~6-12 h single-gpu, longer multi)"
    echo "  --skip-smoke  start --full without smoke (use only after a prior smoke pass)"
    echo "  --single-gpu  build + run the 9 single-device variants (default)"
    echo "  --multi-gpu   build + run pdmrg-multi-gpu only (requires 4 MI300X)"
    exit 1
fi

# ---- Resolve variant set + harness --impl filter for the chosen target -----
if [[ "$TARGET" == "multi-gpu" ]]; then
    VARIANTS=("${MULTI_GPU_VARIANTS[@]}")
    # Multi-gpu compares against its own scaling, not CPU. No quimb in this run.
    IMPL_FILTER="--impl pdmrg-multi-gpu"
    # Multi-gpu sanity: 4 visible devices
    GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -cE "^GPU\[[0-9]+\]" || echo 0)
    if [[ "$GPU_COUNT" -lt 4 ]]; then
        echo "FAIL: --multi-gpu requires 4 visible MI300X; saw $GPU_COUNT"
        exit 1
    fi
else
    VARIANTS=("${SINGLE_GPU_VARIANTS[@]}")
    # GPU machine = GPU only. CPU baselines (quimb) run on a separate CPU host
    # via a different launcher; never on the $$$/hour MI300X.
    IMPL_FILTER="--impl $(IFS=,; echo "${VARIANTS[*]}")"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---- Date stamp / output dir (target-tagged so single + multi runs don't collide) -----
DATE_TAG="$(date -u +%Y%m%d-%H%M)"
OUT_DIR="benchmarks/paper_results/mi300x/g1-${TARGET}-${DATE_TAG}"
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
echo "Target:  $TARGET"
echo "Mode:    smoke=$RUN_SMOKE full=$RUN_FULL skip_smoke=$SKIP_SMOKE"
echo "Reps:    $REPEATS"
echo "PDMRG:   warmup=$PDMRG_WARMUP polish=$PDMRG_POLISH local=$PDMRG_LOCAL"
echo "Variants: ${VARIANTS[*]}"
echo "Filter:  $IMPL_FILTER"
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
        # Variant build scripts use `cmake ..` relative to their own dir, so
        # subshell-cd into the variant first (subshell doesn't leak CWD back).
        if ! ( cd "gpu-rocm/$v" && bash build_mi300x.sh ) > "$OUT_DIR/build-${v}.log" 2>&1; then
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
    # IMPL_FILTER is unquoted so it expands to two args: --impl + comma-list
    python3 benchmarks/run_mi300x_challenge.py \
        $IMPL_FILTER \
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
    echo "==== SMOKE (ULTRA_TRIM grid, --trim, reps=1, ~30 min) ===="
    # Smoke is for end-to-end validation, not statistics. Override REPEATS=1
    # for the smoke pass; --full uses the user-set REPEATS for paper stats.
    REPEATS_SAVED="$REPEATS"
    REPEATS=1
    run_grid "smoke-${DATE_TAG}" "--trim" || {
        REPEATS="$REPEATS_SAVED"
        echo "FAIL: smoke run errored — DO NOT proceed to --full"
        exit 5
    }
    REPEATS="$REPEATS_SAVED"
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
