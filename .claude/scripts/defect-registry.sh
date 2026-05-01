#!/usr/bin/env bash
# Defect-class registry — proactive sweep replacing reactive round-by-round
# orchestration. Scans all in-charter -base/-gpu/-opt/-multi-gpu variants
# for known defect classes (rounds 7-16). Returns non-zero if ANY hit.
#
# Usage:
#   .claude/scripts/defect-registry.sh           # report all hits
#   .claude/scripts/defect-registry.sh --strict  # exit 1 on any hit
#
# A new defect class learned the hard way? Add a class entry below.

set -u
ROOT="${1:-gpu-rocm}"
STRICT="${2:-}"

# In-charter variants (skip radam-gpu, rlbfgs-gpu — out of conformity charter)
VARIANTS=(
    "gpu-rocm/dmrg-gpu-base"
    "gpu-rocm/dmrg-gpu"
    "gpu-rocm/dmrg-gpu-opt"
    "gpu-rocm/dmrg2-gpu-base"
    "gpu-rocm/dmrg2-gpu"
    "gpu-rocm/dmrg2-gpu-opt"
    "gpu-rocm/pdmrg-gpu-base"
    "gpu-rocm/pdmrg-gpu"
    "gpu-rocm/pdmrg-gpu-opt"
    "gpu-rocm/pdmrg-multi-gpu"
)

TOTAL_HITS=0

# scan <name> <regex> [extra-grep-args]
# Reports all hits across the variant set, increments TOTAL_HITS.
scan() {
    local name="$1"
    local pattern="$2"
    shift 2
    local hits
    hits=$(grep -rEn "$pattern" "$@" "${VARIANTS[@]/%//src}" 2>/dev/null | grep -v 'worktrees' || true)
    if [[ -n "$hits" ]]; then
        local count
        count=$(printf '%s\n' "$hits" | wc -l)
        echo
        echo "===================================================================="
        echo "DEFECT CLASS: $name"
        echo "Pattern: $pattern"
        echo "Hits: $count"
        echo "===================================================================="
        printf '%s\n' "$hits"
        TOTAL_HITS=$((TOTAL_HITS + count))
    fi
}

echo "Defect-class registry sweep — $(git -C . rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
echo "Variants scanned: ${#VARIANTS[@]}"

# ----- D1: host pointer-array build (vector<Scalar*> on hot paths) -----
# Acceptable in init/set_mpo paths; NOT acceptable per-call.
scan "D1: host vector<Scalar*> array on hot path" \
     '^[[:space:]]+std::vector<Scalar\*>[[:space:]]+h_[ABC]\b' \
     --include='*_impl.h'

# ----- D2: host pointer-array stack (Scalar* h_X[N]) -----
scan "D2: host Scalar* stack-array on hot path" \
     'Scalar\*[[:space:]]+h_[ABC][0-9]?\[[0-9]' \
     --include='*_impl.h'

# ----- D3: hipMemcpyAsync of host pointer arrays to d_batch_* -----
scan "D3: H2D of host pointer-array (host roundtrip per call)" \
     'hipMemcpyAsync.*h_[ABC][0-9]?\.data\(\).*HostToDevice|hipMemcpyAsync.*h_[ABC][0-9]?,[[:space:]]*[a-z]+[[:space:]]*\*[[:space:]]*sizeof.*HostToDevice' \
     --include='*_impl.h'

# ----- D4: raw rocblas_set_pointer_mode (must use PointerModeGuard RAII) -----
scan "D4: raw rocblas_set_pointer_mode (no RAII)" \
     'rocblas_set_pointer_mode\(' \
     --include='*_impl.h' --include='*.h'

# ----- D5: variant-local PointerModeGuard struct (should use shared) -----
scan "D5: variant-local PointerModeGuard struct" \
     '^struct[[:space:]]+[A-Z][a-zA-Z]*PointerModeGuard\b' \
     --include='*_impl.h' --include='*.h'

# ----- D6: variant-local setup_batch_ptrs kernel (should use shared) -----
scan "D6: variant-local setup_batch_ptrs kernel" \
     '^__global__ void setup_batch_ptrs_' \
     --include='*_impl.h' --include='*.h'

# ----- D7: host LAPACK syev on hot path (must be on-device rocsolver) -----
# Init-time workspace queries and use_cpu_svd_ branch are acceptable.
scan "D7: host lapack_syev (excluding init/use_cpu_svd_)" \
     'lapack_syev\(' \
     --include='*_impl.h'

# ----- D8: host LAPACK gesvd on hot path (must be rocsolver_gesvd) -----
scan "D8: host lapack_gesvd (excluding init/use_cpu_svd_)" \
     'lapack_gesvd\(' \
     --include='*_impl.h'

# ----- D9: missing PhaseTimer instrumentation (declared but never .begin/.end) -----
# This needs a 2-stage check: timers declared in headers minus timers used in impl.
echo
echo "===================================================================="
echo "DEFECT CLASS: D9 — declared PhaseTimers without begin/end sites"
echo "===================================================================="
for variant in "${VARIANTS[@]}"; do
    declared=$(grep -hE '^[[:space:]]+PhaseTimer[[:space:]]+t_[a-z_]+_;' "$variant"/src/*.h 2>/dev/null \
        | grep -oE 't_[a-z_]+_' | sort -u || true)
    [[ -z "$declared" ]] && continue
    while read -r timer; do
        [[ -z "$timer" ]] && continue
        # Check for begin OR end calls in impl
        used=$(grep -rE "${timer}\.(begin|end)\(" "$variant"/src/ 2>/dev/null | wc -l)
        if [[ "$used" -eq 0 ]]; then
            echo "  $variant: $timer declared but no .begin/.end sites — DEAD"
            TOTAL_HITS=$((TOTAL_HITS + 1))
        fi
    done <<< "$declared"
done

# ----- D10: dead set_quiet stub (no test driver caller) -----
echo
echo "===================================================================="
echo "DEFECT CLASS: D10 — set_quiet stubs without test-driver callers"
echo "===================================================================="
for variant in "${VARIANTS[@]}"; do
    has_stub=$(grep -lE 'void[[:space:]]+set_quiet\(bool\)[[:space:]]*\{[[:space:]]*\}' "$variant"/src/*.h 2>/dev/null || true)
    [[ -z "$has_stub" ]] && continue
    callers=$(grep -rE '\.set_quiet\(' "$variant"/src/*.cpp 2>/dev/null | wc -l)
    if [[ "$callers" -eq 0 ]]; then
        echo "  $has_stub: set_quiet stub declared but 0 callers in test driver — DEAD"
        TOTAL_HITS=$((TOTAL_HITS + 1))
    fi
done

# ----- D11: two-half Davidson toggle (setter without ctor gate) -----
# Heuristic: every -opt with set_use_davidson should also have a ctor block
# guarded by lanczos_graph_was_user_enabled_.
echo
echo "===================================================================="
echo "DEFECT CLASS: D11 — Davidson toggle missing ctor-time gate"
echo "===================================================================="
for variant in "${VARIANTS[@]}"; do
    [[ "$variant" != *-opt ]] && continue
    has_setter=$(grep -lE '\bset_use_davidson\b' "$variant"/src/*.h 2>/dev/null || true)
    [[ -z "$has_setter" ]] && continue
    has_ctor_gate=$(grep -E 'opts_\.lanczos_graph[[:space:]]*&&[[:space:]]*use_davidson_' "$variant"/src/*_impl.h 2>/dev/null || true)
    if [[ -z "$has_ctor_gate" ]]; then
        echo "  $variant: set_use_davidson setter present but ctor-time gate MISSING"
        TOTAL_HITS=$((TOTAL_HITS + 1))
    fi
done

# ----- D12: Lanczos host-pointer-mode (host-stack alpha/beta per iter) -----
# Heuristic: a Lanczos that uses &alpha_result, &beta_i (host scalars) instead
# of d_alpha_dev_/d_beta_dev_ + process_alpha/beta_kernel.
scan "D12: Lanczos host-stack alpha/beta scalars (per-iter host roundtrip)" \
     '&alpha_result\b|&beta_i\b' \
     --include='*_impl.h'

# ----- D13: per-iter Step-3 host loop (instead of R3-F1 batched collapse) -----
# Heuristic: Step 3 in apply_heff with a `for (int wp = 0; wp < D; wp++)` AND
# `for (int sp = 0; sp < d; sp++)` AND a Traits::gemm( inside. These should
# use gemm_batched + setup_batch_ptrs_step3_full kernel instead.
echo
echo "===================================================================="
echo "DEFECT CLASS: D13 — Step 3 per-element host loop in apply_heff"
echo "===================================================================="
for variant in "${VARIANTS[@]}"; do
    matches=$(grep -E 'apply_heff' "$variant"/src/*_impl.h 2>/dev/null \
        | head -1 || true)
    [[ -z "$matches" ]] && continue
    # Look inside the impl for the pattern "for (int wp = 0; wp < D; wp++)" with
    # a single Traits::gemm( inside, NOT a gemm_batched / gemm_strided_batched.
    inline=$(awk '
        /apply_heff/{ in_func=1; depth=0 }
        in_func && /\{/{ depth++ }
        in_func && /\}/{ depth--; if (depth<=0) in_func=0 }
        in_func && /for[[:space:]]+\(int wp = 0; wp < D; wp\+\+\)/{ wploop_line=NR; wploop=1 }
        in_func && wploop && /Traits::gemm\(/ && !/gemm_batched|gemm_strided/{
            print FILENAME":"wploop_line": apply_heff per-wp host loop with non-batched gemm";
            wploop=0
        }
    ' "$variant"/src/*_impl.h 2>/dev/null || true)
    if [[ -n "$inline" ]]; then
        echo "$inline"
        TOTAL_HITS=$((TOTAL_HITS + $(printf '%s\n' "$inline" | wc -l)))
    fi
done

# ----- Summary -----
echo
echo "===================================================================="
echo "TOTAL HITS: $TOTAL_HITS"
echo "===================================================================="

if [[ "$STRICT" == "--strict" && "$TOTAL_HITS" -gt 0 ]]; then
    exit 1
fi
exit 0
