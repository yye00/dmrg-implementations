#!/usr/bin/env python3
"""
Uniformity audit for GPU DMRG variants.

Verifies that all -gpu and -gpu-opt variants carry the same optimization
scaffolding (GpuOpts, PhaseTimer, flag members, timer members, env-var
hooks). -base variants are explicitly excluded — they're meant to be
bare-bones reference implementations.

Run:
    python benchmarks/verify_uniformity.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "gpu-rocm"

# Variants that MUST carry the optimization scaffolding
PRIMARY_VARIANTS = ["dmrg-gpu", "dmrg2-gpu", "pdmrg-gpu"]
OPT_VARIANTS     = ["dmrg-gpu-opt", "dmrg2-gpu-opt", "pdmrg-gpu-opt"]

# Variants that must NOT carry advanced opts (explicit exclusion)
BASE_VARIANTS    = ["dmrg-gpu-base", "dmrg2-gpu-base", "pdmrg-gpu-base"]

# Required features: each is a (label, regex) pair checked in the variant's
# header + impl files (case-sensitive, any hit counts).
REQUIRED_FEATURES = [
    ("include_gpu_opts",    r'#include\s+"\.\./\.\./common/gpu_opts\.h"'),
    ("GpuOpts_member",      r'\bGpuOpts\s+opts_'),
    ("opts_accessor",       r'GpuOpts&\s+opts\(\s*\)'),
    ("opts_load_from_env",  r'opts_\.load_from_env\(\s*\)'),
    ("opts_print",          r'opts_\.print\('),
    ("init_timers_decl",    r'void\s+init_timers\(\s*\)'),
    ("report_timers_decl",  r'void\s+report_timers\(\s*\)'),
    ("PhaseTimer_lanczos",  r'PhaseTimer\s+t_lanczos_'),
    ("PhaseTimer_heff",     r'PhaseTimer\s+t_apply_heff_'),
    ("PhaseTimer_svd",      r'PhaseTimer\s+t_svd_'),
    ("device_k_use",        r'opts_\.device_k\b'),
    ("lanczos_fixed_use",   r'opts_\.lanczos_fixed\b'),
]

# Features that MUST NOT appear in -base variants
FORBIDDEN_IN_BASE = [
    ("include_gpu_opts", r'#include\s+"\.\./\.\./common/gpu_opts\.h"'),
    ("GpuOpts_member",   r'\bGpuOpts\s+opts_'),
]

# Features that MUST NOT appear anywhere in -gpu or -gpu-opt (user said
# Newton-Schulz is deprecated)
FORBIDDEN_EVERYWHERE = [
    ("newton_schulz",       r'\bnewton_schulz\b'),
    ("use_ns_split",        r'\buse_ns_split_?\b'),
    ("d_ns_U_",             r'\bd_ns_U_\b'),
]


def files_for(variant: str) -> list:
    src = ROOT / variant / "src"
    if not src.exists():
        return []
    return sorted(src.glob("*.h")) + sorted(src.glob("*.cpp"))


def scan(files, pattern):
    compiled = re.compile(pattern)
    for f in files:
        try:
            if compiled.search(f.read_text(errors="ignore")):
                return str(f.relative_to(ROOT))
        except Exception:
            continue
    return None


def audit():
    failures = []
    print("=" * 70)
    print("GPU DMRG variant uniformity audit")
    print("=" * 70)

    # Primary + opt tier: must have ALL required features
    for tier_name, variants in [("PRIMARY", PRIMARY_VARIANTS),
                                 ("OPT",     OPT_VARIANTS)]:
        print(f"\n[{tier_name} tier — must carry scaffolding]")
        for v in variants:
            files = files_for(v)
            if not files:
                print(f"  {v}: MISSING (no src/ directory)")
                failures.append(f"{v}: missing")
                continue
            missing = []
            for label, regex in REQUIRED_FEATURES:
                if scan(files, regex) is None:
                    missing.append(label)
            if missing:
                print(f"  {v}: MISSING {len(missing)}/{len(REQUIRED_FEATURES)}")
                for m in missing:
                    print(f"      - {m}")
                failures.append(f"{v}: missing {','.join(missing)}")
            else:
                print(f"  {v}: OK ({len(REQUIRED_FEATURES)}/{len(REQUIRED_FEATURES)})")

    # Base tier: must NOT have scaffolding
    print(f"\n[BASE tier — must NOT carry scaffolding]")
    for v in BASE_VARIANTS:
        files = files_for(v)
        if not files:
            print(f"  {v}: (skip — no src/)")
            continue
        violations = []
        for label, regex in FORBIDDEN_IN_BASE:
            hit = scan(files, regex)
            if hit:
                violations.append(f"{label} in {hit}")
        if violations:
            print(f"  {v}: VIOLATION")
            for v2 in violations:
                print(f"      - {v2}")
            failures.append(f"{v}: unexpected {violations}")
        else:
            print(f"  {v}: OK (clean)")

    # Newton-Schulz: must be gone from all -gpu and -gpu-opt
    print(f"\n[Newton-Schulz — must be absent from all -gpu and -gpu-opt]")
    for v in PRIMARY_VARIANTS + OPT_VARIANTS:
        files = files_for(v)
        if not files:
            continue
        ns_hits = []
        for label, regex in FORBIDDEN_EVERYWHERE:
            hit = scan(files, regex)
            if hit:
                ns_hits.append(f"{label} in {hit}")
        if ns_hits:
            print(f"  {v}: NS STILL PRESENT")
            for h in ns_hits:
                print(f"      - {h}")
            failures.append(f"{v}: NS residue")
        else:
            print(f"  {v}: OK (NS-free)")

    print("\n" + "=" * 70)
    if failures:
        print(f"AUDIT FAILED: {len(failures)} issue(s)")
        for f in failures:
            print(f"  * {f}")
        return 1
    print("AUDIT PASSED: all variants uniform within tier")
    return 0


if __name__ == "__main__":
    sys.exit(audit())
