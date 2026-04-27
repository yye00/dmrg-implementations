#!/usr/bin/env python3
"""
benchmarks/run_campaign.py — JSON-driven benchmark campaign runner.

A campaign is a single coherent benchmark run defined by one input JSON
config file. The runner reads that config, builds the requested GPU
variants, pins clocks if requested, runs each (variant × model × config × rep)
cell, and writes per-cell JSON output plus an aggregate campaign manifest.

Every output JSON embeds the full input-config snapshot, the input-config
sha256, the git commit, the binary sha256, and a host/ROCm/env provenance
block, so a result is fully self-describing.

USAGE
-----
    python3 benchmarks/run_campaign.py --config benchmarks/campaigns/foo.json
    python3 benchmarks/run_campaign.py --config foo.json --dry-run
    python3 benchmarks/run_campaign.py --config foo.json --build-only
    python3 benchmarks/run_campaign.py --config foo.json --force  # re-run cells

DESIGN
------
1. Idempotent: a cell whose result JSON already exists is skipped unless
   --force. Re-running an interrupted campaign picks up where it left off.
2. No sed-patches, no env-var overrides of script arrays. The config file
   is the single source of truth for what was run.
3. Reuses benchmarks/lib/{provenance,stats}.py — no duplication.
4. Schema-versioned input + output for forward-compat.

INPUT JSON SCHEMA (schema_version=1)
------------------------------------
See benchmarks/campaigns/g1_baseline_rebench.json for a worked example.

Top-level keys:
  schema_version  : int, must equal 1
  campaign        : {name, tag_template, description}
  build           : {rebuild, variants[], rocm_path, build_script,
                     binary_template}
  clocks          : {pin, perf_level, sclk}     # optional; skipped if absent
  run             : {models[], reps, configs{model: [config_objs]},
                     extra_args_per_variant{}, timeout_sec}
  correctness     : {enabled, tol, config}      # optional
  output          : {dir, include_provenance, manifest}

A "config_obj" is {L, chi, sweeps, nmax}.

The dir field in `output` may contain {tag} which is rendered from
campaign.tag_template (which itself may contain {date} → YYYYMMDD).

OUTPUT
------
  <output.dir>/campaign_manifest.json     (aggregate, always)
  <output.dir>/<variant>/<model>/L<L>_chi<chi>.json   (per-cell)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Path setup so we can import benchmarks/lib/*.
_BENCH_LIB = Path(__file__).resolve().parent / "lib"
sys.path.insert(0, str(_BENCH_LIB))
from provenance import provenance_block, binary_info  # noqa: E402

try:
    from stats import wilson_ci  # noqa: E402
except ImportError:
    wilson_ci = None  # stats module is optional for the runner itself.


SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent

# Output parsers — same regexes as bench_dmrg_gpu_ablate.py for consistency.
ENERGY_RE = re.compile(r"Final energy:\s+([-\d.eE+]+)")
WALL_RE   = re.compile(r"Total wall time:\s+([\d.]+)\s*s")


# ─── Config validation ────────────────────────────────────────────────────────

class ConfigError(ValueError):
    pass


def _require(d: dict, key: str, where: str):
    if key not in d:
        raise ConfigError(f"missing required key '{key}' in {where}")
    return d[key]


def validate_config(cfg: dict) -> None:
    """Raise ConfigError on any structural problem. Run before any work starts."""
    sv = _require(cfg, "schema_version", "top-level")
    if sv != SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version={sv} not supported (this runner: {SCHEMA_VERSION})"
        )

    camp = _require(cfg, "campaign", "top-level")
    _require(camp, "name", "campaign")
    _require(camp, "tag_template", "campaign")

    build = _require(cfg, "build", "top-level")
    variants = _require(build, "variants", "build")
    if not isinstance(variants, list) or not variants:
        raise ConfigError("build.variants must be a non-empty list")
    _require(build, "binary_template", "build")
    _require(build, "build_script", "build")

    run = _require(cfg, "run", "top-level")
    models = _require(run, "models", "run")
    reps = _require(run, "reps", "run")
    if not isinstance(reps, int) or reps < 1:
        raise ConfigError("run.reps must be a positive integer")
    configs_by_model = _require(run, "configs", "run")
    for m in models:
        if m not in configs_by_model:
            raise ConfigError(f"run.configs missing entry for model '{m}'")
        for i, c in enumerate(configs_by_model[m]):
            for k in ("L", "chi", "sweeps", "nmax"):
                if k not in c:
                    raise ConfigError(
                        f"run.configs[{m}][{i}] missing key '{k}'"
                    )

    out = _require(cfg, "output", "top-level")
    _require(out, "dir", "output")


def render_tag(tag_template: str) -> str:
    return tag_template.format(date=datetime.now(timezone.utc).strftime("%Y%m%d"))


def render_output_dir(template: str, tag: str) -> Path:
    rendered = template.format(tag=tag)
    p = Path(rendered)
    return p if p.is_absolute() else REPO_ROOT / p


def hash_input_config(cfg: dict) -> str:
    """Stable sha256 of the input config (sorted keys, no whitespace)."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ─── Build phase ──────────────────────────────────────────────────────────────

def variant_binary_path(build_cfg: dict, variant: str) -> Path:
    """Render binary_template for a variant. Convention: hyphens → underscores."""
    template = build_cfg["binary_template"]
    rendered = template.format(
        variant=variant,
        variant_underscored=variant.replace("-", "_"),
    )
    p = Path(rendered)
    return p if p.is_absolute() else REPO_ROOT / p


def variant_build_dir(build_cfg: dict, variant: str) -> Path:
    """Where the build_script lives for a variant. Convention: gpu-rocm/<variant>/."""
    return REPO_ROOT / "gpu-rocm" / variant


def build_variant(build_cfg: dict, variant: str, dry_run: bool = False) -> dict:
    """Invoke build_mi300x.sh in the variant's directory. Return build record."""
    script = build_cfg["build_script"]
    bdir = variant_build_dir(build_cfg, variant)
    binary = variant_binary_path(build_cfg, variant)
    record = {
        "variant": variant,
        "build_dir": str(bdir.relative_to(REPO_ROOT)),
        "build_script": script,
        "binary_path": str(binary.relative_to(REPO_ROOT)),
        "build_started_utc": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        record["dry_run"] = True
        record["binary_exists"] = binary.exists()
        return record

    if not (bdir / script).exists():
        raise FileNotFoundError(
            f"build script not found: {bdir / script}"
        )

    env = os.environ.copy()
    if build_cfg.get("rocm_path"):
        env["ROCM_PATH"] = build_cfg["rocm_path"]
    print(f"  [build] {variant} via {script} ...", flush=True)
    proc = subprocess.run(
        ["bash", script],
        cwd=bdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    record["build_completed_utc"] = datetime.now(timezone.utc).isoformat()
    record["returncode"] = proc.returncode
    record["stderr_tail"] = "\n".join(proc.stderr.splitlines()[-20:])
    if proc.returncode != 0:
        raise RuntimeError(
            f"build failed for {variant} (rc={proc.returncode}):\n"
            f"{record['stderr_tail']}"
        )
    if not binary.exists():
        raise FileNotFoundError(
            f"build of {variant} returned 0 but binary not at {binary}"
        )
    record["binary_info"] = binary_info(str(binary))
    return record


# ─── Clock pinning ────────────────────────────────────────────────────────────

def pin_clocks(clocks_cfg: dict | None, capture_dir: Path, phase: str) -> dict:
    """Apply clock pin (best-effort) and capture rocm-smi snapshot. Phase = 'pre'|'post'."""
    if not clocks_cfg:
        return {"skipped": True, "reason": "no clocks config"}
    snapshot = {}
    if phase == "pre" and clocks_cfg.get("pin"):
        for cmd in (
            ["rocm-smi", "--setperflevel", clocks_cfg.get("perf_level", "high")],
            ["rocm-smi", "--setsclk", str(clocks_cfg.get("sclk", 7))],
            ["rocm-smi", "--setfan", "100"],
        ):
            r = subprocess.run(cmd, capture_output=True, text=True)
            snapshot.setdefault("pin_attempts", []).append({
                "cmd": " ".join(cmd),
                "rc": r.returncode,
                "stderr_tail": "\n".join(r.stderr.splitlines()[-3:]),
            })
    capture_dir.mkdir(parents=True, exist_ok=True)
    smi_out = capture_dir / f"rocm_smi_{phase}.txt"
    proc = subprocess.run(
        ["rocm-smi", "--showall"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        proc = subprocess.run(["rocm-smi"], capture_output=True, text=True)
    smi_out.write_text(proc.stdout)
    snapshot["smi_capture_path"] = str(smi_out.relative_to(REPO_ROOT))
    snapshot["smi_capture_rc"] = proc.returncode
    return snapshot


# ─── Run phase ────────────────────────────────────────────────────────────────

def build_argv(binary: Path, model: str, config: dict, extra_args: list) -> list:
    """Construct binary CLI invocation."""
    argv = [
        str(binary),
        str(config["L"]),
        str(config["chi"]),
        str(config["sweeps"]),
        "--nmax", str(config["nmax"]),
        "--quiet",
    ]
    if model == "josephson":
        argv.append("--josephson")
    elif model == "tfim":
        argv.append("--tfim")
    # heisenberg is the default; no flag needed.
    if extra_args:
        argv.extend(extra_args)
    return argv


def run_one_rep(binary: Path, model: str, config: dict, extra_args: list,
                timeout_sec: int) -> dict:
    """Single rep. Captures wall time, energy, return code, stderr tail."""
    argv = build_argv(binary, model, config, extra_args)
    t0 = time.time()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        wall = time.time() - t0
        tail = e.stderr or ""
        if isinstance(tail, bytes):
            tail = tail.decode(errors="replace")
        return {
            "wall_s": None,
            "wallclock_s": wall,
            "energy": None,
            "rc": -1,
            "timeout": True,
            "valid": False,
            "stderr_tail": "[TIMEOUT %ds]\n%s" % (
                timeout_sec, "\n".join(tail.splitlines()[-10:])
            ),
        }
    wall = time.time() - t0
    energy = None
    reported_wall = None
    for line in proc.stdout.splitlines() + proc.stderr.splitlines():
        if m := ENERGY_RE.search(line):
            energy = float(m.group(1))
        if m := WALL_RE.search(line):
            reported_wall = float(m.group(1))
    rec = {
        "wall_s": reported_wall if reported_wall is not None else wall,
        "wallclock_s": wall,
        "energy": energy,
        "rc": proc.returncode,
        "timeout": False,
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-10:]),
    }
    rec["valid"] = (rec["rc"] == 0 and rec["energy"] is not None)
    if not rec["valid"]:
        # Crash-filter: invalidate wall_s on bad rep so it cannot pollute medians.
        rec["wall_s"] = None
    return rec


def cell_output_path(out_dir: Path, variant: str, model: str, config: dict) -> Path:
    return out_dir / variant / model / f"L{config['L']}_chi{config['chi']}.json"


def correctness_check(binary: Path, model: str, gate_cfg: dict,
                      timeout_sec: int) -> dict:
    """Single-rep sanity check. Return dict; does not abort."""
    cfg = gate_cfg.get("config", {"L": 8, "chi": 32, "sweeps": 10, "nmax": 2})
    rec = run_one_rep(binary, model, cfg, [], timeout_sec)
    rec["config"] = cfg
    rec["passed"] = rec["valid"]
    return rec


def run_cell(binary: Path, variant: str, model: str, config: dict,
             reps: int, extra_args: list, timeout_sec: int,
             input_cfg: dict, input_cfg_hash: str, tag: str,
             clocks_snapshot: dict, build_record: dict,
             correctness_rec: dict | None) -> dict:
    """Run reps for one (variant, model, config) cell. Return self-describing dict."""
    print(f"  [run] {variant}/{model}/L{config['L']}_chi{config['chi']} × {reps} reps",
          flush=True)
    started = datetime.now(timezone.utc).isoformat()
    raw_reps = []
    for r in range(reps):
        rec = run_one_rep(binary, model, config, extra_args, timeout_sec)
        rec["rep"] = r
        wall_str = "%.2fs" % rec["wall_s"] if rec["wall_s"] is not None else "FAIL"
        print(f"    rep {r}: wall={wall_str} E={rec['energy']} rc={rec['rc']}",
              flush=True)
        raw_reps.append(rec)
    completed = datetime.now(timezone.utc).isoformat()

    walls = [r["wall_s"] for r in raw_reps if r.get("valid") and r["wall_s"] is not None]
    energies = [r["energy"] for r in raw_reps if r.get("valid") and r["energy"] is not None]
    n_attempted = len(raw_reps)
    n_valid = len(walls)

    if n_valid == n_attempted:
        data_quality = "OK"
    elif n_valid >= (n_attempted + 1) // 2:
        data_quality = "DEGRADED"
    else:
        data_quality = "FAILED"

    median_wall = statistics.median(walls) if walls else None
    energy_med  = statistics.median(energies) if energies else None
    energy_max_dev = (max(abs(e - energy_med) for e in energies)
                      if energy_med is not None and len(energies) > 1 else None)

    # Wilson CI on convergence rate (k converged out of n attempted).
    wilson = None
    if wilson_ci is not None:
        lo, hi = wilson_ci(n_valid, n_attempted)
        wilson = {"converged_rate_lo95": lo, "converged_rate_hi95": hi}

    return {
        "schema_version": SCHEMA_VERSION,
        "cell_id": f"{variant}/{model}/L{config['L']}_chi{config['chi']}",
        "campaign": {"name": input_cfg["campaign"]["name"], "tag": tag},
        "input_config_hash": input_cfg_hash,
        "input_config_snapshot": input_cfg,
        "variant": variant,
        "binary_path": str(binary.relative_to(REPO_ROOT)),
        "binary_sha256": build_record.get("binary_info", {}).get("sha256"),
        "model": model,
        "config": config,
        "extra_args": extra_args,
        "reps_attempted": n_attempted,
        "reps_valid": n_valid,
        "data_quality": data_quality,
        "wall_times_sec": walls,
        "energies": energies,
        "median_wall_sec": median_wall,
        "energy_median": energy_med,
        "energy_max_deviation": energy_max_dev,
        "convergence_rate_ci_95": wilson,
        "raw_reps": raw_reps,
        "correctness_check": correctness_rec,
        "clocks_snapshot": clocks_snapshot,
        "provenance": provenance_block(
            repo_root=str(REPO_ROOT),
            script_argv=sys.argv[1:],
            binary_path=str(binary),
        ),
        "started_utc": started,
        "completed_utc": completed,
    }


# ─── Main orchestration ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True,
                    help="path to campaign JSON config")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate, render paths, list cells; do not build or run")
    ap.add_argument("--build-only", action="store_true",
                    help="build variants but do not run the campaign")
    ap.add_argument("--force", action="store_true",
                    help="re-run cells even if their result JSON exists")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(2)

    cfg = json.loads(cfg_path.read_text())
    try:
        validate_config(cfg)
    except ConfigError as e:
        print(f"ERROR: invalid campaign config: {e}", file=sys.stderr)
        sys.exit(2)

    cfg_hash = hash_input_config(cfg)
    tag = render_tag(cfg["campaign"]["tag_template"])
    out_dir = render_output_dir(cfg["output"]["dir"], tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate cells (variant × model × config).
    variants = cfg["build"]["variants"]
    cells = []
    for variant in variants:
        for model in cfg["run"]["models"]:
            for config in cfg["run"]["configs"][model]:
                cells.append((variant, model, config))

    print(f"=== Campaign: {cfg['campaign']['name']} ===")
    print(f"  config:    {cfg_path}")
    print(f"  hash:      {cfg_hash}")
    print(f"  tag:       {tag}")
    print(f"  output:    {out_dir}")
    print(f"  variants:  {variants}")
    print(f"  models:    {cfg['run']['models']}")
    print(f"  reps/cell: {cfg['run']['reps']}")
    print(f"  total cells: {len(cells)}  ({len(cells) * cfg['run']['reps']} reps)")
    print()

    if args.dry_run:
        print("=== Dry-run cell list ===")
        for variant, model, config in cells:
            cell_path = cell_output_path(out_dir, variant, model, config)
            exists = "EXISTS" if cell_path.exists() else "missing"
            print(f"  {variant}/{model}/L{config['L']}_chi{config['chi']:>4} → {cell_path.relative_to(REPO_ROOT)}  [{exists}]")
        print("\nDry-run complete. No work performed.")
        sys.exit(0)

    # Build phase.
    print("=== Build phase ===")
    build_cfg = cfg["build"]
    build_records = {}
    for variant in variants:
        binary = variant_binary_path(build_cfg, variant)
        if binary.exists() and not build_cfg.get("rebuild", False):
            print(f"  [build] {variant}: binary exists at {binary.relative_to(REPO_ROOT)}, skipping (rebuild=false)")
            build_records[variant] = {
                "variant": variant,
                "binary_path": str(binary.relative_to(REPO_ROOT)),
                "binary_info": binary_info(str(binary)),
                "skipped_existing_binary": True,
            }
            continue
        build_records[variant] = build_variant(build_cfg, variant)
        print(f"  [build] {variant}: ok ({build_records[variant]['binary_info']['sha256'][:16]}...)")

    if args.build_only:
        print("\n--build-only: skipping run phase.")
    else:
        # Pre-clock pinning + snapshot.
        print("\n=== Clocks (pre) ===")
        clocks_pre = pin_clocks(cfg.get("clocks"), out_dir, phase="pre")

        # Run phase.
        print("\n=== Run phase ===")
        timeout_sec = cfg["run"].get("timeout_sec", 1800)
        gate_cfg = cfg.get("correctness", {})
        extra_args_per_variant = cfg["run"].get("extra_args_per_variant", {}) or {}

        n_completed = 0
        n_skipped = 0
        n_failed = 0
        cell_records = []
        for variant, model, config in cells:
            cell_path = cell_output_path(out_dir, variant, model, config)
            cell_path.parent.mkdir(parents=True, exist_ok=True)
            if cell_path.exists() and not args.force:
                print(f"  [skip] {cell_path.relative_to(REPO_ROOT)} (exists)")
                cell_records.append({"variant": variant, "model": model, "config": config,
                                     "result_path": str(cell_path.relative_to(REPO_ROOT)),
                                     "status": "skipped_existing"})
                n_skipped += 1
                continue
            binary = variant_binary_path(build_cfg, variant)
            extra_args = extra_args_per_variant.get(variant, [])

            # Per-cell correctness check (cheap, gives us a smoke signal in the JSON).
            correctness_rec = None
            if gate_cfg.get("enabled", False):
                correctness_rec = correctness_check(binary, model, gate_cfg, timeout_sec)

            try:
                result = run_cell(
                    binary=binary,
                    variant=variant,
                    model=model,
                    config=config,
                    reps=cfg["run"]["reps"],
                    extra_args=extra_args,
                    timeout_sec=timeout_sec,
                    input_cfg=cfg,
                    input_cfg_hash=cfg_hash,
                    tag=tag,
                    clocks_snapshot=clocks_pre,
                    build_record=build_records[variant],
                    correctness_rec=correctness_rec,
                )
            except Exception as e:
                print(f"  [error] cell crashed: {e}")
                cell_records.append({"variant": variant, "model": model, "config": config,
                                     "result_path": str(cell_path.relative_to(REPO_ROOT)),
                                     "status": "error", "error": str(e)})
                n_failed += 1
                continue

            cell_path.write_text(json.dumps(result, indent=2, default=str))
            cell_records.append({"variant": variant, "model": model, "config": config,
                                 "result_path": str(cell_path.relative_to(REPO_ROOT)),
                                 "status": "ok",
                                 "data_quality": result["data_quality"],
                                 "median_wall_sec": result["median_wall_sec"]})
            n_completed += 1

        # Post-clock snapshot.
        print("\n=== Clocks (post) ===")
        clocks_post = pin_clocks(cfg.get("clocks"), out_dir, phase="post")

        # Manifest.
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "campaign": cfg["campaign"],
            "tag": tag,
            "input_config_path": str(cfg_path.relative_to(REPO_ROOT))
                                  if cfg_path.is_absolute() and REPO_ROOT in cfg_path.parents
                                  else str(cfg_path),
            "input_config_hash": cfg_hash,
            "input_config_snapshot": cfg,
            "build_records": build_records,
            "clocks_pre": clocks_pre,
            "clocks_post": clocks_post,
            "cells": cell_records,
            "summary": {
                "total_cells": len(cells),
                "completed": n_completed,
                "skipped_existing": n_skipped,
                "failed": n_failed,
            },
            "provenance": provenance_block(
                repo_root=str(REPO_ROOT),
                script_argv=sys.argv[1:],
            ),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        print(f"\n=== Summary ===")
        print(f"  total cells:       {len(cells)}")
        print(f"  completed:         {n_completed}")
        print(f"  skipped existing:  {n_skipped}")
        print(f"  failed:            {n_failed}")
        print(f"  manifest:          {(out_dir / 'campaign_manifest.json').relative_to(REPO_ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
