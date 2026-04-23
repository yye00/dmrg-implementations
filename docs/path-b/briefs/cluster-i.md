# Cluster I: AI-disclosure paragraph, .dev/ cleanup, per-variant build scripts, README fixes

Ground truth: docs/PATH_B_GROUND_TRUTH.md (locked at commit 6f45533)
Planner output SHA (source): a87fa7895ccd0d50a.output
Date planned: 2026-04-23

---

Verified. I have everything I need. Writing the report now.

# DEFECT CLUSTER I — Reproducibility, CPC Library, Repo Cleanup, AI Disclosure

## 1. AI-disclosure statement (manuscript)

**Where**: New unnumbered section "Declaration of generative AI and AI-assisted technologies in the writing process" placed immediately before the Acknowledgements (Elsevier policy, in force since 2023). Also: remove `Claude <noreply@anthropic.com>` from author/affiliation block — Elsevier explicitly forbids listing AI as an author.

**Exact text** (drop-in):

> During the preparation of this work the author(s) used Anthropic Claude (models Opus 4.6/4.7) for (a) drafting and revising portions of the manuscript prose, including the rewrite captured in commit `40af7bf` and the CPC submission preparation (`bb8bcd0`); and (b) generating, refactoring, and reviewing source code in the `gpu-rocm/` and `cpu/` trees. Approximately 38% of the 113 git commits in the public repository (43 commits) were authored by an AI assistant operating under human direction; all such commits were reviewed and accepted by the human author(s) before merge. After using these tools, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the publication.

**Decision on co-authorship**: Remove "Claude" as co-author. Elsevier policy: "AI tools cannot be listed as authors […] cannot take responsibility for the submitted work." Replace with the disclosure paragraph above and (optional) an Acknowledgement: "We thank Anthropic Claude for code-generation and editorial assistance."

## 2. Repo cleanup

**Recommendation: `git mv` to `.dev/` (kept under version control, hidden from casual browsing) for files that document development history; `git rm` for pure noise.**

```bash
# .dev/prompts/    (kept for provenance; LLM workflow inputs)
git mv paper_prompt.md                              .dev/prompts/root_paper_prompt.md
git mv paper/PAPER_PROMPT.md                        .dev/prompts/
git mv paper/REWRITE_PROMPT.md                      .dev/prompts/
git mv paper/REVIEWER_PROMPT.md                     .dev/prompts/
git mv gpu-rocm/pdmrg-gpu/BOUNDARY_MERGE_FIX_PROMPT.md .dev/prompts/
git mv gpu-rocm/pdmrg-gpu/OPTIMIZATION_REPORT.md    .dev/reports/

# .dev/a2dmrg-history/    (LLM run logs — keep one copy, scrub)
git mv cpu/a2dmrg/claude-progress.txt   .dev/a2dmrg-history/
git mv cpu/a2dmrg/TASK_ORIENTATION.md   .dev/a2dmrg-history/
git mv cpu/a2dmrg/PERFORMANCE_FINDINGS.txt .dev/a2dmrg-history/
git rm cpu/a2dmrg/test{35,38,40}_output.txt cpu/a2dmrg/benchmark_output.txt

# Reorganise tests
mkdir -p cpu/a2dmrg/tests
git mv cpu/a2dmrg/test_*.py cpu/a2dmrg/tests/
# add cpu/a2dmrg/tests/__init__.py and a conftest.py if pytest needs sys.path
```

Add a top-level `.dev/README.md` reading: "Development artefacts retained for transparency. Not part of the published software; not built, not tested, not part of the CPC submission."

## 3. Per-variant build scripts

**Recommend: per-variant `build_mi300x.sh` (one per variant directory) PLUS a thin `gpu-rocm/build_all.sh` wrapper.** Per-variant scripts let reviewers build only what they need and survive future divergence in flags. The wrapper is a 20-line `for d in dmrg-gpu dmrg-gpu-opt dmrg2-gpu dmrg2-gpu-opt pdmrg-gpu pdmrg-gpu-opt; do (cd "$d" && ./build_mi300x.sh) || exit 1; done` plus a `--clean` flag. Copy `gpu-rocm/dmrg-gpu/build_mi300x.sh` as the template; per-variant edits are typically only `BIN_NAME`, source list, and (for `-opt`) the `-DUSE_BLOCK_DAVIDSON` macro (currently hardcoded but should at least be visible in the build line for auditability). 6 build scripts × ~50 LOC + 1 wrapper.

## 4. README rewrite plan

Replace lines 13, 83, 85, 120-122, 133-137 with a working Quick Start:

```
## Quick Start (MI300X)
1. ./gpu-rocm/build_all.sh          # builds all 6 GPU binaries
2. cd benchmarks/scripts
3. python run_mi300x_challenge.py --variant dmrg-gpu --quick   # ~5 min smoke test
4. python run_mi300x_challenge.py --variant dmrg-gpu           # full grid (~hours)
Outputs land in benchmarks/results/mi300x/<timestamp>/

## Reproducing the paper
make -C paper figures && make -C paper      # regenerates fig6 + main.pdf
See REPRODUCIBILITY.md for binary SHAs and commit pins (Tables 4-7).
```

Strike: `heisenberg_benchmark.py`, `heisenberg_long_benchmark.py`, `cpu/pdmrg/venv/bin/python run_pdmrg_np1.py`, "PDMRG-OPT: spec only" (1032 LOC exists), and the false "Boundary Merge Disabled" caveat. Keep the np=1→quimb fallback caveat (true in CPU only). Add an explicit "Known divergences from paper" section pointing reviewers to the audit appendix.

## 5. Figures

`fig1`-`fig5` are orphans; `fig6` is the only one in `main.tex`. Two options:

- **(Recommended) Delete fig1-5** (`git rm paper/figures/fig{1,2,3,4,5}_*.pdf`) and remove `generate_figures.py` from `paper/Makefile`. The orphans imply unfinished work to a reviewer.
- Alternatively, integrate them into the manuscript (none of the existing analysis appears to need them).

Patch `paper/Makefile`:
```
figures: figures/fig6_ablation.pdf
figures/fig6_ablation.pdf: ../benchmarks/scripts/generate_ablation_figure.py
	cd ../benchmarks/scripts && python generate_ablation_figure.py --out ../../paper/figures/fig6_ablation.pdf
```

## 6. LOC count correction

Run `cloc --exclude-dir=.dev,build,results --include-lang=C++,C/C++ Header,Python,HIP gpu-rocm cpu` to get fresh numbers, then in Program Summary replace "~12 kLOC C++, ~6 kLOC Python" with the audited counts (per ground-truth: ~32 kLOC C++/HIP and ~27-41 kLOC Python — pin the exact number from `cloc` and quote the command in a footnote so future reviewers can reproduce).

## 7. License hygiene

- Root `LICENSE`: replace "DMRG Implementations Contributors" with the actual copyright holder (institution + lead author name) once §8 is filled.
- 3 sub-package `LICENSE` files: same fix; 2024 → 2024-2026.
- Add top-level `NOTICE`:

```
This software incorporates or depends on:
  - quimb         Apache-2.0   https://github.com/jcmgray/quimb
  - OpenBLAS      BSD-3-Clause https://github.com/OpenMathLib/OpenBLAS
  - rocBLAS/rocSOLVER  MIT     https://github.com/ROCm/rocBLAS
  - hipBLAS       MIT          https://github.com/ROCm/hipBLAS
  - pybind11      BSD-3-Clause https://github.com/pybind/pybind11
All listed licenses are compatible with this project's MIT license.
```

## 8. Affiliation + corresponding author

`main.tex:27` — replace with concrete values, e.g.:

```latex
\affiliation[inst1]{
  organization={<Department>, <University>},
  addressline={<street>}, city={<city>},
  postcode={<zip>}, country={<country>}}
\cortext[cor1]{Corresponding author}
\ead{<author@domain>}            % attach to \author[inst1]{...}\corref{cor1}
```

CPC will desk-reject without a corresponding-author email.

## 9. Undocumented GPU directories

`pdmrg-multi-gpu/`, `radam-gpu/`, `rlbfgs-gpu/` are not in the paper, README, or build wrapper. **Recommend: move to `gpu-rocm/experimental/` with a one-line README per directory** ("Multi-GPU PDMRG prototype, not benchmarked in this paper"). Removing them outright loses honest WIP signal; leaving them at the same level as the published variants implies they were evaluated. Same treatment for the `*-base` variants (`dmrg-gpu-base`, etc.) if they are pre-optimisation snapshots — either document explicitly as "reference baseline preserved for diff" or move under `experimental/`.

## 10. Version tag + reproducibility checklist

```bash
git tag -a v1.0.0 -m "CPC submission, paper rev 2026-04-23"
echo "1.0.0" > VERSION
```

Add `REPRODUCIBILITY.md` with: (a) per-table provenance pointer (commit SHA, JSON path, binary SHA — see Cluster H); (b) `make smoke` target running `dmrg-gpu` on L=16, χ=64 (≤2 min, runnable on a single MI200/MI300 or — via a CPU fallback build — any x86 box); (c) `EXPECTED_OUTPUT.md` showing the smoke test's expected ground-state energy, sweep timing band, and convergence trace, plus tolerance ("E0 within 1e-9 of -7.142296..." etc.).

Document `uv.lock` in README: `uv sync` for uv users; `pip install -e .` from `pyproject.toml` for pip users.

## 11. Effort + dependencies

| Item | Effort | Depends on |
|------|------:|------------|
| AI disclosure paragraph + author block edit (§1, §8) | 0.5 d | author/affiliation decision |
| Repo cleanup script + `.dev/` move (§2) | 0.5 d | none |
| 6 build scripts + wrapper (§3) | 0.5 d | none (template exists) |
| README rewrite (§4) | 1 d | §3 (build paths), §10 (smoke test) |
| Figures Makefile fix + orphan delete (§5) | 0.25 d | none |
| LOC recount footnote (§6) | 0.25 d | `cloc` install |
| LICENSE/NOTICE (§7) | 0.25 d | §1 (author identity) |
| Document/move experimental GPU dirs (§9) | 0.25 d | none |
| Tag v1.0 + REPRODUCIBILITY.md + smoke test (§10) | 1.5 d | §3, §4; CPU fallback build for smoke |
| **Total** | **~5 days** | sequencing: §1,§2,§3,§5,§6,§7,§9 in parallel; §4 after §3; §10 last |

External dependencies: `cloc` (LOC count), Anthropic policy doc URL for the disclosure footnote, decision on whether to keep `pdmrg-multi-gpu` etc. for a future paper, and (critically) the human author's real name/affiliation/email — currently nothing in the repo identifies them.

Relevant files: `/home/user/dmrg-implementations/paper/main.tex` (line 27 affiliation, disclosure insert point), `/home/user/dmrg-implementations/paper/Makefile`, `/home/user/dmrg-implementations/paper/figures/`, `/home/user/dmrg-implementations/README.md`, `/home/user/dmrg-implementations/LICENSE`, `/home/user/dmrg-implementations/cpu/a2dmrg/`, `/home/user/dmrg-implementations/gpu-rocm/dmrg-gpu/build_mi300x.sh` (template), `/home/user/dmrg-implementations/paper_prompt.md`, `/home/user/dmrg-implementations/gpu-rocm/{pdmrg-multi-gpu,radam-gpu,rlbfgs-gpu}/`.
