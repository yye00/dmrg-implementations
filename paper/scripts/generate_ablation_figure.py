#!/usr/bin/env python3
"""
Generate fig6_ablation.pdf — per-flag speedup heatmap across the six
-gpu/-gpu-opt variants on Josephson L=32, n_max=2 ablation runs.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "benchmarks", "data", "gpu_ablation")
OUT  = os.path.join(REPO, "paper", "figures", "fig6_ablation.pdf")

# Ablation runs (one per variant, picked the most recent valid run).
RUNS = [
    ('20260420T181425Z/dmrg-gpu/results.json',     'dmrg-gpu'),
    ('20260420T181425Z/dmrg2-gpu/results.json',    'dmrg2-gpu'),
    ('20260421T190910Z/pdmrg-gpu/results.json',    'pdmrg-gpu'),
    ('20260421T004212Z/dmrg-gpu-opt/results.json', 'dmrg-gpu-opt'),
    ('20260421T182820Z/dmrg2-gpu-opt/results.json','dmrg2-gpu-opt'),
]

FLAGS = ['DEVICE_K', 'LANCZOS_GRAPH', 'RSVD', 'SPARSE_MPO', 'FUSE_LANCZOS', 'D_PAD']
PROBLEMS = ['josephson_L32_chi128', 'josephson_L32_chi256']
PROBLEM_LABELS = [r'$\chi=128$', r'$\chi=256$']

# Variants whose LANCZOS_GRAPH is force-disabled (Block-Davidson incompatible)
# — show as "n/a" rather than the spurious crash speedup the bench would record.
LG_DISABLED_OPT = {'dmrg-gpu-opt', 'dmrg2-gpu-opt'}
# Variants without device-side SVD truncation kernel
DEVICEK_NA = {'dmrg-gpu-opt', 'dmrg2-gpu-opt'}
# RSVD inapplicable to CPU-LAPACK SVD path (-opt variants)
RSVD_NA = {'dmrg-gpu-opt', 'dmrg2-gpu-opt'}

def collect():
    n_rows = len(RUNS) * len(PROBLEMS)
    matrix = np.full((n_rows, len(FLAGS)), np.nan)
    row_labels = []
    valid = np.zeros((n_rows, len(FLAGS)), dtype=bool)
    for i, (sub, variant) in enumerate(RUNS):
        path = os.path.join(DATA, sub)
        if not os.path.exists(path):
            for p in PROBLEM_LABELS:
                row_labels.append(f"{variant} {p}")
            continue
        d = json.load(open(path))
        for j_p, prob in enumerate(PROBLEMS):
            row = i * len(PROBLEMS) + j_p
            row_labels.append(f"{variant} {PROBLEM_LABELS[j_p]}")
            meds = {r['config']: r.get('median_wall_s') for r in d['results']
                    if r['problem'] == prob}
            base = meds.get('baseline')
            if not base:
                continue
            for j_f, flag in enumerate(FLAGS):
                # Inapplicability checks — leave NaN, label "n/a" later
                if flag == 'DEVICE_K' and variant in DEVICEK_NA:
                    continue
                if flag == 'RSVD' and variant in RSVD_NA:
                    continue
                if flag == 'LANCZOS_GRAPH' and variant in LG_DISABLED_OPT:
                    continue
                w = meds.get(f'only_{flag}')
                if w is None:
                    continue
                spd = base / w
                # Sanity check: flag run must have valid energy and rc==0
                row_data = next((r for r in d['results']
                                 if r['problem'] == prob
                                 and r['config'] == f'only_{flag}'), None)
                if row_data:
                    rep_ok = sum(1 for rep in row_data.get('reps', [])
                                 if rep.get('returncode', 0) == 0
                                 and rep.get('energy') is not None)
                    if rep_ok == 0:
                        continue
                matrix[row, j_f] = spd
                valid[row, j_f] = True
    return matrix, row_labels, valid

def plot(matrix, row_labels, valid):
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    # log2 mapping centered on 1.0; clip outliers
    plot_m = matrix.copy()
    plot_m = np.clip(plot_m, 0.4, 7.0)
    log_m = np.where(np.isnan(plot_m), np.nan, np.log2(plot_m))

    cmap = plt.cm.RdBu_r
    norm = TwoSlopeNorm(vmin=-1.4, vcenter=0.0, vmax=2.8)
    im = ax.imshow(log_m, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(len(FLAGS)))
    ax.set_xticklabels(FLAGS, rotation=30, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Cell annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                # Inapplicable / disabled
                ax.text(j, i, 'n/a', ha='center', va='center',
                        fontsize=8, color='gray')
            else:
                color = 'white' if abs(np.log2(min(max(v, 0.4), 7.0))) > 1.0 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=9, color=color, weight='bold' if v > 1.05 else 'normal')

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                        ticks=[-1, 0, 1, 2])
    cbar.ax.set_yticklabels([r'$0.5\times$', r'$1\times$', r'$2\times$', r'$4\times$'])
    cbar.set_label(r'Speedup vs.\ baseline (log$_2$ scale)', rotation=90,
                   labelpad=12)

    ax.set_xlabel('Optimization flag')
    ax.set_title(r'Per-flag ablation speedup --- Josephson, $L=32$, $n_{\max}=2$')

    # Light row separators between variants
    for i in range(len(RUNS) - 1):
        y = (i + 1) * len(PROBLEMS) - 0.5
        ax.axhline(y, color='black', linewidth=0.6, alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT)
    print(f"wrote {OUT}")

if __name__ == "__main__":
    m, labels, v = collect()
    plot(m, labels, v)
