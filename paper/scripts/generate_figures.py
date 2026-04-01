#!/usr/bin/env python3
"""
Generate publication-quality figures for the CPC paper.
Reads benchmark data from benchmarks/paper_results/ and outputs PDF figures to paper/figures/.
"""
import json
import csv
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict

# Paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "benchmarks", "paper_results")
FIG_DIR = os.path.join(REPO_ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.figsize': (7, 5),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Markers and colors for implementations
IMPL_STYLE = {
    'dmrg-gpu':      {'color': '#1f77b4', 'marker': 'o', 'label': 'dmrg-gpu (1-site)'},
    'dmrg2-gpu':     {'color': '#ff7f0e', 'marker': 's', 'label': 'dmrg2-gpu (2-site)'},
    'dmrg-gpu-opt':  {'color': '#1f77b4', 'marker': 'o', 'label': 'dmrg-gpu-opt', 'ls': '--'},
    'dmrg2-gpu-opt': {'color': '#ff7f0e', 'marker': 's', 'label': 'dmrg2-gpu-opt', 'ls': '--'},
    'pdmrg-gpu':     {'color': '#2ca02c', 'marker': '^', 'label': 'pdmrg-gpu'},
    'pdmrg-gpu-opt': {'color': '#2ca02c', 'marker': '^', 'label': 'pdmrg-gpu-opt', 'ls': '--'},
    'quimb-dmrg1':   {'color': '#d62728', 'marker': 'D', 'label': 'quimb-dmrg1 (CPU)'},
    'quimb-dmrg2':   {'color': '#9467bd', 'marker': 'v', 'label': 'quimb-dmrg2 (CPU)'},
}


def load_main_results():
    """Load the 604-entry main benchmark dataset."""
    with open(os.path.join(DATA_DIR, "results.json")) as f:
        return json.load(f)


def load_gpu_opt_results():
    """Load the 108-entry GPU opt scaling study."""
    with open(os.path.join(DATA_DIR, "gpu_opt_bench.json")) as f:
        return json.load(f)


def load_opt_csv():
    """Load opt vs baseline CSV."""
    path = os.path.join(DATA_DIR, "bench_opt_results.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def best_cpu_time(results, model, L, chi):
    """Get best CPU wall time across all quimb runs and thread counts."""
    times = []
    for r in results:
        if (r.get('model') == model and r.get('L') == L and r.get('chi') == chi
            and 'quimb' in r.get('impl', '') and r.get('wall_time') is not None
            and r.get('success', False)):
            times.append(r['wall_time'])
    return min(times) if times else None


def best_gpu_time(results, model, L, chi):
    """Get best GPU wall time across baseline GPU impls (no -opt)."""
    times = []
    for r in results:
        impl = r.get('impl', '')
        if (r.get('model') == model and r.get('L') == L and r.get('chi') == chi
            and 'gpu' in impl and 'opt' not in impl
            and r.get('wall_time') is not None and r.get('success', False)
            and not r.get('segments')):
            times.append(r['wall_time'])
    return min(times) if times else None


# ============================================================
# Figure 1: Wall time vs chi (3-panel, by model)
# ============================================================
def fig1_walltime_vs_chi():
    results = load_main_results()
    models = ['heisenberg', 'josephson', 'tfim']
    titles = ['Heisenberg ($d=2$, real)', 'Josephson ($d=5$, complex)', 'TFIM ($d=2$, critical)']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, model, title in zip(axes, models, titles):
        # Group by impl, collect (chi, best_time) for L=32
        L_target = 32
        impls_to_plot = ['dmrg-gpu', 'dmrg2-gpu', 'quimb-dmrg1', 'quimb-dmrg2']

        for impl_name in impls_to_plot:
            chi_times = defaultdict(list)
            for r in results:
                if (r.get('model') == model and r.get('impl') == impl_name
                    and r.get('L') == L_target and r.get('wall_time') is not None
                    and r.get('success', False) and not r.get('segments')):
                    chi_times[r['chi']].append(r['wall_time'])

            if not chi_times:
                continue
            chis = sorted(chi_times.keys())
            times = [min(chi_times[c]) for c in chis]

            style = IMPL_STYLE.get(impl_name, {})
            ax.plot(chis, times, marker=style.get('marker', 'o'),
                    color=style.get('color', 'gray'),
                    linestyle=style.get('ls', '-'),
                    label=style.get('label', impl_name),
                    markersize=6, linewidth=1.5)

        ax.set_xlabel('Bond dimension $\\chi$')
        ax.set_title(title)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Wall time (s)')
    axes[0].legend(loc='upper left', framealpha=0.9)
    fig.suptitle(f'Wall time vs. bond dimension ($L = 32$)', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_walltime_vs_chi.pdf'))
    plt.close()
    print("  fig1_walltime_vs_chi.pdf")


# ============================================================
# Figure 2: GPU speedup heatmap (L x chi)
# ============================================================
def fig2_gpu_speedup_heatmap():
    results = load_main_results()
    model = 'heisenberg'

    # Collect (L, chi) -> speedup
    configs = set()
    for r in results:
        if r.get('model') == model and r.get('success', False) and r.get('wall_time'):
            configs.add((r['L'], r['chi']))

    speedups = {}
    for L, chi in configs:
        gpu_t = best_gpu_time(results, model, L, chi)
        cpu_t = best_cpu_time(results, model, L, chi)
        if gpu_t and cpu_t:
            speedups[(L, chi)] = cpu_t / gpu_t

    if not speedups:
        print("  fig2: no data, skipping")
        return

    Ls = sorted(set(k[0] for k in speedups))
    chis = sorted(set(k[1] for k in speedups))

    data = np.full((len(Ls), len(chis)), np.nan)
    for i, L in enumerate(Ls):
        for j, chi in enumerate(chis):
            if (L, chi) in speedups:
                data[i, j] = speedups[(L, chi)]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0.3, vmax=3.0,
                   origin='lower')
    ax.set_xticks(range(len(chis)))
    ax.set_xticklabels(chis)
    ax.set_yticks(range(len(Ls)))
    ax.set_yticklabels(Ls)
    ax.set_xlabel('Bond dimension $\\chi$')
    ax.set_ylabel('System size $L$')
    ax.set_title('GPU speedup over best CPU (Heisenberg)')

    # Annotate cells
    for i in range(len(Ls)):
        for j in range(len(chis)):
            if not np.isnan(data[i, j]):
                val = data[i, j]
                color = 'white' if val > 2.0 or val < 0.5 else 'black'
                ax.text(j, i, f'{val:.1f}x', ha='center', va='center',
                        color=color, fontsize=9, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, label='Speedup (>1 = GPU wins)')
    cbar.ax.axhline(y=1.0, color='black', linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_gpu_speedup_heatmap.pdf'))
    plt.close()
    print("  fig2_gpu_speedup_heatmap.pdf")


# ============================================================
# Figure 3: Opt vs baseline scatter
# ============================================================
def fig3_opt_vs_baseline():
    # Opt vs baseline data lives in gpu_opt_bench.json (has -gpu-opt impls)
    gpu_opt = load_gpu_opt_results()

    pairs = []  # (baseline_time, opt_time, label)

    # Group by (model, L, chi, type=1site/2site)
    baseline_map = {}
    opt_map = {}
    for r in gpu_opt:
        if not r.get('wall_time') or r.get('label'):
            continue  # skip batched/baseline labeled entries
        if r.get('segments'):
            continue
        impl = r.get('impl', '')
        key = (r['model'], r['L'], r['chi'])
        if impl == 'dmrg-gpu':
            baseline_map.setdefault(('1site', *key), []).append(r['wall_time'])
        elif impl == 'dmrg2-gpu':
            baseline_map.setdefault(('2site', *key), []).append(r['wall_time'])
        elif impl == 'dmrg-gpu-opt':
            opt_map.setdefault(('1site', *key), []).append(r['wall_time'])
        elif impl == 'dmrg2-gpu-opt':
            opt_map.setdefault(('2site', *key), []).append(r['wall_time'])

    for key in baseline_map:
        if key in opt_map:
            bt = min(baseline_map[key])
            ot = min(opt_map[key])
            pairs.append((bt, ot, key[0]))

    if not pairs:
        print("  fig3: no paired data, skipping")
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot pairs
    for bt, ot, label in pairs:
        color = '#1f77b4' if label == '1site' else '#ff7f0e'
        marker = 'o' if label == '1site' else 's'
        ax.scatter(bt, ot, c=color, marker=marker, s=40, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Diagonal (parity)
    lims = [0.3, max(max(p[0], p[1]) for p in pairs) * 1.5]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Parity')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Baseline wall time (s)')
    ax.set_ylabel('Optimized variant wall time (s)')
    ax.set_title('Newton--Schulz/Davidson vs.\\ Baseline')

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
               markersize=8, label='1-site'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff7f0e',
               markersize=8, label='2-site'),
        Line2D([0], [0], color='black', linestyle='--', label='Parity'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Annotate: all points above diagonal
    ax.fill_between(lims, lims, [lims[1]]*2, alpha=0.05, color='red')
    ax.text(0.95, 0.05, 'Variant slower\n(all points)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, color='red', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_opt_vs_baseline.pdf'))
    plt.close()
    print("  fig3_opt_vs_baseline.pdf")


# ============================================================
# Figure 4: SVD time fraction (stacked bar)
# ============================================================
def fig4_svd_fraction():
    """SVD dominance at chi=256 from profiling data."""
    # Data from pdmrg-gpu/OPTIMIZATION_REPORT.md profiling
    configs = ['$\\chi=128$\n$L=64$', '$\\chi=256$\n$L=64$']
    svd_frac = [0.95, 0.975]  # approximate from report
    lanczos_frac = [0.04, 0.02]
    other_frac = [0.01, 0.005]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(configs))
    width = 0.5

    bars1 = ax.bar(x, svd_frac, width, label='SVD (CPU LAPACK)', color='#d62728')
    bars2 = ax.bar(x, lanczos_frac, width, bottom=svd_frac, label='Lanczos + $H_\\mathrm{eff}$', color='#1f77b4')
    combined = [s + l for s, l in zip(svd_frac, lanczos_frac)]
    bars3 = ax.bar(x, other_frac, width, bottom=combined, label='Env updates + other', color='#2ca02c')

    ax.set_ylabel('Fraction of sweep wall time')
    ax.set_title('Per-sweep time breakdown (dmrg2-gpu, Heisenberg)')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_svd_fraction.pdf'))
    plt.close()
    print("  fig4_svd_fraction.pdf")


# ============================================================
# Figure 5: Batched sweep scaling
# ============================================================
def fig5_batched_sweep():
    gpu_opt = load_gpu_opt_results()

    # Pair baseline/batched
    baseline = {}
    batched = {}
    for r in gpu_opt:
        if r.get('label') == 'baseline' and r.get('wall_time'):
            key = (r['model'], r['L'], r['chi'], r.get('segments'))
            baseline[key] = r['wall_time']
        elif r.get('label') == 'batched' and r.get('wall_time'):
            key = (r['model'], r['L'], r['chi'], r.get('segments'))
            batched[key] = r['wall_time']

    pairs = []
    for key in baseline:
        if key in batched:
            pairs.append({
                'model': key[0], 'L': key[1], 'chi': key[2], 'seg': key[3],
                'baseline': baseline[key], 'batched': batched[key],
                'speedup': baseline[key] / batched[key]
            })

    if not pairs:
        print("  fig5: no batched pairs, skipping")
        return

    pairs.sort(key=lambda p: (p['chi'], p['seg'], p['L']))

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    base_times = []
    batch_times = []
    for p in pairs:
        labels.append(f"L={p['L']}\n$\\chi$={p['chi']}\nP={p['seg']}")
        base_times.append(p['baseline'])
        batch_times.append(p['batched'])

    x = np.arange(len(pairs))
    width = 0.35
    ax.bar(x - width/2, base_times, width, label='Thread-per-segment', color='#1f77b4')
    ax.bar(x + width/2, batch_times, width, label='Batched GEMM', color='#ff7f0e')

    ax.set_ylabel('Wall time (s)')
    ax.set_title('Cross-segment batched GEMM: baseline vs.\\ batched sweep')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Mark the one winner
    for i, p in enumerate(pairs):
        if p['speedup'] > 1.05:
            ax.annotate(f"{p['speedup']:.2f}x", (i, min(p['baseline'], p['batched'])),
                       textcoords="offset points", xytext=(0, -15),
                       ha='center', fontsize=8, color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig5_batched_sweep.pdf'))
    plt.close()
    print("  fig5_batched_sweep.pdf")


# ============================================================
# Main
# ============================================================
def main():
    print("Generating figures for CPC paper...")
    print(f"  Data: {DATA_DIR}")
    print(f"  Output: {FIG_DIR}")
    print()

    fig1_walltime_vs_chi()
    fig2_gpu_speedup_heatmap()
    fig3_opt_vs_baseline()
    fig4_svd_fraction()
    fig5_batched_sweep()

    print("\nDone!")


if __name__ == "__main__":
    main()
