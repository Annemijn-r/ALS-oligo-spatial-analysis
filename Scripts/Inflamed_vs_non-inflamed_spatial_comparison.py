"""
Script 06 — Inflamed vs Non-inflamed LCT Analysis
==================================================
Generates Figure 12-16 for the results section:
  Figure 12 — Cluster composition: dominance of cluster 6 in ALS
  Figure 13 — Tissue composition: cluster 6+7 proportion per sample
  Figure 14 — Category summary: % hexbins per category (inflamed vs non-inflamed)
  Figure 15 — Mann-Whitney: metabolic stress and microglia scores cluster 4 vs 6
  Figure 16 — Heatmap + forest plot: gene expression inflamed vs non-inflamed

Dependencies: matplotlib, pandas, numpy, scipy, statsmodels, seaborn
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════════
# Paths
# ════════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

REGION_PATH = SCRIPT_DIR

OUTPUT_PATH = os.path.join(REGION_PATH, 'Final_figures', 'Inflamed_non-inflamed_spatial_comparison')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── Style settings ────────────────────────────────────────────────────────────
plt.rcParams['font.family']       = 'Arial'
plt.rcParams['axes.facecolor']    = 'white'
plt.rcParams['figure.facecolor']  = 'white'
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

COLOR_MICROGLIA = '#6D597A'
COLOR_LIPID     = '#E5989B'
COLOR_AXONAL    = '#588157'
COLOR_MITO      = '#84A59D'
COLOR_ALS       = '#8C6275'
COLOR_CTRL      = '#688A7A'

CATEGORY_COLORS = {
    'Metabolic stress only':     COLOR_LIPID,
    'Microglia activation only': COLOR_MICROGLIA,
    'Both present':              '#B5838D',
    'Neither':                   '#D9D9D9',
}

GENE_CATEGORIES = {
    'Metabolic Stress':     ['PDK4', 'DDIT4'],
    'Myelin Breakdown':     ['CYP51A1', 'PLAU'],
    'Microglia Activation': ['CHIT1', 'LYZ'],
}

GENE_CAT_COLORS = {
    'Metabolic Stress':     COLOR_LIPID,
    'Myelin Breakdown':     COLOR_AXONAL,
    'Microglia Activation': COLOR_MICROGLIA,
}

# ── Sample lists ──────────────────────────────────────────────────────────────
ALS_SAMPLES = [
    ('ALS S09-055', 'Disease_hexpaint/LCT/ALS_S09-055_selection.csv.gz'),
    ('ALS S13-047', 'Disease_hexpaint/LCT/ALS_S13-047_selection.csv.gz'),
    ('ALS S13-054', 'Disease_hexpaint/LCT/ALS_S13-054_selection.csv.gz'),
    ('ALS S18-070', 'Disease_hexpaint/LCT/ALS_S18-070_selection.csv.gz'),
    ('ALS S19-061', 'Disease_hexpaint/LCT/ALS_S19-061_selection.csv.gz'),
]
CONTROL_SAMPLES = [
    ('CTR S12-186', 'Healthy_hexpaint/LCT/healthy_S12-186_selection.csv.gz'),
    ('CTR S13-133', 'Healthy_hexpaint/LCT/healthy_S13-133_selection.csv.gz'),
    ('CTR S15-025', 'Healthy_hexpaint/LCT/healthy_S15-025_selection.csv.gz'),
    ('CTR S15-051', 'Healthy_hexpaint/LCT/healthy_S15-051_selection.csv.gz'),
    ('CTR S17-062', 'Healthy_hexpaint/LCT/healthy_S17-062_selection.csv.gz'),
]

# ── Cluster definitions ───────────────────────────────────────────────────────
INFLAMED_CLUSTERS     = [6, 7]
NON_INFLAMED_CLUSTERS = [1, 3, 4, 5]
ALL_LCT_CLUSTERS      = INFLAMED_CLUSTERS + NON_INFLAMED_CLUSTERS

INFLAMED_CLUSTER     = 6   # primary inflamed cluster for detailed comparison
NON_INFLAMED_CLUSTER = 4   # primary non-inflamed cluster for detailed comparison

METABOLIC_GENES = ['PDK4', 'DDIT4']
MICROGLIA_GENES = ['CHIT1', 'LYZ']
ALL_TARGET      = ['PDK4', 'DDIT4', 'CYP51A1', 'PLAU', 'CHIT1', 'LYZ']
GENE_ORDER      = ['PDK4', 'DDIT4', 'CYP51A1', 'PLAU', 'CHIT1', 'LYZ']

# Log2FC values from DESeq2 inflamed vs non-inflamed analysis
LOG2FC_DATA = {
    'gene':    ['PDK4','PDK4','PDK4','PDK4','PDK4',
                'DDIT4','DDIT4','DDIT4','DDIT4','DDIT4',
                'CYP51A1','CYP51A1','CYP51A1','CYP51A1','CYP51A1',
                'PLAU','PLAU','PLAU','PLAU','PLAU',
                'CHIT1','CHIT1','CHIT1','CHIT1','CHIT1',
                'LYZ','LYZ','LYZ','LYZ','LYZ'],
    'sample':  ['ALS S09-055','ALS S13-047','ALS S13-054','ALS S18-070','ALS S19-061'] * 6,
    'log2FC':  [-0.44,-0.31,-0.49,-0.08,-0.15,
                 0.38, 0.04, 0.15,-0.15, 0.43,
                -0.58,-0.48,-0.11,-0.42,-0.22,
                 0.42, 0.52, 0.13, 0.36, 0.16,
                 0.81, 0.11, 0.85, 0.82, 0.13,
                 0.98, 1.23, 0.41, 0.42, 0.46],
    'significant': [True,False,True,False,True,
                    True,False,False,True,True,
                    True,True,True,True,True,
                    True,False,False,True,False,
                    True,False,True,True,False,
                    True,True,True,True,True],
    'category': (['Metabolic Stress']*5*2 +
                 ['Myelin Breakdown']*5*2 +
                 ['Microglia Activation']*5 +
                 ['Microglia Activation']*5),
}


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════════

def load_sample_clusters(filepath, sample_name):
    """Load hexbin data and return cluster composition per sample."""
    print(f"  Loading: {sample_name}...")
    df = pd.read_csv(filepath, compression='gzip',
                     usecols=['hexbin_id', 'L1_region_cluster', 'geneName', 'MIDCount'])
    df['L1_region_cluster'] = df['L1_region_cluster'].astype(int)
    return df


def load_and_score(filepath):
    """Load hexbin data, filter to LCT clusters, compute scores per hexbin."""
    genes = METABOLIC_GENES + MICROGLIA_GENES
    chunks = []
    for chunk in pd.read_csv(filepath, compression='gzip', chunksize=300_000,
                              usecols=lambda c: c in [
                                  'geneName', 'hexbin_id', 'hexbin_i', 'hexbin_j',
                                  'L1_region_cluster', 'MIDCount', 'median_normalized'
                              ]):
        sub = chunk[chunk['geneName'].isin(genes)]
        if len(sub): chunks.append(sub)

    if not chunks:
        return None
    df = pd.concat(chunks, ignore_index=True)

    # Use median_normalized if available, else MIDCount
    value_col = 'median_normalized' if 'median_normalized' in df.columns else 'MIDCount'

    coords = df.drop_duplicates('hexbin_id')[
        ['hexbin_id', 'hexbin_i', 'hexbin_j', 'L1_region_cluster']
    ].set_index('hexbin_id')

    pivot = df.pivot_table(index='hexbin_id', columns='geneName',
                           values=value_col, aggfunc='mean').fillna(0)
    pivot.columns.name = None
    pivot = pivot.join(coords)

    met_cols = [g for g in METABOLIC_GENES if g in pivot.columns]
    mic_cols = [g for g in MICROGLIA_GENES if g in pivot.columns]
    pivot['metabolic_score'] = pivot[met_cols].mean(axis=1) if met_cols else 0
    pivot['microglia_score'] = pivot[mic_cols].mean(axis=1) if mic_cols else 0

    return pivot


def classify_zone(zone_df):
    """Classify hexbins into four categories based on local median thresholds."""
    met_nz     = zone_df.loc[zone_df['metabolic_score'] > 0, 'metabolic_score']
    mic_nz     = zone_df.loc[zone_df['microglia_score'] > 0, 'microglia_score']
    met_thresh = met_nz.median() if len(met_nz) > 0 else 0.5
    mic_thresh = mic_nz.median() if len(mic_nz) > 0 else 0.5

    is_met = zone_df['metabolic_score'] >= met_thresh
    is_mic = zone_df['microglia_score'] >= mic_thresh

    zone_df = zone_df.copy()
    zone_df['category'] = np.select(
        [is_met & ~is_mic, is_mic & ~is_met, is_met & is_mic],
        ['Metabolic stress only', 'Microglia activation only', 'Both present'],
        default='Neither'
    )
    return zone_df, met_thresh, mic_thresh


def save(fig, name, dpi=300):
    path = os.path.join(OUTPUT_PATH, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 12 — Cluster composition: dominance of cluster 6 in ALS
# ════════════════════════════════════════════════════════════════════════════════

def fig_cluster_composition_dominance(all_samples):
    """Stacked bar chart showing cluster composition per sample."""
    print("\n── Figure 12: Cluster composition dominance ─────────")

    cluster_colors = {
        1: '#DCE4EC',  
        3: '#B2C6D4',  
        4: '#94A684',  
        5: '#E1D9EC',  
        6: '#83708F',  
        7: '#E5989B', 
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))

    sample_names = []
    cluster_data = {c: [] for c in ALL_LCT_CLUSTERS}

    for sample_name, filepath, condition in all_samples:
        if not os.path.exists(filepath): continue
        df = pd.read_csv(filepath, compression='gzip',
                         usecols=['hexbin_id', 'L1_region_cluster'])
        df['L1_region_cluster'] = df['L1_region_cluster'].astype(int)
        df = df[df['L1_region_cluster'].isin(ALL_LCT_CLUSTERS)]
        hexbins = df.drop_duplicates('hexbin_id')
        total = len(hexbins)
        if total == 0: continue

        sample_names.append(f"{condition}\n{sample_name.replace('ALS ','').replace('CTR ','')}")
        for c in ALL_LCT_CLUSTERS:
            pct = (hexbins['L1_region_cluster'] == c).sum() / total * 100
            cluster_data[c].append(pct)

    x = np.arange(len(sample_names))
    bottom = np.zeros(len(sample_names))

    for c in ALL_LCT_CLUSTERS:
        vals = cluster_data[c]
        ax.bar(x, vals, bottom=bottom, color=cluster_colors.get(c, '#C1C1C1'),
               label=f'Cluster {c}', edgecolor='white', linewidth=0.4)
        bottom += np.array(vals)

    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False)

    ax.set_xticks(x)
    ax.set_xticklabels(sample_names, fontsize=8.5, fontweight='bold', rotation=25, ha='right')
    
    for i, label in enumerate(sample_names):
        condition = label.split('\n')[0]
        if condition == 'ALS':
            ax.get_xticklabels()[i].set_color(COLOR_ALS)
        else:
            ax.get_xticklabels()[i].set_color(COLOR_CTRL)

    ax.set_ylabel('Percentage of hexbins (%)', fontsize=10, fontweight='bold')
    ax.set_title('Figure 12: LCT Cluster Composition per Sample',
                 fontsize=12, fontweight='bold', pad=35)
    ax.set_ylim(0, 102)

    handles, labels = ax.get_legend_handles_labels()
    sorted_pairs = sorted(zip(handles, labels), key=lambda t: int(t[1].split()[-1]))
    ordered_handles, ordered_labels = zip(*sorted_pairs)
   
    ax.legend(ordered_handles, ordered_labels,
              fontsize=8.5, 
              loc='lower center', 
              ncol=6, 
              frameon=False, 
              bbox_to_anchor=(0.5, 1.02))
    
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=10)

    plt.tight_layout()
    save(fig, 'Figure12_cluster_composition_dominance.png')


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 13 — Tissue composition: cluster 6+7 proportion per sample
# ════════════════════════════════════════════════════════════════════════════════

def fig_tissue_composition_67(all_samples):
    """Bar chart showing proportion of cluster 6+7 vs remaining tissue per sample."""
    print("\n── Figure 13: Tissue composition cluster 6+7 ───────────────────")

    fig, ax = plt.subplots(figsize=(12, 6))

    sample_names = []
    pct_67       = []
    pct_rest     = []
    conditions   = []

    for sample_name, filepath, condition in all_samples:
        if not os.path.exists(filepath): continue
        df = pd.read_csv(filepath, compression='gzip',
                         usecols=['hexbin_id', 'L1_region_cluster'])
        df['L1_region_cluster'] = df['L1_region_cluster'].astype(int)
        df = df[df['L1_region_cluster'].isin(ALL_LCT_CLUSTERS)]
        hexbins = df.drop_duplicates('hexbin_id')
        total   = len(hexbins)
        if total == 0: continue

        p67   = (hexbins['L1_region_cluster'].isin([6, 7])).sum() / total * 100
        prest = 100 - p67
        sample_names.append(sample_name)
        pct_67.append(p67)
        pct_rest.append(prest)
        conditions.append(condition)

    x = np.arange(len(sample_names))
    bar_colors = [COLOR_ALS if c == 'ALS' else COLOR_CTRL for c in conditions]

    ax.bar(x, pct_rest, color='#D9D9D9', label='Remaining tissue', edgecolor='white')
    ax.bar(x, pct_67,   bottom=pct_rest, color=bar_colors,
           label='Target Cluster (6+7)', edgecolor='white', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(sample_names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Percentage of selected tissue (%)', fontsize=10)
    ax.set_title('Tissue composition: Cluster 6 & 7 per selection',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 110)

    legend_elements = [
        mpatches.Patch(facecolor='#D9D9D9', label='Remaining Tissue'),
        mpatches.Patch(facecolor=COLOR_ALS,  label='Target Cluster (6+7) — ALS'),
        mpatches.Patch(facecolor=COLOR_CTRL, label='Target Cluster (6+7) — Control'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    save(fig, 'Figure13_tissue_composition_cluster67.png')


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 14 — Category summary: % hexbins per category
# ════════════════════════════════════════════════════════════════════════════════

def fig_category_summary(all_summaries):
    """% hexbins per category per zone per sample."""
    print("\n── Figure 14: Category summary bars ───────────────────────────")

    cats    = ['Metabolic stress only', 'Microglia activation only',
               'Both present', 'Neither']
    samples = list(all_summaries.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle('% hexbins per category per zone\n'
                 'Orange = metabolic stress without microglia activation',
                 fontsize=12, fontweight='bold')

    for ax_i, (zone, title) in enumerate([
        ('cl6', 'Cluster 6 — Inflamed LCT'),
        ('cl4', 'Cluster 4 — Non-inflamed LCT'),
    ]):
        ax = axes[ax_i]
        x  = np.arange(len(samples))
        w  = 0.18

        for i, cat in enumerate(cats):
            vals = [all_summaries[s][zone].get(cat, 0) for s in samples]
            ax.bar(x + i*w, vals, w, color=CATEGORY_COLORS[cat],
                   edgecolor='white', label=cat, zorder=3)

        ax.set_xticks(x + w*1.5)
        ax.set_xticklabels([f'ALS\n{s}' for s in samples], fontsize=9)
        ax.set_ylabel('% hexbins', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 75)
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        if ax_i == 1:
            ax.legend(fontsize=8, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    save(fig, 'Figure14_category_summary.png')


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 15 — Mann-Whitney: cluster 4 vs cluster 6
# ════════════════════════════════════════════════════════════════════════════════

def fig_statistical_comparison(all_results):
    """Mann-Whitney U: metabolic and microglia scores cluster 4 vs cluster 6."""
    print("\n── Figure 15: Mann-Whitney comparison ──────────────────────────")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('Statistical comparison: Cluster 4 vs Cluster 6\n'
                 'Mann-Whitney U test | Expected: metabolic stress high in cl4, microglia high in cl6',
                 fontsize=12, fontweight='bold')

    for ax_i, (score, ylabel) in enumerate([
        ('metabolic_score', 'Metabolic score\n(PDK4 + DDIT4 mean)'),
        ('microglia_score', 'Microglia score\n(CHIT1 + LYZ mean)'),
    ]):
        ax      = axes[ax_i]
        samples = list(all_results.keys())
        x = np.arange(len(samples))
        w = 0.32

        cl6_means, cl4_means = [], []
        cl6_sems,  cl4_sems  = [], []
        pvals = []

        for sname in samples:
            c6 = all_results[sname]['cl6'][score].values
            c4 = all_results[sname]['cl4'][score].values
            cl6_means.append(c6.mean())
            cl4_means.append(c4.mean())
            cl6_sems.append(c6.std() / np.sqrt(len(c6)))
            cl4_sems.append(c4.std() / np.sqrt(len(c4)))
            _, p = stats.mannwhitneyu(c6, c4, alternative='two-sided')
            pvals.append(p)

        ax.bar(x - w/2, cl6_means, w, color=COLOR_MICROGLIA, alpha=0.85,
               label='Cluster 6 (inflamed)', zorder=3)
        ax.bar(x + w/2, cl4_means, w, color=COLOR_AXONAL, alpha=0.85,
               label='Cluster 4 (non-inflamed)', zorder=3)
        ax.errorbar(x - w/2, cl6_means, yerr=cl6_sems, fmt='none',
                    color='#4A235A', lw=1.5, capsize=4, zorder=4)
        ax.errorbar(x + w/2, cl4_means, yerr=cl4_sems, fmt='none',
                    color='#2E5E4A', lw=1.5, capsize=4, zorder=4)

        for xi, p in enumerate(pvals):
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else
                  ('*' if p < 0.05 else 'ns'))
            y_top = max(cl6_means[xi] + cl6_sems[xi],
                        cl4_means[xi] + cl4_sems[xi]) * 1.1
            ax.plot([xi-w/2, xi-w/2, xi+w/2, xi+w/2],
                    [y_top, y_top*1.04, y_top*1.04, y_top],
                    color='black', lw=1)
            ax.text(xi, y_top*1.05, sig,
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([f'ALS\n{s}' for s in samples], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    save(fig, 'Figure15_mannwhitney_cl4_vs_cl6.png')


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 16 — Heatmap + forest plot
# ════════════════════════════════════════════════════════════════════════════════

def fig_heatmap_forest(all_results_df):
    """Publication-quality heatmap + forest plot of log2FC per gene per sample."""
    print("\n── Figure 16: Heatmap + forest plot ───────────────────────────")

    SAMPLE_COLORS  = ['#8C6275', '#A67F8E', '#C4A0AD', '#688A7A', '#8FAF9F']
    SAMPLE_MARKERS = ['o', 's', '^', 'D', 'v']

    pivot_fc  = all_results_df.pivot_table(
        index='gene', columns='sample', values='log2FC').reindex(GENE_ORDER)
    pivot_sig = all_results_df.pivot_table(
        index='gene', columns='sample', values='significant').reindex(GENE_ORDER)

    fig = plt.figure(figsize=(260/25.4, 155/25.4))
    gs_main = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1.0, 1.6],
        wspace=0.22, left=0.06, right=0.86, top=0.91, bottom=0.20,
    )
    ax_heat   = fig.add_subplot(gs_main[0])
    ax_forest = fig.add_subplot(gs_main[1])

    # ── Heatmap ───────────────────────────────────────────────────────────────
    cmap_custom = LinearSegmentedColormap.from_list(
        'als_custom', ['#588157', '#F5F0F0', '#C2527A'], N=256
    )
    vmax = max(abs(pivot_fc.values[~np.isnan(pivot_fc.values)])) * 1.05
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im   = ax_heat.imshow(pivot_fc.values, cmap=cmap_custom, norm=norm, aspect='auto')
    ax_heat.grid(False)

    for i, gene in enumerate(GENE_ORDER):
        for j, sample in enumerate(pivot_fc.columns):
            val = pivot_fc.loc[gene, sample]
            sig = pivot_sig.loc[gene, sample] if gene in pivot_sig.index else False
            if np.isnan(val): continue
            star     = '*' if sig else ''
            bright   = abs(val) / vmax
            txtcolor = 'white' if bright > 0.45 else 'black'
            ax_heat.text(j, i, f'{val:.2f}{star}',
                         ha='center', va='center', fontsize=7.5,
                         color=txtcolor,
                         fontweight='bold' if sig else 'normal')

    ax_heat.set_xticks(range(len(pivot_fc.columns)))
    ax_heat.set_xticklabels(
        [s.replace('ALS ', '') for s in pivot_fc.columns],
        rotation=35, ha='right', fontsize=8)
    ax_heat.set_yticks(range(len(GENE_ORDER)))
    ax_heat.set_yticklabels(GENE_ORDER, fontsize=9, fontstyle='italic')

    for i, gene in enumerate(GENE_ORDER):
        cat   = next((c for c, gs in GENE_CATEGORIES.items() if gene in gs), None)
        color = GENE_CAT_COLORS.get(cat, 'black')
        ax_heat.get_yticklabels()[i].set_color(color)
        ax_heat.get_yticklabels()[i].set_fontweight('bold')

    for brk in [2, 4]:
        ax_heat.axhline(brk - 0.5, color='white', lw=2.0)

    ax_heat.set_xlabel('Sample', fontsize=9, labelpad=6)
    ax_heat.set_ylabel('Gene', fontsize=9, labelpad=6)
    ax_heat.tick_params(length=0)

    cbar_ax = inset_axes(
        ax_heat, width='80%', height='5%', loc='lower center',
        bbox_to_anchor=(0, -0.22, 1, 1),
        bbox_transform=ax_heat.transAxes, borderpad=0,
    )
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Log₂FC (Inflamed vs Non-inflamed)', fontsize=7.5, labelpad=3)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)

    ax_heat.text(-0.22, 1.04, 'A', transform=ax_heat.transAxes,
                 fontsize=13, fontweight='bold', va='top')

    # ── Forest plot ───────────────────────────────────────────────────────────
    y_positions = {gene: i for i, gene in enumerate(reversed(GENE_ORDER))}
    samples     = list(all_results_df['sample'].unique())
    n_samples   = len(samples)
    dodge_width = 0.80

    for gene, y in y_positions.items():
        if y % 2 == 0:
            ax_forest.axhspan(y - 0.5, y + 0.5, color='#f5f5f5', zorder=0)

    ax_forest.axvline(0, color='#aaaaaa', lw=0.8, ls='--', zorder=1)

    for si, sample in enumerate(samples):
        s_data = all_results_df[all_results_df['sample'] == sample]
        offset = (si - (n_samples - 1) / 2) * (dodge_width / max(n_samples - 1, 1))
        for _, row in s_data.iterrows():
            if row['gene'] not in y_positions: continue
            y_coord = y_positions[row['gene']] + offset
            ax_forest.scatter(
                row['log2FC'], y_coord,
                marker=SAMPLE_MARKERS[si],
                color=SAMPLE_COLORS[si],
                s=28, alpha=0.92, zorder=4,
                edgecolors='black' if row['significant'] else 'none',
                linewidths=0.7,
            )

    for gene, y in y_positions.items():
        gene_data = all_results_df[all_results_df['gene'] == gene]
        mean_fc   = gene_data['log2FC'].mean()
        sd_fc     = gene_data['log2FC'].std()
        cat   = next((c for c, gs in GENE_CATEGORIES.items() if gene in gs), None)
        color = GENE_CAT_COLORS.get(cat, 'grey')

        ax_forest.plot([mean_fc - sd_fc, mean_fc + sd_fc], [y, y],
                       color=color, lw=2.5, alpha=0.45, zorder=6,
                       solid_capstyle='round')
        ax_forest.scatter(mean_fc, y, marker='D', color=color, s=110, zorder=7,
                          edgecolors='white', linewidths=1.0, alpha=0.9)

    ax_forest.set_yticks(list(y_positions.values()))
    ax_forest.set_yticklabels(list(y_positions.keys()), fontsize=9, fontstyle='italic')
    for label in ax_forest.get_yticklabels():
        gene  = label.get_text()
        cat   = next((c for c, gs in GENE_CATEGORIES.items() if gene in gs), None)
        color = GENE_CAT_COLORS.get(cat, 'black')
        label.set_color(color)
        label.set_fontweight('bold')

    ax_forest.tick_params(axis='y', pad=4)
    ax_forest.set_xlabel('Log₂FC (Inflamed vs Non-inflamed LCT)', fontsize=9, labelpad=6)
    ax_forest.set_ylabel('')
    ax_forest.yaxis.set_ticks_position('none')
    ax_forest.tick_params(axis='x', labelsize=8)
    ax_forest.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax_forest.xaxis.grid(True, which='major', color='#e0e0e0', lw=0.5, zorder=0)

    # Legend
    sample_handles = [
        plt.Line2D([0], [0], marker=SAMPLE_MARKERS[i], color='w',
                   markerfacecolor=SAMPLE_COLORS[i],
                   markeredgecolor='black', markeredgewidth=0.4,
                   markersize=5, label=s.replace('ALS ', ''))
        for i, s in enumerate(samples)
    ]
    mean_handle = plt.Line2D([0], [0], marker='D', color='grey',
                             markerfacecolor='grey', markeredgecolor='white',
                             markeredgewidth=0.5, markersize=6,
                             linestyle='-', linewidth=1.5, label='Mean ± SD')
    sig_handle  = plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor='#888888', markeredgecolor='black',
                             markeredgewidth=0.6, markersize=5, label='FDR < 0.05')
    sep_handle  = plt.Line2D([0], [0], color='none', label=' ')
    cat_handles = [
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=GENE_CAT_COLORS[cat], markersize=7,
                   markeredgewidth=0, label=cat)
        for cat in GENE_CATEGORIES
    ]

    leg = fig.legend(
        handles=sample_handles + [mean_handle, sig_handle, sep_handle] + cat_handles,
        labels=[s.replace('ALS ', '') for s in samples]
                + ['Mean ± SD', 'FDR < 0.05', ' ']
                + list(GENE_CATEGORIES.keys()),
        title='Sample', title_fontsize=8, fontsize=7.5,
        loc='upper right', bbox_to_anchor=(0.995, 0.93),
        frameon=True, framealpha=0.95, edgecolor='#cccccc',
        fancybox=False, borderpad=0.7, labelspacing=0.32, handlelength=1.4,
    )
    texts = leg.get_texts()
    n_sample_entries = len(samples) + 2
    texts[n_sample_entries].set_text('Category')
    texts[n_sample_entries].set_fontweight('bold')
    texts[n_sample_entries].set_fontsize(8)

    ax_forest.text(-0.10, 1.04, 'B', transform=ax_forest.transAxes,
                   fontsize=13, fontweight='bold', va='top')

    fig.suptitle('Gene expression: Inflamed vs Non-inflamed LCT regions in ALS',
                 fontsize=10.5, fontweight='bold', y=0.99)

    save(fig, 'Figure16_heatmap_forest_plot.png')


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  Script 06 — Inflamed vs Non-inflamed LCT Analysis")
    print("=" * 65)

    # Build full sample list for cluster composition figures
    all_samples = []
    for name, fname in ALS_SAMPLES:
        all_samples.append((name, os.path.join(REGION_PATH, fname), 'ALS'))
    for name, fname in CONTROL_SAMPLES:
        all_samples.append((name, os.path.join(REGION_PATH, fname), 'Control'))

    # ── Figures 12 and 13: cluster composition ──────────────────────────────
    print("\n── Loading cluster data ─────────────────────────────────────────")
    fig_cluster_composition_dominance(all_samples)
    fig_tissue_composition_67(all_samples)

    # ── Figures 14 and 15: category summary and Mann-Whitney ────────────────
    print("\n── Loading ALS samples for category analysis ────────────────────")
    all_summaries = {}
    all_results   = {}

    for sample_name, fname in ALS_SAMPLES:
        filepath = os.path.join(REGION_PATH, fname)
        if not os.path.exists(filepath):
            print(f"  WARNING: {filepath} not found"); continue
        print(f"  Processing: {sample_name}")

        pivot = load_and_score(filepath)
        if pivot is None: continue

        cl6 = pivot[pivot['L1_region_cluster'] == INFLAMED_CLUSTER].copy()
        cl4 = pivot[pivot['L1_region_cluster'] == NON_INFLAMED_CLUSTER].copy()

        if len(cl6) == 0 or len(cl4) == 0:
            print(f"  WARNING: cluster 6 or 4 not found — skipping"); continue

        cl6, _, _ = classify_zone(cl6)
        cl4, _, _ = classify_zone(cl4)

        sname = sample_name.replace('ALS ', '')
        all_summaries[sname] = {
            'cl6': {cat: cl6['category'].eq(cat).sum() / len(cl6) * 100
                    for cat in CATEGORY_COLORS},
            'cl4': {cat: cl4['category'].eq(cat).sum() / len(cl4) * 100
                    for cat in CATEGORY_COLORS},
        }
        all_results[sname] = {'cl6': cl6, 'cl4': cl4}
        del pivot; gc.collect()

    if all_summaries:
        fig_category_summary(all_summaries)
    if all_results:
        fig_statistical_comparison(all_results)

    # ── Figure 16: heatmap + forest plot ─────────────────────────────────────
    all_results_df = pd.DataFrame(LOG2FC_DATA)
    fig_heatmap_forest(all_results_df)

    print("\n" + "=" * 65)
    print(f"  Done! All figures saved to:")
    print(f"  {OUTPUT_PATH}")
    print("=" * 65)
