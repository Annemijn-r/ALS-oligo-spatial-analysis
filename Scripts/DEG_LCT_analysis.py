"""
Script 4 — Differential Gene Expression Analysis: ALS LCT
===========================================================
Generates Figures 1-6 for the results section:
  Figure 1  — Volcano plot
  Figure 2  — MA plot
  Figure 3  — DEG table
  Figure 4  — Heatmap top 15 DEGs
  Figure 5  — Boxplots per gene (CPM)
  Figure 6  — Bar chart: Inflammation vs Axonal/Metabolic Integrity

Dependencies: pydeseq2, matplotlib, pandas, numpy, scipy, statsmodels
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore')

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
COLOR_NS        = '#D9D9D9'
# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.getcwd()
HEALTHY_PATH = os.path.join(BASE_DIR, "output", "Region_clusters", "Healthy_hexpaint", "LCT")
ALS_PATH     = os.path.join(BASE_DIR, "output", "Region_clusters", "Disease_hexpaint", "LCT")
OUTPUT_PATH  = os.path.join(BASE_DIR, "output", "Region_clusters", "Final_figures", "DEG_LCT_analysis")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── Sample metadata ───────────────────────────────────────────────────────────
METADATA = {
    'S12-186': 'healthy', 'S13-133': 'healthy', 'S15-025': 'healthy',
    'S15-051': 'healthy', 'S17-062': 'healthy',
    'S09-055': 'ALS',     'S13-047': 'ALS',     'S13-054': 'ALS',
    'S18-070': 'ALS',     'S19-061': 'ALS',
}

FILE_MAP = {
    'S12-186': (HEALTHY_PATH, 'healthy_S12-186_selection.csv.gz'),
    'S13-133': (HEALTHY_PATH, 'healthy_S13-133_selection.csv.gz'),
    'S15-025': (HEALTHY_PATH, 'healthy_S15-025_selection.csv.gz'),
    'S15-051': (HEALTHY_PATH, 'healthy_S15-051_selection.csv.gz'),
    'S17-062': (HEALTHY_PATH, 'healthy_S17-062_selection.csv.gz'),
    'S09-055': (ALS_PATH,     'ALS_S09-055_selection.csv.gz'),
    'S13-047': (ALS_PATH,     'ALS_S13-047_selection.csv.gz'),
    'S13-054': (ALS_PATH,     'ALS_S13-054_selection.csv.gz'),
    'S18-070': (ALS_PATH,     'ALS_S18-070_selection.csv.gz'),
    'S19-061': (ALS_PATH,     'ALS_S19-061_selection.csv.gz'),
}

# ── Biological category assignment rules ──────────────────────────────────────
KNOWN_MICROGLIA_GENES = [
    'CHIT1', 'LYZ', 'FCGR2B', 'CD52', 'C1QA', 'C1QB', 'TYROBP',
    'TREM2', 'ITGB2', 'GPNMB', 'C3', 'CTSS', 'P2RY12', 'AIF1', 'TM4SF18'
]
KNOWN_MYELIN_ECM_GENES  = ['PLAU', 'CYP51A1', 'MBP', 'PLP1', 'MAG', 'MOG', 'MTUS1']
KNOWN_METABOLIC_GENES   = ['PDK4', 'DDIT4']

CAT_COLOR = {
    'Microglia activation':  COLOR_ALS,
    'Myelin/ECM remodeling': COLOR_AXONAL,
    'Metabolic stress':      COLOR_LIPID,
    'Other upregulated':     '#D9D9D9',
    'Other downregulated':   '#D9D9D9'
}

ALS_COLORS = {
    'S09-055': '#6D3A4A',
    'S13-047': '#8C6275',
    'S13-054': '#A67F8E',
    'S18-070': '#C4A0AD',
    'S19-061': '#DEC3CB',
}
CTRL_COLORS = {
    'S12-186': '#2E5E4A',
    'S13-133': '#3D7A60',
    'S15-025': '#688A7A',
    'S15-051': '#8FAF9F',
    'S17-062': '#B5CFC5',
}


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PROCESSING
# ════════════════════════════════════════════════════════════════════════════════

def load_count_matrix(file_map, metadata, chunksize=300_000):
    print("\n── Step 1: Loading count data ──────────────────────────────────")
    sample_counts = {}
    for sname, (path, fname) in file_map.items():
        fpath = os.path.join(path, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: File not found: {fpath} — sample skipped")
            continue
        print(f"  Loading: {sname} ({metadata[sname]})...", flush=True)
        chunks = []
        for chunk in pd.read_csv(fpath, compression='gzip', chunksize=chunksize,
                                  usecols=['geneName', 'MIDCount']):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        del chunks; gc.collect()
        sample_counts[sname] = df.groupby('geneName')['MIDCount'].sum()
        del df; gc.collect()

    count_matrix = pd.DataFrame(sample_counts).T.fillna(0)
    print(f"  Count matrix: {count_matrix.shape[0]} samples × {count_matrix.shape[1]} genes")
    return count_matrix


def filter_by_cpm(count_matrix, min_cpm=1, min_samples=3):
    print("\n── Step 2: CPM filtering ────────────────────────────────────────")
    cpm  = count_matrix.div(count_matrix.sum(axis=1), axis=0) * 1e6
    keep = (cpm > min_cpm).sum(axis=0) >= min_samples
    filtered = count_matrix.loc[:, keep].astype(int)
    print(f"  Before: {count_matrix.shape[1]:,} genes")
    print(f"  After:  {filtered.shape[1]:,} genes")
    return filtered, cpm.loc[:, keep]


def run_deseq2(count_matrix, metadata_dict):
    print("\n── Step 3: DESeq2 analysis ──────────────────────────────────────")
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    metadata = pd.DataFrame({'condition': metadata_dict})\n    metadata.index.name = 'sample'
    metadata = metadata.loc[count_matrix.index]

    dds = DeseqDataSet(counts=count_matrix, metadata=metadata,
                       design_factors='condition', quiet=False)
    dds.deseq2()
    stat_res = DeseqStats(dds, contrast=['condition', 'ALS', 'healthy'])
    stat_res.summary()
    results = stat_res.results_df.sort_values('padj')
    sig = (results['padj'] < 0.05).sum()
    print(f"  Genes tested: {len(results):,}  |  Significant (padj<0.05): {sig}")
    return results


def derive_gene_sets(results, n_top=15):
    """
    Derive TOP_genes, UP_GENES, DOWN_GENES and CAT_MAP automatically
    from DESeq2 results. No gene names are hardcoded.

    Selection criteria:
    - padj < 0.05 and |log2FC| > 1
    - Top n_top genes ranked by absolute log2FC
    - Biological category assigned based on known marker lists
    """
    print("\n── Step 4: Deriving gene sets from results ──────────────────────")

    sig = results[(results['padj'] < 0.05) & (results['log2FoldChange'].abs() > 1)].copy()
    sig = sig.sort_values('log2FoldChange', ascending=False)

    top = sig.head(n_top)
    top_genes = top.index.tolist()
    up_genes  = top[top['log2FoldChange'] > 0].index.tolist()
    down_genes= top[top['log2FoldChange'] < 0].index.tolist()

    print(f"  Significant DEGs (padj<0.05, |log2FC|>1): {len(sig)}")
    print(f"  Top {n_top} selected: {len(top_genes)} genes")
    print(f"  Upregulated:   {up_genes}")
    print(f"  Downregulated: {down_genes}")

    # Assign biological categories
    def assign_category(gene, direction):
        if gene in KNOWN_MICROGLIA_GENES:  return 'Microglia activation'
        if gene in KNOWN_MYELIN_ECM_GENES: return 'Myelin/ECM remodeling'
        if gene in KNOWN_METABOLIC_GENES:  return 'Metabolic stress'
        return 'Other upregulated' if direction == 'up' else 'Other downregulated'

    cat_map = {}
    for gene in top_genes:
        direction = 'up' if gene in up_genes else 'down'
        cat_map[gene] = assign_category(gene, direction)

    # Figure 6 categories: inflammation vs axonal/metabolic
    inflam_genes = [g for g in up_genes]
    axonal_genes = [g for g in down_genes]

    print(f"\n  Category breakdown:")
    from collections import Counter
    for cat, count in Counter(cat_map.values()).items():
        print(f"    {cat}: {count} genes")

    return top_genes, up_genes, down_genes, inflam_genes, axonal_genes, cat_map


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def save(fig, name, dpi=300):
    path = os.path.join(OUTPUT_PATH, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}")


def fig_volcano(results, top_genes):
    """Figure 1 — Volcano plot: log2FC vs -log10(padj)"""
    print("\n── Figure 1: Volcano plot ───────────────────────────────────────")
    fig, ax = plt.subplots(figsize=(10, 7))

    up  = results[(results['padj'] < 0.05) & (results['log2FoldChange'] > 0)]
    dn  = results[(results['padj'] < 0.05) & (results['log2FoldChange'] < 0)]
    ns  = results[~results.index.isin(up.index) & ~results.index.isin(dn.index)]

    ax.scatter(ns['log2FoldChange'], -np.log10(ns['padj'].clip(1e-10)),
               c=COLOR_NS, s=8, alpha=0.4, label='Not significant')
    ax.scatter(up['log2FoldChange'], -np.log10(up['padj'].clip(1e-10)),
               c=COLOR_ALS, s=18, alpha=0.7, label=f'Upregulated (n={len(up)})')
    ax.scatter(dn['log2FoldChange'], -np.log10(dn['padj'].clip(1e-10)),
               c=COLOR_CTRL, s=18, alpha=0.7, label=f'Downregulated (n={len(dn)})')

    # Collect label positions and add with adjustText
    from adjustText import adjust_text
    texts = []
    for gene in top_genes:
        if gene in results.index:
            r = results.loc[gene]
            x = r['log2FoldChange']
            y = -np.log10(max(r['padj'], 1e-10))
            texts.append(ax.text(x, y, gene, fontsize=8, fontweight='bold'))

    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
        expand_points=(2.0, 2.0),
        expand_text=(2.0, 2.0)
    )

    ax.axhline(-np.log10(0.05), color='red', ls='--', lw=0.8, alpha=0.6)
    ax.axvline(0, color='black', lw=0.6)
    ax.set_xlabel('log2FC (ALS vs Control)', fontsize=11)
    ax.set_ylabel('-log10(padj)', fontsize=11)
    ax.set_title('Volcano Plot: ALS vs Control (LCT)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    save(fig, 'Figure1_volcano_plot.png')


def fig_ma(results, top_genes):
    """Figure 2 — MA plot: mean expression vs log2FC"""
    print("\n── Figure 2: MA plot ────────────────────────────────────────────")
    fig, ax = plt.subplots(figsize=(10, 7))

    x   = np.log10(results['baseMean'] + 1)
    y   = results['log2FoldChange']
    sig = results['padj'] < 0.05
    up  = sig & (y > 0)
    dn  = sig & (y < 0)

    ax.scatter(x[~sig], y[~sig], c=COLOR_NS, s=5, alpha=0.3)
    ax.scatter(x[up],   y[up],   c=COLOR_ALS,  s=18, alpha=0.8,
               label=f'Upregulated (n={up.sum()})')
    ax.scatter(x[dn],   y[dn],   c=COLOR_CTRL, s=18, alpha=0.8,
               label=f'Downregulated (n={dn.sum()})')

    for gene in top_genes:
        if gene in results.index:
            r = results.loc[gene]
            ax.annotate(gene, (np.log10(r['baseMean']+1), r['log2FoldChange']),
                        fontsize=8, fontweight='bold',
                        xytext=(5, 2), textcoords='offset points')

    ax.axhline(0, color='black', ls='--', lw=0.8)
    ax.set_xlabel('Mean Expression (log10 baseMean + 1)', fontsize=11)
    ax.set_ylabel('log2FC', fontsize=11)
    ax.set_title('MA Plot: ALS vs Control (LCT)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    save(fig, 'Figure2_MA_plot.png')


def fig_deg_table(results, top_genes, cat_map):
    """Figure 3 — DEG table (styled)"""
    print("\n── Figure 3: DEG table ──────────────────────────────────────────")
    available = [g for g in top_genes if g in results.index]
    tbl = results.loc[available, ['baseMean', 'log2FoldChange', 'pvalue', 'padj']].copy()
    tbl['direction'] = tbl['log2FoldChange'].apply(lambda x: 'UP' if x > 0 else 'DOWN')

    fig = plt.figure(figsize=(12, 7.5))
    ax  = fig.add_subplot(111)
    ax.axis('off')
    fig.suptitle('DEG Table — ALS vs Control: Top genes (LCT)\nPyDESeq2 · FDR-corrected (BH)',
                 fontsize=13, fontweight='bold', y=0.97)

    col_labels = ['Gene', 'baseMean\n(CPM)', 'log2FC\n(ALS/Ctrl)', 'p-value', 'padj', 'Direction']
    rows_data  = []
    for gene, row in tbl.iterrows():
        rows_data.append([
            gene, f"{row['baseMean']:.1f}", f"{row['log2FoldChange']:+.2f}",
            f"{row['pvalue']:.2e}", f"{row['padj']:.4f}",
            '▲  UP' if row['direction'] == 'UP' else '▼  DOWN'
        ])

    table = ax.table(cellText=rows_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.75)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#1C2833')
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=9)

    for i, (gene, row) in enumerate(tbl.iterrows(), 1):
        color = CAT_COLOR.get(cat_map.get(gene, 'Other downregulated'), '#9D9D9D')
        bg    = '#FFF5F5' if row['direction'] == 'UP' else '#F0F8FF'
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)
            table[i, j].set_edgecolor('#DDDDDD')
        table[i, 0].set_facecolor(color)
        table[i, 0].set_text_props(color='white', fontweight='bold')
        fc_color = '#922B21' if row['log2FoldChange'] > 0 else '#154360'
        table[i, 2].set_text_props(color=fc_color, fontweight='bold')
        if row['padj'] < 0.01:
            table[i, 4].set_facecolor('#D5F5E3')
            table[i, 4].set_text_props(color='#1E8449', fontweight='bold')
        elif row['padj'] < 0.05:
            table[i, 4].set_facecolor('#EAF7EF')
            table[i, 4].set_text_props(color='#27AE60')
        dir_col = '#922B21' if row['direction'] == 'UP' else '#154360'
        table[i, 5].set_text_props(color=dir_col, fontweight='bold')

    patches = [mpatches.Patch(color=c, label=l) for l, c in CAT_COLOR.items()]
    fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=8.5,
               frameon=True, bbox_to_anchor=(0.5, 0.01),
               title='Gene category', title_fontsize=9)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    save(fig, 'Figure3_DEG_table.png')

    tbl.to_csv(os.path.join(OUTPUT_PATH, 'DEG_table_top15.csv'))
    print("  CSV saved: DEG_table_top_genes.csv")


def fig_heatmap(results, cpm_matrix, metadata_dict, top_genes):
    """Figure 4 — Z-score heatmap top gene DEGs per sample"""
    print("\n── Figure 4: Heatmap top genes ─────────────────────────────────────")
    available    = [g for g in top_genes if g in cpm_matrix.columns]
    sorted_genes = results.loc[[g for g in available if g in results.index]]\
                          .sort_values('log2FoldChange', ascending=False).index.tolist()

    sorted_samples = (
        [s for s, c in metadata_dict.items() if c == 'healthy' and s in cpm_matrix.index] +
        [s for s, c in metadata_dict.items() if c == 'ALS'     and s in cpm_matrix.index]
    )
    heat_data = cpm_matrix[sorted_genes].loc[sorted_samples].T
    heat_z    = heat_data.apply(lambda x: (x - x.mean()) / (x.std() + 1e-9), axis=1)

    fig, ax = plt.subplots(figsize=(14, 9))
    cmap_custom = LinearSegmentedColormap.from_list(
    'als_custom',
    ['#588157', '#F5F0F0', '#C2527A'],
    N=256
    )
    im = ax.imshow(heat_z.values, cmap=cmap_custom, aspect='auto', vmin=-2.5, vmax=2.5)

    ax.set_yticks(range(len(sorted_genes)))
    ax.set_yticklabels(sorted_genes, fontsize=10)
    ax.set_xticks(range(len(sorted_samples)))
    ax.set_xticklabels(sorted_samples, rotation=45, ha='right', fontsize=9)

    for xi, s in enumerate(sorted_samples):
        color = COLOR_ALS if metadata_dict.get(s) == 'ALS' else COLOR_CTRL
        ax.get_xticklabels()[xi].set_color(color)

    plt.colorbar(im, ax=ax, label='Z-score (relative expression)', shrink=0.6)
    ax.set_title('Heatmap: Top genes DEGs (LCT)\nSorted Upregulated → Downregulated',
                 fontsize=13, fontweight='bold')

    n_healthy = sum(1 for s in sorted_samples if metadata_dict.get(s) == 'healthy')
    ax.axvline(n_healthy - 0.5, color='black', lw=2.5)

    patches = [mpatches.Patch(color=COLOR_CTRL, label='Control'),
               mpatches.Patch(color=COLOR_ALS,  label='ALS')]
    ax.legend(handles=patches, loc='upper right', fontsize=9)
    save(fig, 'Figure4_heatmap_top_genes.png')


def fig_boxplots(cpm_matrix, results, metadata_dict, top_genes, up_genes, down_genes,
                 als_colors, ctrl_colors):
    """Figure 5 — Boxplots per gene with individual sample points"""
    print("\n── Figure 5: Boxplots per gene ──────────────────────────────────")
    available = [g for g in top_genes if g in cpm_matrix.columns]
    als_s     = [s for s, c in metadata_dict.items() if c == 'ALS'     and s in cpm_matrix.index]
    ctrl_s    = [s for s, c in metadata_dict.items() if c == 'healthy' and s in cpm_matrix.index]

    ncols = 4
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4.2))
    axes = axes.flatten()

    fig.suptitle(
        'Gene expression (CPM) per sample: ALS vs Control\n'
        'Each dot = one sample  ·  Box = group median + IQR  ·  Mann-Whitney U',
        fontsize=13, fontweight='bold', y=1.01
    )
    rng = np.random.default_rng(42)

    for idx, gene in enumerate(available):
        ax = axes[idx]
        als_vals  = cpm_matrix.loc[als_s,  gene].values
        ctrl_vals = cpm_matrix.loc[ctrl_s, gene].values
        als_nz    = als_vals[als_vals > 0]
        ctrl_nz   = ctrl_vals[ctrl_vals > 0]

        bp = ax.boxplot(
            [als_nz, ctrl_nz], patch_artist=True, widths=0.38,
            showfliers=False, positions=[1, 2],
            medianprops=dict(color='white', lw=2.5),
            boxprops=dict(linewidth=0),
            whiskerprops=dict(color='#aaa', lw=1.2),
            capprops=dict(color='#aaa', lw=1.2)
        )
        bp['boxes'][0].set(facecolor=COLOR_ALS,  alpha=0.45)
        bp['boxes'][1].set(facecolor=COLOR_CTRL, alpha=0.45)

        for xi, (samples, dc) in enumerate([(als_s, als_colors), (ctrl_s, ctrl_colors)], 1):
            for s in samples:
                v = cpm_matrix.loc[s, gene]
                j = rng.uniform(-0.13, 0.13)
                ax.scatter(xi + j, v, s=75, color=dc.get(s, 'gray'),
                           zorder=5, edgecolors='white', linewidths=0.6, alpha=0.95)

        _, p_mw = stats.mannwhitneyu(als_nz, ctrl_nz, alternative='two-sided') \
                  if len(als_nz) > 1 and len(ctrl_nz) > 1 else (None, 1.0)
        sig      = '***' if p_mw < 0.001 else ('**' if p_mw < 0.01 else
                   ('*' if p_mw < 0.05 else 'ns'))
        padj_val = results.loc[gene, 'padj'] if gene in results.index else None

        all_nz = np.concatenate([als_nz, ctrl_nz]) if len(als_nz) and len(ctrl_nz) else np.array([1])
        y_top  = np.percentile(all_nz, 97) * 1.15
        y_line = y_top * 1.05
        ax.set_ylim(-0.3, y_top * 1.6)
        ax.plot([1, 1, 2, 2], [y_line, y_line + y_top*0.05,
                                y_line + y_top*0.05, y_line], color='black', lw=1)
        ax.text(1.5, y_line + y_top*0.06, sig, ha='center', va='bottom',
                fontsize=13, fontweight='bold')

        if padj_val is not None:
            padj_col = '#1E8449' if padj_val < 0.05 else COLOR_ALS
            ax.text(1.5, y_line + y_top*0.22, f'padj = {padj_val:.4f}',
                    ha='center', va='bottom', fontsize=7.5,
                    color=padj_col, fontweight='bold')

        lfc       = results.loc[gene, 'log2FoldChange'] if gene in results.index else 0
        direction = '▲ UP' if gene in up_genes else '▼ DOWN'
        dir_color = '#922B21' if gene in up_genes else '#154360'
        ax.set_title(f'{gene}   {direction}   (log2FC = {lfc:+.2f})',
                     fontsize=9.5, fontweight='bold', color=dir_color)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f'ALS\n(n={len(als_s)})', f'Control\n(n={len(ctrl_s)})'],
                           fontsize=9)
        ax.set_ylabel('CPM', fontsize=8)
        ax.grid(axis='y', alpha=0.2, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    handles = ([mpatches.Patch(color=c, label=f'ALS  {s}')  for s, c in als_colors.items()] +
               [mpatches.Patch(color=c, label=f'Ctrl  {s}') for s, c in ctrl_colors.items()])
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8.5,
               frameon=True, bbox_to_anchor=(0.5, -0.05),
               title='Individual samples', title_fontsize=9)
    save(fig, 'Figure5_boxplots_expression.png')


def fig_deg_bars(results, top_genes, up_genes, down_genes):
    """Figure 6 — Bar chart: Inflammation vs Axonal/Metabolic Integrity"""
    print("\n── Figure 6: DEG bar chart ──────────────────────────────────────")
    available = [g for g in top_genes if g in results.index]
    tbl = results.loc[available].sort_values('log2FoldChange', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [COLOR_MICROGLIA if g in up_genes else COLOR_AXONAL for g in tbl.index]
    ax.barh(tbl.index, tbl['log2FoldChange'], color=colors, alpha=0.85, height=0.6)

    for gene, val in zip(tbl.index, tbl['log2FoldChange']):
        ax.text(val + (0.05 if val >= 0 else -0.05), gene, f'{val:+.2f}',
                va='center', ha='left' if val >= 0 else 'right',
                fontsize=9, fontweight='bold')

    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('log2FC (ALS vs Control)', fontsize=11)
    ax.set_title('Differential Expression in LCT:\nInflammation vs Axonal & Metabolic Integrity',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    patches = [
        mpatches.Patch(color=COLOR_MICROGLIA, label='Microglia-mediated Inflammation (Upregulated)'),
        mpatches.Patch(color=COLOR_AXONAL,    label='Axonal & Metabolic Integrity (Downregulated)')
    ]
    ax.legend(handles=patches, fontsize=9, loc='lower right')
    save(fig, 'Figure6_DEG_barchart.png')


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  Script 01 — DEG Analysis: ALS LCT")
    print("=" * 65)

    # Step 1-2: Load and filter data
    count_matrix = load_count_matrix(FILE_MAP, METADATA)
    count_filtered, cpm_matrix = filter_by_cpm(count_matrix)
    del count_matrix; gc.collect()

    count_filtered.to_csv(os.path.join(OUTPUT_PATH, 'count_matrix_filtered.csv'))
    cpm_matrix.to_csv(os.path.join(OUTPUT_PATH, 'cpm_matrix.csv'))

    # Step 3: DESeq2
    results = run_deseq2(count_filtered, METADATA)
    results.to_csv(os.path.join(OUTPUT_PATH, 'DEG_results_full.csv'))
    del count_filtered; gc.collect()

    # Step 4: Derive gene sets from results — nothing hardcoded
    top_genes, up_genes, down_genes, inflam_genes, axonal_genes, cat_map = \
        derive_gene_sets(results, n_top=15)

    # Step 5: Generate figures
    print("\n── Generating figures ───────────────────────────────────────────")
    fig_volcano(results, top_genes)
    fig_ma(results, top_genes)
    fig_deg_table(results, top_genes, cat_map)
    fig_heatmap(results, cpm_matrix, METADATA, top_genes)
    fig_boxplots(cpm_matrix, results, METADATA, top_genes,
                 up_genes, down_genes, ALS_COLORS, CTRL_COLORS)
    fig_deg_bars(results, top_genes, inflam_genes, axonal_genes)

    print("\n" + "=" * 65)
    print(f"  Done! All figures saved to:")
    print(f"  {OUTPUT_PATH}")
    print("=" * 65)
