"""
Script 8 — Oligodendrocyte Marker Analysis
============================================
Generates Figures 21-22 for the results section:
  Figure 21 — Bar plot: oligodendrocyte marker expression per region
  Figure 22 — Heatmap: individual gene expression per region

Dependencies: matplotlib, pandas, numpy, scipy, statsmodels
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import LinearSegmentedColormap
import warnings
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

COLOR_LCT     = '#3D5A80'
COLOR_DORSAL  = '#778D45'
COLOR_VENTRAL = '#CD7F6D'


# ════════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION & PATHS
# ════════════════════════════════════════════════════════════════════════════════

# Automatically find the directory where this script is saved
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

BASE_PATH = SCRIPT_DIR

DORSAL_PATH  = os.path.join(BASE_PATH, "LCT_vs_dorsal_ventral", "Dorsal")
VENTRAL_PATH = os.path.join(BASE_PATH, "LCT_vs_dorsal_ventral", "Ventral")

HEALTHY_DORSAL  = os.path.join(BASE_PATH, "Healthy_hexpaint", "Dorsal")
ALS_DORSAL      = os.path.join(BASE_PATH, "Disease_hexpaint", "Dorsal")
HEALTHY_VENTRAL = os.path.join(BASE_PATH, "Healthy_hexpaint", "Ventral")
ALS_VENTRAL     = os.path.join(BASE_PATH, "Disease_hexpaint", "Ventral")

# Ensure sub-directories exist for downstream caching
os.makedirs(DORSAL_PATH, exist_ok=True)
os.makedirs(VENTRAL_PATH, exist_ok=True)

# Output directory for generated figures
OUTPUT_PATH = os.path.join(BASE_PATH, 'Final_figures', 'Oligodendrocytes')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── Analysis configurations ───────────────────────────────────────────────────
WM_CLUSTERS = ['4', '5', '6', '7']  # Stored as strings for alignment with L1 label types

# ── Oligodendrocyte gene sets ─────────────────────────────────────────────────
MATURE_OLIGO = ['ERMN', 'MAG', 'CLDN11', 'PLP1', 'MYRF', 'MOG', 'MOBP', 'MBP']
OPC_MARKERS  = ['OLIG1', 'OLIG2', 'SOX10', 'PDGFRA']
CHOL_SYNTH   = ['DHCR7', 'CYP51A1', 'SQLE', 'FDFT1', 'HMGCR']
ALL_OLIGO    = MATURE_OLIGO + OPC_MARKERS + CHOL_SYNTH

OLIGO_CATEGORIES = {
    'Mature oligodendrocytes':     MATURE_OLIGO,
    'OPC markers':                 OPC_MARKERS,
    'Oligo cholesterol synthesis': CHOL_SYNTH,
}

OLIGO_CAT_COLOR = {
    'Mature oligodendrocytes':     COLOR_AXONAL,
    'OPC markers':                 COLOR_MICROGLIA,
    'Oligo cholesterol synthesis': COLOR_LIPID,
}


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADING & DEG ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

def load_and_run_deg(healthy_path, als_path, output_csv, region_name):
    """Load or compute pseudo-bulk DEG analysis for a region."""
    if os.path.exists(output_csv):
        print(f"  {region_name}: loading existing DEG results")
        return pd.read_csv(output_csv, index_col=0)

    print(f"\n── {region_name}: loading raw data ─────────────────────────────")
    file_list = []
    groups    = {}

    if os.path.exists(healthy_path):
        for f in os.listdir(healthy_path):
            if f.endswith('.gz'):
                sid = f.split('_selection')[0]
                file_list.append((os.path.join(healthy_path, f), sid, 'Control'))
                groups[sid] = 'Control'

    if os.path.exists(als_path):
        for f in os.listdir(als_path):
            if f.endswith('.gz'):
                sid = f.split('_selection')[0]
                file_list.append((os.path.join(als_path, f), sid, 'ALS'))
                groups[sid] = 'ALS'

    if not file_list:
        print(f"  WARNING: No data discovered for {region_name} mapping routes. Returning empty fallback.")
        return pd.DataFrame(columns=['log2FoldChange', 'pvalue', 'baseMean', 'padj'])

    pseudo_bulk = {}
    for full_path, sid, group in file_list:
        df = pd.read_csv(full_path, compression='gzip')
        df['L1_region_cluster'] = df['L1_region_cluster'].astype(str)
        df = df[df['L1_region_cluster'].isin(WM_CLUSTERS)]
        pseudo_bulk[sid] = df.groupby('geneName')['MIDCount'].sum()

    count_matrix  = pd.DataFrame(pseudo_bulk).fillna(0).astype(int)
    sample_groups = pd.Series(groups)
    als_samples   = sample_groups[sample_groups == 'ALS'].index.tolist()
    ctrl_samples  = sample_groups[sample_groups == 'Control'].index.tolist()

    lib_sizes   = count_matrix.sum(axis=0)
    median_lib  = lib_sizes.median()
    norm_matrix = count_matrix.divide(lib_sizes, axis=1) * median_lib

    results = []
    for gene in count_matrix.index:
        als_vals  = norm_matrix.loc[gene, als_samples].values if als_samples else np.array([0])
        ctrl_vals = norm_matrix.loc[gene, ctrl_samples].values if ctrl_samples else np.array([0])
        if als_vals.mean() == 0 and ctrl_vals.mean() == 0: continue
        
        mean_als     = als_vals.mean() + 0.1
        mean_control = ctrl_vals.mean() + 0.1
        log2fc    = np.log2(mean_als / mean_control)
        baseMean  = (als_vals.mean() + ctrl_vals.mean()) / 2
        
        try:
            _, pvalue = mannwhitneyu(als_vals, ctrl_vals, alternative='two-sided')
        except:
            pvalue = 1.0
            
        results.append({'geneName': gene, 'log2FoldChange': log2fc,
                        'pvalue': pvalue, 'baseMean': baseMean})

    results_df = pd.DataFrame(results).set_index('geneName')
    if not results_df.empty:
        _, padj, _, _ = multipletests(results_df['pvalue'], method='fdr_bh')
        results_df['padj'] = padj
    else:
        results_df['padj'] = pd.Series(dtype=float)
        
    results_df.to_csv(output_csv)
    return results_df


def compute_region_summary(deg_df, region_name):
    """Compute mean log2FC and significance per oligodendrocyte category."""
    summary = {}
    if deg_df.empty: return summary
    
    for cat, genes in OLIGO_CATEGORIES.items():
        found = deg_df[deg_df.index.isin(genes)]
        if len(found) == 0: continue
        mean_lfc = found['log2FoldChange'].mean()
        
        if len(found) > 1:
            _, p_val = stats.ttest_1samp(found['log2FoldChange'], 0)
        else:
            p_val = 1.0  # Cannot perform a T-test on single element dimensions safely
            
        summary[cat] = {
            'mean_lfc': mean_lfc,
            'sem':      found['log2FoldChange'].sem() if len(found) > 1 else 0,
            'p':        p_val,
            'n_sig':    (found['padj'] < 0.05).sum() if 'padj' in found.columns else 0,
        }
    return summary


def save(fig, name, dpi=300):
    path = os.path.join(OUTPUT_PATH, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def fig_oligo_barplot(lct_deg, dorsal_deg, ventral_deg):
    """
    Figure 21 — Bar plot: mean log2FC per oligodendrocyte category per region.
    Three bars per category (LCT, Dorsal, Ventral).
    """
    print("\n── Figure 21: Oligodendrocyte bar plot ──────────────────────────")

    regions = {
        'LCT':     (lct_deg,     COLOR_LCT),
        'Dorsal':  (dorsal_deg,  COLOR_DORSAL),
        'Ventral': (ventral_deg, COLOR_VENTRAL),
    }

    categories = list(OLIGO_CATEGORIES.keys())
    x      = np.arange(len(categories))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for ri, (region_name, (deg_df, color)) in enumerate(regions.items()):
        summary = compute_region_summary(deg_df, region_name)
        means   = [summary.get(cat, {}).get('mean_lfc', 0) for cat in categories]
        sems    = [summary.get(cat, {}).get('sem', 0)      for cat in categories]
        p_vals  = [summary.get(cat, {}).get('p', 1)        for cat in categories]

        offset = (ri - 1) * width
        ax.bar(x + offset, means, width, color=color, alpha=0.85,
               label=region_name, yerr=sems, capsize=4,
               error_kw={'linewidth': 1})

        for i, (mean, p) in enumerate(zip(means, p_vals)):
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            if sig:
                y_pos = (mean + sems[i] + 0.02) if mean >= 0 else (mean - sems[i] - 0.04)
                va    = 'bottom' if mean >= 0 else 'top'
                ax.text(x[i] + offset, y_pos, sig, ha='center', va=va,
                        fontsize=11, fontweight='bold')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5)
    ax.set_ylabel('Average log2FC (ALS vs Control)', fontsize=10.5)
    ax.set_title('Oligodendrocyte dysregulation in ALS\nper white matter region (WM filtered)',
                 fontsize=12, fontweight='bold')

    region_patches = [mpatches.Patch(color=COLOR_LCT,     label='LCT'),
                      mpatches.Patch(color=COLOR_DORSAL,  label='Dorsal'),
                      mpatches.Patch(color=COLOR_VENTRAL, label='Ventral')]
    ax.legend(handles=region_patches, fontsize=9.5, loc='upper right')

    sig_text = "* p < 0.05     ** p < 0.01     *** p < 0.001"
    fig.text(0.5, -0.02, sig_text, ha='center', fontsize=8.5, style='italic')
    save(fig, 'Figure21_oligo_barplot.png')


def fig_oligo_heatmap(lct_deg, dorsal_deg, ventral_deg):
    """
    Figure 22 — Heatmap: individual gene log2FC per region.
    Genes grouped by oligodendrocyte category with a clean legend.
    """
    print("\n── Figure 22: Oligodendrocyte heatmap ───────────────────────────")

    gene_order = MATURE_OLIGO + OPC_MARKERS + CHOL_SYNTH
    regions    = {'LCT': lct_deg, 'Dorsal': dorsal_deg, 'Ventral': ventral_deg}

    mat = pd.DataFrame(index=gene_order, columns=regions.keys())
    for region_name, deg_df in regions.items():
        for gene in gene_order:
            if deg_df is not None and gene in deg_df.index:
                mat.loc[gene, region_name] = deg_df.loc[gene, 'log2FoldChange']
            else:
                mat.loc[gene, region_name] = np.nan
    mat = mat.astype(float)

    # Compute mean log2FC for each category across regions
    cat_means = {}
    for cat, genes in OLIGO_CATEGORIES.items():
        cat_means[cat] = mat.loc[mat.index.isin(genes)].mean(axis=0)

    # Build matrix with summary rows injected
    extended_rows = []
    row_labels = []
    is_summary_row = []
    row_colors = []

    for cat, genes in OLIGO_CATEGORIES.items():
        # Add individual genes
        for g in genes:
            extended_rows.append(mat.loc[g].values)
            row_labels.append(g)
            is_summary_row.append(False)
            row_colors.append(OLIGO_CAT_COLOR[cat])
        # Inject Summary Row
        extended_rows.append(cat_means[cat].values)
        row_labels.append("MEAN")
        is_summary_row.append(True)
        row_colors.append(OLIGO_CAT_COLOR[cat])

    extended_mat = pd.DataFrame(extended_rows, index=row_labels, columns=regions.keys())

    fig, ax = plt.subplots(figsize=(6.0, len(extended_mat) * 0.25 + 2))
    vmax = max(abs(mat.values[~np.isnan(mat.values)]).max(), 0.5)
    cmap_custom = LinearSegmentedColormap.from_list(
        'als_custom',
        ['#588157', '#F5F0F0', '#C2527A'],
        N=256
    )

    im = ax.imshow(extended_mat.values, cmap=cmap_custom, aspect='auto',
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(list(regions.keys()), fontsize=8.5, fontweight='bold')
    
    ax.get_xticklabels()[0].set_color(COLOR_LCT)
    ax.get_xticklabels()[1].set_color(COLOR_DORSAL)
    ax.get_xticklabels()[2].set_color(COLOR_VENTRAL)

    ax.set_yticks(range(len(extended_mat)))
    ax.set_yticklabels(extended_mat.index, fontsize=7)

    # Color labels and format fonts
    for ti, (label, is_summary, col) in enumerate(zip(extended_mat.index, is_summary_row, row_colors)):
        if is_summary:
            ax.get_yticklabels()[ti].set_weight('bold')
            ax.get_yticklabels()[ti].set_fontsize(7.5)
        ax.get_yticklabels()[ti].set_color(col)

    # Overlay text values inside heatmap cells
    for i in range(len(extended_mat)):
        for j, region in enumerate(regions.keys()):
            val = extended_mat.iloc[i, j]
            if not np.isnan(val):
                is_sum = is_summary_row[i]
                font_w = 'bold' if is_sum else 'normal'
                font_s = 7.5 if is_sum else 6.5
                
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=font_s, color='white' if abs(val) > vmax*0.5 else 'black',
                        fontweight=font_w)

    # Visual separation lines separating the subcategory blocks
    current_idx = 0
    for cat, genes in OLIGO_CATEGORIES.items():
        current_idx += len(genes) + 1  # count individual genes + the injected summary row
        if current_idx < len(extended_mat):
            ax.axhline(current_idx - 0.5, color='white', lw=3.0)

    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.08)
    cbar.set_label('log2FC (ALS vs Control)', fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    legend_patches = [
        mpatches.Patch(color=COLOR_AXONAL, label='Mature oligodendrocytes'),
        mpatches.Patch(color=COLOR_MICROGLIA, label='OPC markers'),
        mpatches.Patch(color=COLOR_LIPID, label='Oligo cholesterol synthesis')
    ]
    
    ax.legend(handles=legend_patches, 
              bbox_to_anchor=(1.12, 0.0),  
              loc='lower left', 
              fontsize=7.5, 
              frameon=False,               
              title='Gene Categories',    
              title_fontsize=8)
    ax.get_legend().get_title().set_weight('bold')
    ax.set_title('Oligodendrocyte gene expression per region\n(WM filtered)',
                 fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save(fig, 'Figure22_oligo_heatmap.png')

# ════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTOR
# ════════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Script 8 — Oligodendrocyte Marker Analysis")
    print("=" * 65)

    # ── Step 1: Load or compute DEG results per region ────────────────────────
    lct_path_filtered = os.path.join(BASE_PATH, 'LCT_WM_filtered_DEG.csv')
    lct_path_full     = os.path.join(BASE_PATH, 'DEG_results_full.csv')
    
    if os.path.exists(lct_path_filtered):
        lct_deg = pd.read_csv(lct_path_filtered, index_col=0)
    elif os.path.exists(lct_path_full):
        lct_deg = pd.read_csv(lct_path_full, index_col=0)
    else:
        print("  WARNING: LCT base tracking reference file not found. Initializing empty dataset structure.")
        lct_deg = pd.DataFrame(columns=['log2FoldChange', 'pvalue', 'baseMean', 'padj'])

    dorsal_deg  = load_and_run_deg(
        HEALTHY_DORSAL, ALS_DORSAL,
        os.path.join(DORSAL_PATH, 'dorsal_DEG_results.csv'), 'Dorsal')

    ventral_deg = load_and_run_deg(
        HEALTHY_VENTRAL, ALS_VENTRAL,
        os.path.join(VENTRAL_PATH, 'ventral_DEG_results.csv'), 'Ventral')

    # ── Step 2: Generate figures ──────────────────────────────────────────────
    print("\n── Generating figures ───────────────────────────────────────────")
    fig_oligo_barplot(lct_deg, dorsal_deg, ventral_deg)
    fig_oligo_heatmap(lct_deg, dorsal_deg, ventral_deg)

    print("\n" + "=" * 65)
    print(f"  Done! All figures saved to:")
    print(f"  {OUTPUT_PATH}")
    print("=" * 65)


if __name__ == '__main__':
    main()
