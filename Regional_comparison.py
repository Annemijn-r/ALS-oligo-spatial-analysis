"""
Script 05 — Regional White Matter Comparison: LCT vs Dorsal vs Ventral
=======================================================================
Generates Figures 7-11 for the results section:
  Figure 7  — Scatter plot: LCT vs Dorsal (individual genes per pathway)
  Figure 8  - Bar plot: LCT vs Dorsal
  Figure 9  — Scatter plot: LCT vs Ventral (individual genes per pathway)
  Figure 10 - Bar plot: LCT vs Ventral
  Figure 11 — Combined bar plot: average log2FC per pathway per region

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
from matplotlib.ticker import FormatStrFormatter
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

# ── Paths ─────────────────────────────────────────────────────────────────────
# Determine base directory dynamically based on the location of this script
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
BASE_PATH       = os.path.join(SCRIPT_DIR, "Figures-plots2")

DORSAL_PATH     = os.path.join(BASE_PATH, "LCT_vs_dorsal_ventral", "Dorsal") + os.sep
VENTRAL_PATH    = os.path.join(BASE_PATH, "LCT_vs_dorsal_ventral", "Ventral") + os.sep
OUTPUT_PATH     = os.path.join(SCRIPT_DIR, "Final_figures", "Regional_comparison") + os.sep
HEALTHY_DORSAL  = os.path.join(SCRIPT_DIR, "Healthy_hexpaint", "Dorsal") + os.sep
ALS_DORSAL      = os.path.join(SCRIPT_DIR, "Disease_hexpaint", "Dorsal") + os.sep
HEALTHY_VENTRAL = os.path.join(SCRIPT_DIR, "Healthy_hexpaint", "Ventral") + os.sep
ALS_VENTRAL     = os.path.join(SCRIPT_DIR, "Disease_hexpaint", "Ventral") + os.sep

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── Pathway gene sets ─────────────────────────────────────────────────────────
PATHWAY_GENES = {
    "Microglia activation": [
        "CHIT1", "LYZ", "FCGR2B", "CD52", "PLAU", "C1QA", "C1QB", "TYROBP",
        "TREM2", "ITGB2", "GPNMB", "C3", "CTSS", "P2RY12", "AIF1"
    ],
    "Lipid metabolism": [
        "HMGCR", "FDFT1", "SQLE", "CYP51A1", "DDIT4", "PDK4",
        "FASN", "SCD", "FADS1", "APOE", "ABCA1", "NPC1"
    ],
    "Axonal degeneration": [
        "NEFL", "NEFM", "NEFH", "KIF5A", "DYNC1H1", "DCTN1",
        "APP", "STMN2", "SYP", "SNAP25", "MBP", "PLP1", "MAG"
    ],
    "Mitochondrial function": [
        "TOMM20", "TOMM40", "TIMM23", "TIMM44", "COX4I1", "COX5A", "COX6A1",
        "COX7A2", "MT-CO1", "MT-CO2", "NDUFS1", "NDUFS3", "SDHA", "SDHB",
        "UQCRC1", "ATP5F1A", "ATP5F1B", "MFN1", "MFN2", "OPA1", "DRP1",
        "DNM1L", "PINK1", "PRKN", "FIS1", "PDK1", "PDK2", "PDK4", "PDHA1"
    ],
}

PATHWAY_COLOR = {
    "Microglia activation":   COLOR_MICROGLIA,
    "Lipid metabolism":       COLOR_LIPID,
    "Axonal degeneration":    COLOR_AXONAL,
    "Mitochondrial function": COLOR_MITO,
}

WM_CLUSTERS = [4, 5, 6, 7]
PATHWAYS = list(PATHWAY_GENES.keys())

# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADING & DEG ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

def load_and_run_deg(healthy_path, als_path, output_csv, region_name):
    """Load raw hexbin data, filter WM clusters, run pseudo-bulk DEG analysis."""
    if os.path.exists(output_csv):
        print(f"  {region_name}: loading existing DEG results from {output_csv}")
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
        print(f"  Warning: No data found for {region_name}. Returning empty DataFrame.")
        return pd.DataFrame(columns=['log2FoldChange', 'pvalue', 'baseMean', 'padj'])

    pseudo_bulk = {}
    for full_path, sid, group in file_list:
        df = pd.read_csv(full_path, compression='gzip')
        df = df[df['L1_region_cluster'].isin(WM_CLUSTERS)]
        pseudo_bulk[sid] = df.groupby('geneName')['MIDCount'].sum()

    count_matrix   = pd.DataFrame(pseudo_bulk).fillna(0).astype(int)
    sample_groups  = pd.Series(groups)
    als_samples    = sample_groups[sample_groups == 'ALS'].index.tolist()
    ctrl_samples   = sample_groups[sample_groups == 'Control'].index.tolist()

    lib_sizes   = count_matrix.sum(axis=0)
    median_lib  = lib_sizes.median()
    norm_matrix = count_matrix.divide(lib_sizes, axis=1) * median_lib

    results = []
    for gene in count_matrix.index:
        als_vals  = norm_matrix.loc[gene, als_samples].values
        ctrl_vals = norm_matrix.loc[gene, ctrl_samples].values
        if als_vals.mean() == 0 and ctrl_vals.mean() == 0:
            continue
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
    _, padj, _, _ = multipletests(results_df['pvalue'], method='fdr_bh')
    results_df['padj'] = padj
    results_df = results_df.sort_values('padj')
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv)
    print(f"  Saved: {output_csv}")
    return results_df

def build_pathway_csv(results_df, output_csv, region_name):
    """Extract pathway genes from DEG results and save as CSV."""
    if os.path.exists(output_csv):
        print(f"  {region_name}: loading existing pathway CSV")
        return pd.read_csv(output_csv, index_col=0)

    if results_df.empty:
        return pd.DataFrame()

    all_hits = []
    for pathway, genes in PATHWAY_GENES.items():
        found = results_df[results_df.index.isin(genes)].copy()
        found['pathway'] = pathway
        all_hits.append(found)

    pathway_df = pd.concat(all_hits)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pathway_df.to_csv(output_csv)
    print(f"  Saved: {output_csv}")
    return pathway_df

def save(fig, name, dpi=300):
    path = os.path.join(OUTPUT_PATH, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}")

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def fig_scatter_region(lct_df, region_df, region_name, fig_number):
    """Figure 7/9 — Scatter: LCT log2FC vs region log2FC per gene per pathway."""
    print(f"\n── Figure {fig_number}: Scatter LCT vs {region_name} ──────────────────────")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

    for ax, pathway in zip(axes, PATHWAYS):
        color = PATHWAY_COLOR[pathway]
        pathway_gene_list = PATHWAY_GENES[pathway]

        if lct_df.empty or region_df.empty:
            common_genes = []
        else:
            common_genes = [g for g in pathway_gene_list if g in lct_df.index and g in region_df.index]
        
        if len(common_genes) == 0:
            ax.axhline(0, color='gray', linewidth=0.6, linestyle='--', zorder=1)
            ax.axvline(0, color='gray', linewidth=0.6, linestyle='--', zorder=1)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_xlabel('LCT log2FC', fontsize=11)
            ax.set_ylabel(f'{region_name} log2FC', fontsize=11)
            ax.set_title(pathway, fontsize=11, fontweight='bold', color=color)
            ax.set_aspect('equal')
            continue

        lct_vals = lct_df.loc[common_genes, 'log2FoldChange']
        reg_vals = region_df.loc[common_genes, 'log2FoldChange']
        
        lct_sig = lct_df.loc[common_genes, 'padj'] < 0.05
        reg_sig = region_df.loc[common_genes, 'padj'] < 0.05
        either_sig = (lct_sig | reg_sig).to_dict()

        sig_genes   = [g for g in common_genes if either_sig[g]]
        insig_genes = [g for g in common_genes if not either_sig[g]]

        if insig_genes:
            ax.scatter(lct_vals.loc[insig_genes], reg_vals.loc[insig_genes],
                       color=color, alpha=0.35, s=50, zorder=2)
            
        if sig_genes:
            ax.scatter(lct_vals.loc[sig_genes], reg_vals.loc[sig_genes],
                       color=color, alpha=1.0, s=100, edgecolors='black',
                       linewidths=0.8, zorder=3)

        for gene in sig_genes:
            x_coord = lct_vals[gene].iloc[0] if isinstance(lct_vals[gene], pd.Series) else lct_vals[gene]
            y_coord = reg_vals[gene].iloc[0] if isinstance(reg_vals[gene], pd.Series) else reg_vals[gene]
            
            ax.annotate(gene, (float(x_coord), float(y_coord)),
                        fontsize=8, ha='left', va='bottom',
                        xytext=(4, 4), textcoords='offset points')

        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--', zorder=1)
        ax.axvline(0, color='gray', linewidth=0.6, linestyle='--', zorder=1)
        
        valid_lct = lct_vals[np.isfinite(lct_vals)]
        valid_reg = reg_vals[np.isfinite(reg_vals)]
        
        if len(valid_lct) > 0 and len(valid_reg) > 0:
            lim = max(abs(valid_lct).max(), abs(valid_reg).max()) * 1.2
        else:
            lim = 1.0
            
        ax.plot([-lim, lim], [-lim, lim], color='lightgray', linewidth=0.8,
                linestyle=':', zorder=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        r, p = stats.spearmanr(lct_vals, reg_vals)
        ax.text(0.05, 0.95, f'r = {r:.2f}\np = {p:.3f}',
                transform=ax.transAxes, fontsize=9, va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='lightgray',
                          boxstyle='round,pad=0.3'))

        ax.set_xlabel('LCT log2FC', fontsize=11)
        ax.set_ylabel(f'{region_name} log2FC', fontsize=11)
        ax.set_title(pathway, fontsize=11, fontweight='bold', color=color)
        ax.set_aspect('equal')
        print(f"  {pathway}: {len(common_genes)} genes | r={r:.2f} p={p:.3f}")

    sig_patch   = mpatches.Patch(facecolor='gray', edgecolor='black', label='padj < 0.05 (LCT or region)')
    insig_patch = mpatches.Patch(facecolor='gray', alpha=0.35, label='padj ≥ 0.05')
    fig.legend(handles=[sig_patch, insig_patch], loc='lower center', ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(f'Gene-level log2FC comparison: LCT vs {region_name}\n(large dots = padj < 0.05 | dotted line = x=y)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save(fig, f'Figure{fig_number}_scatter_LCT_vs_{region_name}.png')

def fig_barplot_region(lct_df, region_df, region_name, fig_number):
    """Figure 8/10 — Grouped Bar Chart: Pathway-level mean log2FC comparison."""
    print(f"\n── Figure {fig_number}: Barplot LCT vs {region_name} ──────────────────────")
    fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.35
    x = np.arange(len(PATHWAYS))

    lct_vals = []
    reg_vals = []
    lct_sig_markers = []
    reg_sig_markers = []
    colors = []

    for pathway in PATHWAYS:
        color = PATHWAY_COLOR[pathway]
        pathway_gene_list = PATHWAY_GENES[pathway]
        colors.append(color)

        if lct_df.empty or region_df.empty:
            common_genes = []
        else:
            common_genes = [g for g in pathway_gene_list if g in lct_df.index and g in region_df.index]

        if len(common_genes) == 0:
            lct_vals.append(0)
            reg_vals.append(0)
            lct_sig_markers.append(False)
            reg_sig_markers.append(False)
            continue

        mean_lct = lct_df.loc[common_genes, 'log2FoldChange'].mean()
        mean_reg = region_df.loc[common_genes, 'log2FoldChange'].mean()
        lct_vals.append(mean_lct if np.isfinite(mean_lct) else 0)
        reg_vals.append(mean_reg if np.isfinite(mean_reg) else 0)

        _, p_lct = stats.ttest_1samp(lct_df.loc[common_genes, 'log2FoldChange'].dropna(), 0)
        _, p_reg = stats.ttest_1samp(region_df.loc[common_genes, 'log2FoldChange'].dropna(), 0)
        
        lct_sig_markers.append(p_lct < 0.05)
        reg_sig_markers.append(p_reg < 0.05)
        
        print(f"  {pathway}: LCT mean={mean_lct:.2f} (p={p_lct:.2f}) | {region_name} mean={mean_reg:.2f} (p={p_reg:.2f})")

    bars_lct = ax.bar(x - width/2, lct_vals, width, color=colors, alpha=1.0, edgecolor=None, zorder=2)
    bars_reg = ax.bar(x + width/2, reg_vals, width, color=colors, alpha=0.45, edgecolor=None, zorder=2)

    for i, (lv, rv, ls, rs) in enumerate(zip(lct_vals, reg_vals, lct_sig_markers, reg_sig_markers)):
        if ls:
            offset = 0.04 if lv >= 0 else -0.06
            ax.text(i - width/2, lv + offset, '*', ha='center', va='bottom' if lv >= 0 else 'top',
                    fontsize=14, fontweight='bold', color='black', zorder=5)
        if rs:
            offset = 0.04 if rv >= 0 else -0.06
            ax.text(i + width/2, rv + offset, '*', ha='center', va='bottom' if rv >= 0 else 'top',
                    fontsize=14, fontweight='bold', color='black', zorder=5)

    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False)

    ax.set_xticks(x)
    ax.set_xticklabels(PATHWAYS, fontsize=10)
    
    ax.set_ylabel('Average log2FoldChange (ALS vs Control)', fontsize=11)
    ax.set_title(f'Pathway Dysregulation: LCT vs {region_name}\n(* = t-test p < 0.05)',
                 fontsize=13, fontweight='bold')

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    lct_patch = mpatches.Patch(facecolor='gray', alpha=1.0, label='LCT')
    reg_patch = mpatches.Patch(facecolor='gray', alpha=0.45, label=f'{region_name} Column')
    ax.legend(handles=[lct_patch, reg_patch], fontsize=10, loc='upper right', frameon=False)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=10, clip_on=False)

    plt.tight_layout()
    save(fig, f'Figure{fig_number}_barplot_LCT_vs_{region_name}.png')

def fig_combined_barplot(lct_df, dorsal_df, ventral_df):
    """Figure 11 — Combined bar plot: mean log2FC per pathway for LCT, Dorsal, Ventral."""
    print("\n── Figure 11: Combined regional bar plot ─────────────────────────")

    results = {}
    for region_name, region_df in [("Dorsal", dorsal_df), ("Ventral", ventral_df)]:
        results[region_name] = {}
        for pathway in PATHWAYS:
            pathway_gene_list = PATHWAY_GENES[pathway]
            
            if lct_df.empty:
                lct_available = pd.Index([])
            else:
                lct_available = lct_df.index.intersection(pathway_gene_list)

            if region_df.empty:
                reg_available = pd.Index([])
            else:
                reg_available = region_df.index.intersection(pathway_gene_list)

            lct_vals    = lct_df.loc[lct_available, 'log2FoldChange'].dropna().values if not lct_available.empty else np.array([])
            region_vals = region_df.loc[reg_available, 'log2FoldChange'].dropna().values if not reg_available.empty else np.array([])
            results[region_name][pathway] = {'LCT': lct_vals, 'Region': region_vals}

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig2.suptitle('Average log2FC per Pathway\nLCT vs Region (ALS vs Control)', fontsize=13, fontweight='bold')

    for ax2, region_name in zip(axes2, ["Dorsal", "Ventral"]):
        x      = np.arange(len(PATHWAYS))
        width  = 0.35
        colors = [PATHWAY_COLOR[p] for p in PATHWAYS]

        lct_means  = [np.mean(results[region_name][p]['LCT']) if len(results[region_name][p]['LCT']) > 0 else 0 for p in PATHWAYS]
        reg_means  = [np.mean(results[region_name][p]['Region']) if len(results[region_name][p]['Region']) > 0 else 0 for p in PATHWAYS]
        lct_sems   = [stats.sem(results[region_name][p]['LCT'])   if len(results[region_name][p]['LCT'])    > 1 else 0 for p in PATHWAYS]
        reg_sems   = [stats.sem(results[region_name][p]['Region'])if len(results[region_name][p]['Region']) > 1 else 0 for p in PATHWAYS]

        ax2.bar(x - width/2, reg_means, width, color=colors, alpha=0.35,
                label='Region', yerr=reg_sems, capsize=4, error_kw={'linewidth': 1})
        ax2.bar(x + width/2, lct_means, width, color=colors, alpha=1.0,
                label='LCT',    yerr=lct_sems, capsize=4, error_kw={'linewidth': 1})

        for i, pathway in enumerate(PATHWAYS):
            lct_v = results[region_name][pathway]['LCT']
            reg_v = results[region_name][pathway]['Region']
            if len(lct_v) >= 3 and len(reg_v) >= 3:
                _, p = stats.mannwhitneyu(lct_v, reg_v, alternative='two-sided')
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                if sig:
                    if lct_means[i] >= 0:
                        y_pos = max(lct_means[i] + lct_sems[i], reg_means[i] + reg_sems[i]) + 0.08
                        va = 'bottom'
                    else:
                        y_pos = min(lct_means[i] - lct_sems[i], reg_means[i] - reg_sems[i]) - 0.08
                        va = 'top'
                    ax2.text(i, y_pos, sig, ha='center', va=va, fontsize=12, fontweight='bold')

        ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax2.set_xticks(x)
        ax2.set_xticklabels(PATHWAYS, rotation=15, ha='right', fontsize=10)
        ax2.set_ylabel('Average log2FC (ALS vs Control)', fontsize=11)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.set_title(f'LCT vs {region_name} Column', fontsize=11, fontweight='bold')

    regio_patch = mpatches.Patch(facecolor='gray', alpha=0.35, label='Region')
    lct_patch   = mpatches.Patch(facecolor='gray', alpha=1.0,  label='LCT')
    fig2.legend(handles=[regio_patch, lct_patch], loc='lower center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.08))
    sig_text = "* p < 0.05     ** p < 0.01     *** p < 0.001"
    fig2.text(0.5, -0.13, sig_text, ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    save(fig2, 'Figure11_regional_comparison_barplot.png')

# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  Script 02 — Regional White Matter Comparison")
    print("=" * 65)

    # ── Step 1: Run or load DEG analysis per region ───────────────────────────
    dorsal_deg  = load_and_run_deg(HEALTHY_DORSAL, ALS_DORSAL, os.path.join(DORSAL_PATH, 'dorsal_DEG_results.csv'), 'Dorsal')
    ventral_deg = load_and_run_deg(HEALTHY_VENTRAL, ALS_VENTRAL, os.path.join(VENTRAL_PATH, 'ventral_DEG_results.csv'), 'Ventral')

    # ── Step 2: Build or load pathway CSVs ───────────────────────────────────
    dorsal_pw  = build_pathway_csv(dorsal_deg,  os.path.join(DORSAL_PATH, 'dorsal_pathway_genes.csv'),  'Dorsal')
    ventral_pw = build_pathway_csv(ventral_deg, os.path.join(VENTRAL_PATH, 'ventral_pathway_genes.csv'), 'Ventral')
    
    lct_pw_path = os.path.join(BASE_PATH, 'pathway_genes_all.csv')
    if os.path.exists(lct_pw_path):
        lct_pw  = pd.read_csv(lct_pw_path, index_col=0)
    else:
        print(f"  Warning: {lct_pw_path} not found. Creating empty fallback DataFrame.")
        lct_pw  = pd.DataFrame(columns=['log2FoldChange', 'padj', 'pathway'])

    # ── Step 3: Generate figures ──────────────────────────────────────────────
    print("\n── Generating figures ───────────────────────────────────────────")
    fig_scatter_region(lct_pw, dorsal_pw,  'Dorsal',  7)
    fig_scatter_region(lct_pw, ventral_pw, 'Ventral', 9)
    fig_barplot_region(lct_pw, dorsal_pw,  'Dorsal',  '8_bar')
    fig_barplot_region(lct_pw, ventral_pw, 'Ventral', '10_bar')
    fig_combined_barplot(lct_pw, dorsal_pw, ventral_pw)

    print("\n" + "=" * 65)
    print(f"  Done! All figures saved to:")
    print(f"  {OUTPUT_PATH}")
    print("=" * 65)
