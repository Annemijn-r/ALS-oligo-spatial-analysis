"""
Script 07 — Spearman Correlations: Microglia vs Axonal Markers
===============================================================
Generates Figures 17-20 for the results section:
  Figure 17 — Scatter plots: microglia x axonal gene expression per hexbin
  Figure 18 — Slope plot: Spearman rho Non-inflamed → Inflamed per gene pair
  Figure 19 — Spearman correlation heatmap (All / Inflamed / Non-inflamed)
  Figure 20 — Fisher z-test: ALS vs Control Spearman rho

Dependencies: matplotlib, pandas, numpy, scipy, statsmodels
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
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


# ════════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION & PATHS
# ════════════════════════════════════════════════════════════════════════════════

# Automatically find the directory where this script is saved
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

REGION_PATH = SCRIPT_DIR

OUTPUT_PATH = os.path.join(REGION_PATH, 'Final_figures', 'Spearman_correlations')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── Sample lists ──────────────────────────────────────────────────────────────
ALS_SAMPLES = [
    ('ALS S09-055', 'Disease_hexpaint/LCT/ALS_S09-055_selection.csv.gz'),
    ('ALS S13-054', 'Disease_hexpaint/LCT/ALS_S13-054_selection.csv.gz'),
    ('ALS S13-047', 'Disease_hexpaint/LCT/ALS_S13-047_selection.csv.gz'),
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

# ── Gene sets ─────────────────────────────────────────────────────────────────
MICROGLIA_GENES = ['CHIT1', 'LYZ']
AXONAL_GENES    = ['PDK4', 'DDIT4', 'CYP51A1', 'PLAU']
ALL_GENES       = MICROGLIA_GENES + AXONAL_GENES
METABOLIC       = ['PDK4', 'DDIT4']
MYELIN_ECM      = ['CYP51A1', 'PLAU']

INFLAMED_CLUSTERS     = ['6', '7']
NON_INFLAMED_CLUSTERS = ['1', '3', '4', '5']

GENE_CATEGORIES = {
    'Metabolic stress':     ['PDK4', 'DDIT4'],
    'Myelin/ECM':           ['CYP51A1', 'PLAU'],
    'Microglia activation': ['CHIT1', 'LYZ'],
}

CATEGORY_COLORS = {
    'Metabolic stress':     COLOR_LIPID,
    'Myelin/ECM':           COLOR_AXONAL,
    'Microglia activation': COLOR_MICROGLIA,
}

ALS_SAMPLE_COLORS     = {'S09-055': '#6D3A4A',
                         'S13-047': '#8C6275',
                         'S13-054': '#A67F8E',
                         'S18-070': '#C4A0AD',
                         'S19-061': '#DEC3CB',
                        }
CONTROL_SAMPLE_COLORS = {'S12-186': '#2E5E4A',
                         'S13-133': '#3D7A60',
                         'S15-025': '#688A7A',
                         'S15-051': '#8FAF9F',
                         'S17-062': '#B5CFC5',
                        }


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADING & NORMALIZATION
# ════════════════════════════════════════════════════════════════════════════════

def load_and_normalize(filepath, condition_label):
    available_cols = pd.read_csv(filepath, nrows=0).columns.tolist()
    has_midcount   = 'MIDCount' in available_cols
    has_hexbin_id  = 'hexbin_id' in available_cols

    usecols = ['L1_region_cluster', 'geneName']
    if has_midcount:  usecols.append('MIDCount')
    if has_hexbin_id: usecols.append('hexbin_id')
    else:
        for c in ['hexbin_i', 'hexbin_j']:
            if c in available_cols: usecols.append(c)

    df = pd.read_csv(filepath, usecols=lambda c: c in usecols)
    df['L1_region_cluster'] = df['L1_region_cluster'].astype(str)

    if 'hexbin_id' not in df.columns:
        df['hexbin_id'] = (df['hexbin_i'].astype(str) + '_' +
                           df['hexbin_j'].astype(str))

    if has_midcount:
        hexbin_totals = df.groupby('hexbin_id')['MIDCount'].sum().rename('total')
    else:
        hexbin_totals = df.groupby('hexbin_id')['geneName'].count().rename('total')

    df_genes = df[df['geneName'].isin(ALL_GENES)].copy()
    df_genes['lct_label'] = df_genes['L1_region_cluster'].apply(
        lambda x: 'Inflamed'     if x in INFLAMED_CLUSTERS else
                  'Non-inflamed' if x in NON_INFLAMED_CLUSTERS else 'Other'
    )
    df_genes = df_genes[df_genes['lct_label'] != 'Other'].copy()

    count_col = 'MIDCount' if has_midcount else 'L1_region_cluster'
    aggfunc   = 'sum'      if has_midcount else 'count'
    pivot = df_genes.pivot_table(
        index=['hexbin_id', 'lct_label'], columns='geneName',
        values=count_col, aggfunc=aggfunc, fill_value=0
    ).reset_index()

    pivot = pivot.merge(hexbin_totals.reset_index(), on='hexbin_id', how='left')
    gene_cols = [c for c in pivot.columns if c not in ['hexbin_id', 'lct_label', 'total']]
    pivot[gene_cols] = pivot[gene_cols].div(pivot['total'].replace(0, 1), axis=0) * 10_000
    pivot = pivot.drop(columns=['total'])
    pivot['condition'] = condition_label
    return pivot


def compute_correlations(pivot, sample_name):
    rows = []
    for group in ['All', 'Inflamed', 'Non-inflamed']:
        subset = pivot if group == 'All' else pivot[pivot['lct_label'] == group]
        if len(subset) < 10: continue
        for mg in MICROGLIA_GENES:
            for ag in AXONAL_GENES:
                if mg not in subset.columns or ag not in subset.columns: continue
                x    = subset[mg].values
                y    = subset[ag].values
                mask = (x > 0) | (y > 0)
                if mask.sum() < 10: continue
                rho, pval = stats.spearmanr(x[mask], y[mask])
                rows.append({
                    'sample': sample_name, 'condition': pivot['condition'].iloc[0],
                    'microglia_gene': mg, 'axonal_gene': ag,
                    'group': group, 'rho': rho, 'p_value': pval,
                    'n_hexbins': int(mask.sum()),
                })
    return pd.DataFrame(rows)


def apply_fdr(df):
    _, padj, _, _ = multipletests(df['p_value'], method='fdr_bh')
    df['p_adj']       = padj
    df['significant'] = df['p_adj'] < 0.05
    return df


def save(fig, name, dpi=300):
    path = os.path.join(OUTPUT_PATH, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def fig_scatter_pairs(all_pivots):
    """Figure 17 — Scatter: microglia x axonal gene expression per hexbin"""
    print("\n── Figure 17: Scatter plots ─────────────────────────────────────")
    n_cols = len(AXONAL_GENES)
    n_rows = len(MICROGLIA_GENES)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4),
                             squeeze=False)
    fig.subplots_adjust(wspace=0.30, hspace=0.45)

    for ri, mg in enumerate(MICROGLIA_GENES):
        for ci, ag in enumerate(AXONAL_GENES):
            ax = axes[ri][ci]

            for cond, base_color in [('ALS', COLOR_ALS), ('Control', COLOR_CTRL)]:
                subset = all_pivots[all_pivots['condition'] == cond]
                if subset.empty: continue
                if mg not in subset.columns or ag not in subset.columns: continue

                x    = subset[mg].values
                y    = subset[ag].values
                mask = (x > 0) | (y > 0)
                x, y = x[mask], y[mask]
                if len(x) < 10: continue

                if len(x) > 3000:
                    idx_s = np.random.choice(len(x), 3000, replace=False)
                    xs, ys = x[idx_s], y[idx_s]
                else:
                    xs, ys = x, y

                ax.scatter(xs, ys, color=base_color, alpha=0.20, s=4,
                           linewidths=0, rasterized=True)

                slope, intercept, *_ = stats.linregress(x, y)
                xline = np.linspace(x.min(), x.max(), 100)
                ax.plot(xline, slope * xline + intercept,
                        color=base_color, lw=1.8, label=cond)

                rho, pval = stats.spearmanr(x, y)
                star  = '**' if pval < 0.01 else ('*' if pval < 0.05 else '')
                y_pos = 0.97 if cond == 'ALS' else 0.86
                ax.text(0.97, y_pos, f'ρ = {rho:.2f}{star}',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=7.5, color=base_color, fontweight='bold')

            ag_cat   = next((c for c, gs in GENE_CATEGORIES.items() if ag in gs), None)
            ag_color = CATEGORY_COLORS.get(ag_cat, 'black')
            mg_color = CATEGORY_COLORS['Microglia activation']

            ax.set_xlabel(ag, fontsize=8, fontstyle='italic',
                          fontweight='bold', color=ag_color, labelpad=3)
            ax.set_ylabel(mg, fontsize=8, fontstyle='italic',
                          fontweight='bold', color=mg_color, labelpad=3)
            ax.tick_params(labelsize=7)

    handles = [
        plt.Line2D([0], [0], color=COLOR_ALS,  lw=2, label='ALS'),
        plt.Line2D([0], [0], color=COLOR_CTRL, lw=2, label='Control'),
    ]
    fig.legend(handles=handles, fontsize=8.5, loc='upper right',
               bbox_to_anchor=(0.99, 0.99), frameon=True)
    fig.suptitle('Scatter: Microglia × Axonal gene expression per hexbin (CP10K)\n'
                 'Pooled across all samples per condition',
                 fontsize=11, fontweight='bold', y=1.01)
    save(fig, 'Figure17_spearman_scatter.png')


def fig_slope_plot(df_corr):
    """Figure 18 — Slope plot: Spearman rho Non-inflamed → Inflamed per gene pair"""
    print("\n── Figure 18: Slope plot ────────────────────────────────────────")
    pairs      = [(mg, ag) for mg in MICROGLIA_GENES for ag in AXONAL_GENES]
    conditions = [c for c in ['ALS', 'Control'] if c in df_corr['condition'].unique()]
    n_pairs    = len(pairs)
    n_cond     = len(conditions)

    fig, axes = plt.subplots(n_cond, n_pairs,
                             figsize=(n_pairs * 1.7, n_cond * 3.1),
                             sharey='row', squeeze=False)
    fig.subplots_adjust(wspace=0.06, hspace=0.55)

    for ri, cond in enumerate(conditions):
        cond_color    = COLOR_ALS if cond == 'ALS' else COLOR_CTRL
        sample_colors = ALS_SAMPLE_COLORS if cond == 'ALS' else CONTROL_SAMPLE_COLORS
        samples       = df_corr[df_corr['condition'] == cond]['sample'].unique()

        for ci, (mg, ag) in enumerate(pairs):
            ax = axes[ri][ci]

            sub_in  = df_corr[(df_corr['condition'] == cond) &
                               (df_corr['group'] == 'Inflamed') &
                               (df_corr['microglia_gene'] == mg) &
                               (df_corr['axonal_gene'] == ag)]
            sub_nin = df_corr[(df_corr['condition'] == cond) &
                               (df_corr['group'] == 'Non-inflamed') &
                               (df_corr['microglia_gene'] == mg) &
                               (df_corr['axonal_gene'] == ag)]

            for si, sample in enumerate(samples):
                r_in  = sub_in[sub_in['sample']   == sample]['rho'].values
                r_nin = sub_nin[sub_nin['sample'] == sample]['rho'].values
                if len(r_in) == 0 or len(r_nin) == 0: continue
                    
                col = cond_color
                for short_name, hex_color in sample_colors.items():
                    if short_name in str(sample):
                        col = hex_color
                        break
                        
                ax.plot([0, 1], [r_nin[0], r_in[0]],
                        color=col, lw=1.0, alpha=0.65, zorder=2)
                ax.scatter([0, 1], [r_nin[0], r_in[0]],
                           color=col, s=28, zorder=3,
                           edgecolors='white', linewidths=0.5)

            m_nin = sub_nin['rho'].mean()
            m_in  = sub_in['rho'].mean()
            if not (np.isnan(m_nin) or np.isnan(m_in)):
                ax.plot([0, 1], [m_nin, m_in],
                        color='black', lw=2.0, ls='--', zorder=4)
                ax.scatter([0, 1], [m_nin, m_in],
                           color='black', s=45, zorder=5, marker='D',
                           edgecolors='white', linewidths=0.6)

            ax.axhline(0, color='#bbbbbb', lw=0.7, ls=':', zorder=1)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Non-\ninflamed', 'Inflamed'], fontsize=7.5)
            ax.set_xlim(-0.35, 1.35)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=7)

            if ri == 0:
                mg_color = CATEGORY_COLORS['Microglia activation']
                ag_cat   = next((c for c, gs in GENE_CATEGORIES.items() if ag in gs), None)
                ag_color = CATEGORY_COLORS.get(ag_cat, 'black')
                ax.text(0.5, 1.12, mg, transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=8,
                        fontstyle='italic', fontweight='bold', color=mg_color)
                ax.text(0.5, 1.04, f'× {ag}', transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=7.5,
                        fontstyle='italic', color=ag_color)

            if ci == 0:
                ax.set_ylabel(f'Spearman ρ\n({cond})', fontsize=8,
                              color=cond_color, fontweight='bold')

    fig.suptitle('Spearman ρ: Non-inflamed → Inflamed LCT per gene pair\n'
                 '(per sample, -- = mean across samples)',
                 fontsize=9.5, fontweight='bold', y=1.01)
    save(fig, 'Figure18_slope_plot.png')


def fig_correlation_heatmap(df_corr):
    """Figure 19 — Spearman correlation heatmap (All / Inflamed / Non-inflamed)"""
    print("\n── Figure 19: Spearman heatmap ──────────────────────────────────")
    conditions = [c for c in ['ALS', 'Control'] if c in df_corr['condition'].unique()]
    groups     = ['All', 'Inflamed', 'Non-inflamed']
    n_cond     = len(conditions)
    n_grp      = len(groups)

    fig, axes = plt.subplots(n_cond, n_grp,
                             figsize=(n_grp * 3.2, n_cond * 2.6),
                             squeeze=False)
    fig.subplots_adjust(wspace=0.25, hspace=0.50)

    vmax = min(
        df_corr.groupby(['condition', 'group', 'microglia_gene', 'axonal_gene'])
        ['rho'].mean().abs().max() * 1.05, 0.90
    )

    cmap_custom = LinearSegmentedColormap.from_list(
        'als_custom',
        ['#588157', '#F5F0F0', '#C2527A'],
        N=256
    )

    for ri, cond in enumerate(conditions):
        cond_data = df_corr[df_corr['condition'] == cond]
        n_samp    = cond_data['sample'].nunique()
        sig_thr   = max(1, round(n_samp * 0.6))

        for ci, grp in enumerate(groups):
            ax  = axes[ri][ci]
            sub = cond_data[cond_data['group'] == grp]
            if sub.empty:
                ax.axis('off'); continue

            agg = sub.groupby(['microglia_gene', 'axonal_gene']).agg(
                mean_rho=('rho', 'mean'), n_sig=('significant', 'sum')
            ).reset_index()

            mat_rho = agg.pivot(index='microglia_gene', columns='axonal_gene',
                                values='mean_rho').reindex(
                index=MICROGLIA_GENES, columns=AXONAL_GENES)
            mat_sig = agg.pivot(index='microglia_gene', columns='axonal_gene',
                                values='n_sig').reindex(
                index=MICROGLIA_GENES, columns=AXONAL_GENES)

            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            ax.imshow(mat_rho.values, cmap=cmap_custom, norm=norm, aspect='auto')

            for i, mg in enumerate(MICROGLIA_GENES):
                for j, ag in enumerate(AXONAL_GENES):
                    rho = mat_rho.loc[mg, ag]
                    sig = mat_sig.loc[mg, ag]
                    if pd.isna(rho): continue
                    star     = '*' if (not pd.isna(sig) and sig >= sig_thr) else ''
                    bright   = abs(rho) / vmax
                    txtcolor = 'white' if bright > 0.45 else 'black'
                    ax.text(j, i, f'{rho:.2f}{star}',
                            ha='center', va='center', fontsize=8.5,
                            color=txtcolor,
                            fontweight='bold' if star else 'normal')

            ax.set_xticks(range(len(AXONAL_GENES)))
            ax.set_xticklabels(AXONAL_GENES, fontsize=8,
                               fontstyle='italic', fontweight='bold')
            for j, gene in enumerate(AXONAL_GENES):
                cat   = next((c for c, gs in GENE_CATEGORIES.items() if gene in gs), None)
                ax.get_xticklabels()[j].set_color(CATEGORY_COLORS.get(cat, 'black'))

            ax.set_yticks(range(len(MICROGLIA_GENES)))
            if ci == 0:
                ax.set_yticklabels(MICROGLIA_GENES, fontsize=8.5,
                                   fontstyle='italic', fontweight='bold',
                                   color=CATEGORY_COLORS['Microglia activation'])
            else:
                ax.set_yticklabels([])
            ax.tick_params(length=0)

            if ri == 0:
                ax.set_title(grp, fontsize=9, fontweight='bold', pad=6)
            if ci == 0:
                col = COLOR_ALS if cond == 'ALS' else COLOR_CTRL
                ax.set_ylabel(cond, fontsize=9.5, fontweight='bold',
                              color=col, labelpad=6)

    cbar_ax = fig.add_axes([0.96, 0.15, 0.012, 0.70])
    sm = plt.cm.ScalarMappable(cmap=cmap_custom, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Mean Spearman ρ', fontsize=8, labelpad=4)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle('Spearman correlations: Microglia × Axonal genes\n'
                 '(* = significant in ≥60% of samples, FDR < 0.05)',
                 fontsize=10, fontweight='bold', y=1.01)
    save(fig, 'Figure19_spearman_heatmap.png')


def fig_fisher_ztest(df_corr):
    """Figure 20 — Fisher z-test: ALS vs Control Spearman rho"""
    print("\n── Figure 20: Fisher z-test ─────────────────────────────────────")

    def fisher_z(r):
        r = np.clip(r, -0.9999, 0.9999)
        return 0.5 * np.log((1+r)/(1-r))

    def fisher_z_test(r1, n1, r2, n2):
        z1, z2 = fisher_z(r1), fisher_z(r2)
        se = np.sqrt(1/(n1-3) + 1/(n2-3))
        z  = (z1-z2)/se
        p  = 2*(1-stats.norm.cdf(abs(z)))
        return z, p

    pairs    = [(mg, ag) for mg in MICROGLIA_GENES for ag in AXONAL_GENES]
    df_g     = df_corr[df_corr['group'] == 'All']
    fdf_rows = []

    for mg, ag in pairs:
        for cond, gdf in df_g.groupby('condition'):
            sub = gdf[(gdf['microglia_gene'] == mg) & (gdf['axonal_gene'] == ag)]
            if len(sub) == 0: continue
            fdf_rows.append({
                'condition': cond, 'microglia': mg, 'axonal': ag,
                'pair':     f"{mg} × {ag}",
                'rho_mean': sub['rho'].mean(),
                'rho_se':   sub['rho'].std() / np.sqrt(len(sub)),
                'n_mean':   sub['n_hexbins'].mean()
            })

    fdf      = pd.DataFrame(fdf_rows)
    test_rows = []

    for mg, ag in pairs:
        als = fdf[(fdf['condition'] == 'ALS')     & (fdf['microglia'] == mg) & (fdf['axonal'] == ag)]
        ctl = fdf[(fdf['condition'] == 'Control') & (fdf['microglia'] == mg) & (fdf['axonal'] == ag)]
        if len(als) == 0 or len(ctl) == 0: continue
        r1, n1 = als['rho_mean'].values[0], int(als['n_mean'].values[0])
        r2, n2 = ctl['rho_mean'].values[0], int(ctl['n_mean'].values[0])
        z, p   = fisher_z_test(r1, n1, r2, n2)
        cat    = 'Metabolic stress' if ag in METABOLIC else 'Myelin/ECM'
        sig    = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        test_rows.append({
            'pair': f"{mg} × {ag}", 'microglia': mg, 'axonal': ag,
            'category': cat, 'rho_ALS': r1, 'rho_CTR': r2,
            'z_stat': z, 'p_val': p, 'sig': sig
        })

    tdf = pd.DataFrame(test_rows)
    cat_color_s = {'Metabolic stress': COLOR_LIPID, 'Myelin/ECM': COLOR_AXONAL}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Fisher's z-test: ALS vs Control Spearman ρ\n(All LCT hexbins)",
                 fontsize=13, fontweight='bold')

    for ax_i, mic in enumerate(['CHIT1', 'LYZ']):
        ax     = axes[ax_i]
        sub_t  = tdf[tdf['microglia'] == mic]
        sub_f  = fdf[fdf['microglia'] == mic]
        x      = np.arange(len(sub_t))
        w      = 0.32

        av = sub_f[sub_f['condition'] == 'ALS']['rho_mean'].values
        cv = sub_f[sub_f['condition'] == 'Control']['rho_mean'].values
        ae = sub_f[sub_f['condition'] == 'ALS']['rho_se'].values
        ce = sub_f[sub_f['condition'] == 'Control']['rho_se'].values

        ax.bar(x-w/2, av, w, color=COLOR_ALS,  alpha=0.85, label='ALS',     zorder=3)
        ax.bar(x+w/2, cv, w, color=COLOR_CTRL, alpha=0.85, label='Control', zorder=3)
        ax.errorbar(x-w/2, av, yerr=ae, fmt='none', color='#7B241C', lw=1.5, capsize=4)
        ax.errorbar(x+w/2, cv, yerr=ce, fmt='none', color='#1A5276', lw=1.5, capsize=4)

        for xi, (_, row) in enumerate(sub_t.iterrows()):
            if row['sig'] != 'ns':
                y_max = max(av[xi], cv[xi]) + 0.05
                ax.plot([xi-w/2, xi-w/2, xi+w/2, xi+w/2],
                        [y_max+0.01, y_max+0.04, y_max+0.04, y_max+0.01],
                        color='black', lw=1)
                ax.text(xi, y_max+0.055, row['sig'], ha='center', va='bottom',
                        fontsize=11, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(sub_t['axonal'].values, fontsize=10)
        ax.set_ylabel('Mean Spearman ρ ± SE', fontsize=10)
        ax.set_title(f'Microglia: {mic}', fontsize=11, fontweight='bold')
        ax.set_ylim(-1.05, 0.15)
        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        for xi, ag in enumerate(sub_t['axonal'].values):
            cat = 'Metabolic stress' if ag in METABOLIC else 'Myelin/ECM'
            ax.text(xi, -1.0, cat, ha='center', va='bottom', fontsize=8,
                    color=cat_color_s[cat], style='italic')

    save(fig, 'Figure20_fisher_ztest.png')

    print("\n  Fisher z-test results:")
    print(tdf[['pair', 'rho_ALS', 'rho_CTR', 'z_stat', 'p_val', 'sig']].to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    csv_path = os.path.join(OUTPUT_PATH, 'Spearman_correlations_raw.csv')

    # ── Step 1: Load or compute correlations ─────────────────────────────────
    if os.path.exists(csv_path):
        print(f"  Loading existing Spearman CSV: {csv_path}")
        combined     = apply_fdr(pd.read_csv(csv_path))
        all_pivots   = []
        for sample_name, fname in ALS_SAMPLES + CONTROL_SAMPLES:
            filepath = os.path.join(REGION_PATH, fname)
            if not os.path.exists(filepath): continue
            cond  = 'ALS' if 'ALS' in sample_name else 'Control'
            pivot = load_and_normalize(filepath, cond)
            pivot['sample'] = sample_name
            all_pivots.append(pivot)

        if len(all_pivots) == 0:
            combined_piv = pd.DataFrame()
        else:
            combined_piv = pd.concat(all_pivots, ignore_index=True)
    else:
        print("\n── Computing Spearman correlations ──────────────────────────────")
        all_corr   = []
        all_pivots = []

        for sample_name, fname in ALS_SAMPLES:
            filepath = os.path.join(REGION_PATH, fname)
            if not os.path.exists(filepath):
                print(f"  WARNING: {filepath} not found"); continue
            print(f"  Processing (ALS): {sample_name}")
            pivot = load_and_normalize(filepath, 'ALS')
            pivot['sample'] = sample_name
            all_pivots.append(pivot)
            all_corr.append(compute_correlations(pivot, sample_name))

        for sample_name, fname in CONTROL_SAMPLES:
            filepath = os.path.join(REGION_PATH, fname)
            if not os.path.exists(filepath):
                print(f"  WARNING: {filepath} not found"); continue
            print(f"  Processing (Control): {sample_name}")
            pivot = load_and_normalize(filepath, 'Control')
            pivot['sample'] = sample_name
            all_pivots.append(pivot)
            all_corr.append(compute_correlations(pivot, sample_name))

        combined     = apply_fdr(pd.concat(all_corr, ignore_index=True))
        combined_piv = pd.concat(all_pivots, ignore_index=True)
        combined.to_csv(csv_path, index=False)
        print(f"  Spearman CSV saved: {csv_path}")

    # ── Step 2: Generate figures ──────────────────────────────────────────────
    print("\n── Generating figures ───────────────────────────────────────────")
    fig_scatter_pairs(combined_piv)
    fig_slope_plot(combined)
    fig_correlation_heatmap(combined)
    fig_fisher_ztest(combined)

    print("\n" + "=" * 65)
    print(f"  Done! All figures saved to:")
    print(f"  {OUTPUT_PATH}")
    print("=" * 65)


if __name__ == '__main__':
    print("=" * 65)
    print("  Script 07 — Spearman Correlations")
    print("=" * 65)
    main()
