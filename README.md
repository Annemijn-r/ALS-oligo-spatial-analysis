# Spatial Analysis of Oligodendrocyte Markers and Regional White Matter Pathology in ALS

This repository contains the complete workflow and Python pipeline used for spatial data processing, differential gene expression profiling, micro-regional microenvironment modeling, and pathway-level analysis across three individual white matter regions in ALS spinal cord tissue sections.

---

## 📊 Pipeline Structure & Figures Mapping

The analysis workflow is executed across 8 individual Jupyter scripts hosted in the root directory. To keep the analysis logical and replicable, the scripts are organized into the following functional layers:

### 1. Spatial Selection & Region Extraction (Interactive Segmentations)
* **[`Hexpaint_LCT.py`](./Scripts/Hexpaint_LCT.py)**
    * Processes raw spatial coordinates and uses an interactive Napari workflow to manually segment, isolate, and extract transcript counts corresponding to the **Lateral Corticospinal Tract (LCT)** white matter region.
* **[`Hexpaint_dorsal.py`](./Scripts/Hexpaint_dorsal.py)**
    * Processes raw spatial coordinates and uses an interactive Napari workflow to manually segment, isolate, and extract transcript counts corresponding to the **dorsal white matter** regions.
* **[`Hexpaint_ventral.py`](./Scripts/Hexpaint_ventral.py)**
    * Processes raw spatial coordinates and uses an interactive Napari workflow to manually segment, isolate, and extract transcript counts corresponding to the **ventral white matter** regions.

### 2. Differential Expression Analysis (PyDESeq2) & Regional Variations
* **[`DEG_LCT_analysis.py`](./Scripts/DEG_LCT_analysis.py)** *(Script 4)*
    * Carries out high-throughput pseudo-bulk differential gene expression (DEG) profiling using `PyDESeq2` comparing ALS vs. Control samples within the primary LCT tract.
    * **📊 Figures Generated:**
        * *Figure 1:* Volcano plot highlighting significant DEGs.
        * *Figure 2:* MA plot visualizing $\log_2\text{FC}$ against mean expression intensities.
        * *Figure 3:* Formatted summary table of top significantly dysregulated DEGs.
        * *Figure 4:* Expression heatmap of the top 15 most variable DEGs across conditions.
        * *Figure 5:* Gene-specific boxplots showing gene expression shifts per sample in counts per million (CPM).
        * *Figure 6:* Bar chart contrasting localized inflammation metrics against axonal and metabolic integrity markers in the LCT                          region.
* **[`Regional_comparison.py`](./Scripts/Regional_comparison.py)** *(Script 5)*
    * Using both parametric and non-parametric test to compare the different white matter tracts to show the unique spatial vulnerability profile of the LCT.
    * **📊 Figures Generated:**
        * *Figure 7:* Scatter plot: LCT vs. Dorsal tracking individual genes grouped per pathway.
        * *Figure 8:* Pathway dysregulation bar plot highlightes the differential effect sizes between LCT and Dorsal columns.
        * *Figure 9:* Scatter plot: LCT vs. Ventral tracking individual genes grouped by pathway.
        * *Figure 10:* Pathway dysregulation bar plot highlightes the differential effect sizes between LCT and Ventral columns.
        * *Figure 11:* Bar plot summarizing the average $\log_2\text{FC}$ across all target pathways per region.

### 3. Local Microenvironment & Structural Dynamics
* **[`Inflamed_vs_non-inflamed_spatial.py`](./Scripts/Inflamed_vs_non-inflamed_spatial.py)** *(Script 6)*
    * Segmentation of micro-spatial tissue devided into inflamed and non-inflamed regions based on regional cluster dominance to observe localized degradation.
    * **📊 Figures Generated:**
        * *Figure 12:* Cluster composition analysis showing the structural dominance of Cluster 6 in ALS tissues.
        * *Figure 13:* Sample-level tissue composition chart mapping the proportion of inflammatory clusters (Clusters 6 + 7).
        * *Figure 14:* Regional category summary comparing the percentage of hexbins classified into distinct inflammatory categories.
        * *Figure 15:* Non-parametric Mann-Whitney $U$ test distributions of metabolic stress parameters and microglia scores variances                        compared between specific tissue clusters (Vluster 4 vs Cluster 6)
        * *Figure 16:* Dual-panel layout showing an expression heatmap and paired forest plot comparing gene regulation patterns across                        inflamed and non-inflamed sub-regions.  
* **[`Spearman_correlations.py`](./Scripts/Spearman_correlations.py)** *(Script 7)*
    * Runs hexbin-level non-parametric Spearman correlation showing co-expression patterns between microglia markers (`CHIT1`, `LYZ`) and localized axonal stress targets.
    * **📊 Figures Generated:**
        * *Figure 17:* Scatter plots tracking Microglia × Axonal gene expression across pooled conditions.
        * *Figure 18:* Slope plot tracking sample-specific changes in Spearman $\rho$ values in the functional shift from Non-inflamed                         $\rightarrow$ Inflamed configurations.
        * *Figure 19:* Spearman correlation heatmap separated by tissue groupings (All / Inflamed / Non-inflamed).
        * *Figure 20:* Bar chart displaying Fisher's $z$-test statistics comparing ALS vs. Control Spearman $\rho$ variances to                                determine correlation significance.
* **[`Oligodendrocytes.py`](./Scripts/Oligodendrocytes.py)** *(Script 8)*
    * Focuses on oligodendrocyte, running targeted differential testing across mature oligodendrocyte structural components, oligodendrocyte precursor cell (OPC) markers, and local cholesterol/lipid synthesis pathways.
    * **📊 Figures Generated:**
        * *Figure 21:* Bar plot showing the average $\log_2\text{FC}$ shifts per oligodendrocyte functional category across all three                          white matter sections (LCT, Dorsal, Ventral).
        * *Figure 22:* Individual gene expression heatmap tracking $\log_2\text{FC}$ values across all genes in the oligodendrocyte                            panel.

---

## 🛠️ System Requirements & Computational Environment

All analyses were executed using Python (version 3.13.9). The exact software environment and specific libraries required to reproduce this pipeline are mentioned below:

* **Interactive Segmentation:** `napari` (v0.7.0)
* **Differential Expression Testing:** `pydeseq2` (v0.5.4)
* **Statistical Operations:** `scipy` (v1.16.3), `statsmodels` (v0.14.6)
* **Data & Matrix Manipulation:** `pandas` (v2.3.3), `numpy` (v2.3.5)
* **Data Visualization:** `matplotlib` (v3.10.6)

---

## 🚀 Usage

1. **Clone this repository** to download the analysis directly:
   ```bash
   git clone [https://github.com/Annemijn-r/ALS-oligo-spatial-analysis.git](https://github.com/Annemijn-r/ALS-oligo-spatial-analysis.git)
   cd ALS-oligo-spatial-analysis
