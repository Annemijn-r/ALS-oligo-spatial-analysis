# Spatial Analysis of Oligodendrocyte Markers and Regional White Matter Pathology in ALS

This repository contains the complete computational workflow and Python pipeline used for spatial data processing, differential gene expression profiling, micro-regional microenvironment modeling, and pathway-level analysis across distinct white matter regions in ALS spinal cord tissue sections.

---

## 📊 Pipeline Structure & Figures Mapping

The analysis workflow is executed sequentially across 8 individual Jupyter Notebooks hosted in the root directory. For proper logical progression and full analytical reproducibility, they align with the following functional layers:

### 1. Spatial Selection & Region Extraction (Interactive Segmentations)
* **[`Hexpaint_LCT.py`](./Hexpaint_LCT.py)**
    * Processes raw spatial spot coordinates and utilizes an interactive Napari workflow to manually segment, isolate, and extract transcript counts specifically corresponding to the **Lateral Corticospinal Tract (LCT)** white matter region.
* **`Hexpaint_dorsal.py`**
    * Isolates spatial coordinates and anatomical expression arrays mapping exclusively to the **dorsal white matter** columns.
* **`Hexpaint_ventral.py`**
    * Isolates spatial coordinates and coordinate arrays mapping exclusively to the **ventral white matter** columns.

### 2. Differential Expression Analysis (PyDESeq2) & Regional Variations
* **`DEG_LCT_analysis.py`** *(Script 4)*
    * Executes high-throughput pseudo-bulk differential gene expression (DEG) profiling using `PyDESeq2` comparing ALS vs. Control samples within the primary LCT tract.
    * **📊 Figures Generated:**
        * *Figure 1:* Volcano plot highlighting globally significant DEGs.
        * *Figure 2:* MA plot visualizing log2 Fold Changes against mean expression intensities.
        * *Figure 3:* Formatted summary table of top significantly dysregulated DEGs.
        * *Figure 4:* Expression heatmap of the top 15 most variable DEGs across conditions.
        * *Figure 5:* Gene-specific boxplots tracking normalized expression shifts in counts per million (CPM).
        * *Figure 6:* Multi-panel bar chart contrasting localized inflammation metrics against axonal and metabolic integrity markers.
* **`Regional_comparison.py`** *(Script 5)*
    * Performs macro-level parametric and non-parametric group comparisons across the different white matter tracts to establish the unique spatial vulnerability profile of the LCT.
    * **📊 Figures Generated:**
        * *Figure 7:* Coordinate scatter plot: LCT vs. Dorsal tracking individual genes grouped by pathway.
        * *Figure 8:* Comparative path-level bar plot highlighting differential effect sizes between LCT and Dorsal columns.
        * *Figure 9:* Coordinate scatter plot: LCT vs. Ventral tracking individual genes grouped by pathway.
        * *Figure 10:* Comparative path-level bar plot highlighting differential effect sizes between LCT and Ventral columns.
        * *Figure 11:* Combined multi-region bar plot summarizing the average $\log_2\text{FC}$ across all target pathways per region.

### 3. Local Microenvironment & Structural Dynamics
* **`Inflamed_vs_non-inflamed_spatial.py`** *(Script 6)*
    * Dissects the micro-spatial tissue architecture by separating spots into distinct inflamed and non-inflamed configurations based on regional cluster dominance to observe localized degradation.
    * **📊 Figures Generated:**
        * *Figure 12:* Cluster composition analysis demonstrating the structural dominance of cellular Cluster 6 in ALS tissues.
        * *Figure 13:* Sample-level tissue composition chart mapping the cumulative proportion of active inflammatory clusters (Clusters 6 + 7).
        * *Figure 14:* Regional category summary comparing the percentage of hexbins classified under specific inflammatory categories.
        * *Figure 15:* Non-parametric Mann-Whitney $U$ test distributions assessing metabolic stress parameters and microglia score variances between tracking clusters (Cluster 4 vs. Cluster 6).
        * *Figure 16:* Integrated dual-panel graphics presenting an expression heatmap juxtaposed with a forest plot evaluating gene patterns across inflamed vs. non-inflamed zones.
* **`Spearman_correlations.py`** *(Script 7)*
    * Runs hexbin-level non-parametric Spearman correlation matrices to track the direct co-expression patterns between active microglia drivers (`CHIT1`, `LYZ`) and localized axonal stress targets.
    * **📊 Figures Generated:**
        * *Figure 17:* Matrix of scatter plots tracking Microglia × Axonal gene expression across pooled conditions.
        * *Figure 18:* Slope plot tracking sample-specific changes in Spearman $\rho$ values during the functional shift from Non-inflamed $\rightarrow$ Inflamed configurations.
        * *Figure 19:* Consolidated Spearman correlation heatmap broken down by local tissue groupings (All / Inflamed / Non-inflamed).
        * *Figure 20:* Dual-panel bar chart displaying Fisher's $z$-test statistics comparing ALS vs. Control Spearman $\rho$ variances to determine correlation significance.
* **`Oligodendrocytes.py`** *(Script 8)*
    * Focuses exclusively on the oligodendrocyte lineage, running targeted differential testing across mature oligodendrocyte structural components, oligodendrocyte precursor cell (OPC) markers, and local cholesterol/lipid synthesis pathways.
    * **📊 Figures Generated:**
        * *Figure 21:* Combined bar plot showing the average $\log_2\text{FC}$ shifts per designated oligodendrocyte functional category across all three white matter territories (LCT, Dorsal, Ventral).
        * *Figure 22:* High-resolution individual gene expression heatmap tracking $\log_2\text{FC}$ values across all genes in the customized oligodendrocyte panel.

---

## 🛠️ System Requirements & Computational Environment

All analyses were executed using Python (version 3.13.9). The exact software environment and specific libraries required to fully reproduce this pipeline are detailed below:

* **Interactive Segmentation:** `napari` (v0.7.0)
* **Differential Expression Testing:** `pydeseq2` (v0.5.4)
* **Statistical Operations:** `scipy` (v1.16.3), `statsmodels` (v0.14.6)
* **Data & Matrix Manipulation:** `pandas` (v2.3.3), `numpy` (v2.3.5)
* **Data Visualization:** `matplotlib` (v3.10.6)

---

## 🚀 Usage

1. **Clone this repository** to download the analysis suite directly:
   ```bash
   git clone [https://github.com/Annemijn-r/ALS-oligo-spatial-analysis.git](https://github.com/Annemijn-r/ALS-oligo-spatial-analysis.git)
   cd ALS-oligo-spatial-analysis
