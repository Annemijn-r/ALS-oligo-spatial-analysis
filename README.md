# Spatial Analysis of Oligodendrocyte Markers in ALS

This repository contains the complete computational workflow and Python pipeline used for the spatial data processing, differential gene expression profiling, and pathway-level analysis of white matter regions in ALS tissue sections.

---

## 📊 Pipeline Structure & Figures Created

The analysis workflow is executed across 8 individual Jupyter Notebooks hosted in the root directory. To successfully replicate the results, notebooks should be evaluated or run according to their numerical/functional sequence below:

### 1. Spatial Selection & Region Extraction (Napari)
* **`Hexpaint_LCT.ipynb`**
    * **What it does:** Processes raw spatial coordinates and utilizes an interactive Napari segmentation workflow to isolate and extract data specifically corresponding to the **Lateral Corticospinal Tract (LCT)** region.
* **`Hexpaint_dorsal.ipynb`**
    * **What it does:** Isolates spatial coordinates and translatational data specifically for the **dorsal white matter** columns.
* **`Hexpaint_ventral.ipynb`**
    * **What it does:** Isolates spatial coordinates and transcript counts specifically for the **ventral white matter** columns.

### 2. Differential Expression Analysis (PyDESeq2)
* **`DEG_LCT_analysis.ipynb`**
    * **What it does:** Executes high-throughput differential gene expression profiling using `PyDESeq2` on pseudo-bulk counts to isolate ALS-specific transcriptional changes within the LCT white matter tract.
* **`Inflamed_vs_non-inflamed_spatial.ipynb`**
    * **What it does:** Runs micro-regional differential expression testing comparing specific high-inflammation spatial configurations against non-inflamed baselines.

### 3. Downstream Profiling, Statistics & Visualizations
* **`Spearman_correlations.ipynb`**
    * **What it does:** Runs cell/hexbin-level non-parametric Spearman correlation matrices assessing the relational trends between microglia activation markers (`CHIT1`, `LYZ`) and localized axonal stress targets.
    * **📊 Figures Generated:**
        * *Figure 17:* Matrix of scatter plots tracking Microglia × Axonal gene expression across pooled conditions.
        * *Figure 18:* Slope plot tracking changes in Spearman $\rho$ values from Non-inflamed to Inflamed configurations per sample.
        * *Figure 19:* Consolidated Spearman correlation heatmap split by tissue groupings (All / Inflamed / Non-inflamed).
        * *Figure 20:* Bar plots displaying Fisher's $z$-test statistics comparing ALS vs. Control Spearman $\rho$ variances.
* **`Oligodendrocytes.ipynb`**
    * **What it does:** Conducts a targeted categorical assessment of mature oligodendrocyte markers, oligodendrocyte precursor cells (OPCs), and cholesterol synthesis pathways across different white matter zones.
    * **📊 Figures Generated:**
        * *Figure 21:* Combined bar plot showing the average $\log_2 \text{Fold Change}$ ($\log_2\text{FC}$) value variations per oligodendrocyte category across all three white matter regions (LCT, Dorsal, Ventral).
        * *Figure 22:* High-resolution expression heatmap tracking individual log2FC expression values across all examined oligodendrocyte panel genes grouped by functional categories.
* **`Regional_comparison.ipynb`**
    * **What it does:** Integrates findings across the three individual regions (LCT, Dorsal, Ventral) to run parametric and non-parametric group comparisons, highlighting statistical regional variations in ALS vulnerability.
    * **📊 Figures Generated:**
        * *Figure 23:* Scatter correlation plots tracking the spatial intersection of mature oligodendrocyte degradation trends against metabolic and cholesterol synthesis pathway metrics.

---

## 🛠️ System Requirements & Computational Environment

All analyses were executed using Python (version 3.13.9). The exact software environment and libraries required to fully reproduce this pipeline are detailed below:

* **Interactive Segmentation:** Napari (v0.7.0)
* **Differential Expression:** PyDESeq2 (v0.5.4)
* **Statistical Operations:** SciPy (v1.16.3), statsmodels (v0.14.6)
* **Data & Matrix Manipulation:** pandas (v2.3.3), NumPy (v2.3.5)
* **Data Visualization:** matplotlib (v3.10.6)

---

## 🚀 Usage

1. **Clone this repository** or download the notebooks directly:
   ```bash
   git clone [https://github.com/Annemijn-r/ALS-oligo-spatial-analysis.git](https://github.com/Annemijn-r/ALS-oligo-spatial-analysis.git)
   cd ALS-oligo-spatial-analysis
