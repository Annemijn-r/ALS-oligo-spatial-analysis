# Spatial Analysis of Oligodendrocyte Markers in ALS

This repository contains the complete computational workflow and Python pipeline used for the spatial data processing, differential gene expression profiling, and pathway-level analysis of white matter regions in ALS tissue sections.

## Pipeline Structure
The analysis workflow is executed across 8 individual Jupyter Notebooks hosted in the root directory. For proper logical progression, they align with the following functional layers:

### 1. Spatial Selection & Region Extraction (Napari)
* `Hexpaint_LCT.ipynb`: Spatial coordinate processing and selection of the Lateral Corticospinal Tract (LCT) region.
* `Hexpaint_dorsal.ipynb`: Spatial coordinate processing and selection of the dorsal white matter.
* `Hexpaint_ventral.ipynb`: Spatial coordinate processing and selection of the ventral white matter.

### 2. Differential Expression Analysis (PyDESeq2)
* `DEG_LCT_analysis.ipynb`: Execution of PyDESeq2 differential expression profiling for the LCT region.
* `Inflamed_vs_non-inflamed_spatial.ipynb`: Regional analysis comparing highly inflamed and non-inflamed tissue configurations.

### 3. Downstream Profiling, Statistics & Visualizations
* `Oligodendrocytes.ipynb`: Comprehensive categorical oligodendrocyte marker analysis, including the generation of Figure 21 (Bar plot of mean $\log_2\text{FC}$ values) and Figure 22 (Expression heatmap).
* `Regional_comparison.ipynb`: Parametric group comparisons and statistical regional variations across different white matter structures.
* `Spearman_correlations.ipynb`: Non-parametric correlation analyses across spatial layout configurations.

## System Requirements & Computational Environment
All analyses were executed using Python (version 3.13.9). The exact software environment and libraries required to fully reproduce this pipeline are detailed below:

* **Interactive Segmentation:** Napari (v0.7.0)
* **Differential Expression:** PyDESeq2 (v0.5.4)
* **Statistical Operations:** SciPy (v1.16.3), statsmodels (v0.14.6)
* **Data & Matrix Manipulation:** pandas (v2.3.3), NumPy (v2.3.5)
* **Data Visualization:** matplotlib (v3.10.6)

## Usage
1. Clone this repository or download the notebooks directly.
2. Ensure your local Python environment matches the specific package versions listed above.
3. Open and run the notebooks via Jupyter based on your phase of analysis (Spatial Selection, DEG, or Downstream Graphics).
