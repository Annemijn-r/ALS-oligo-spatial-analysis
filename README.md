# Spatial Analysis of Oligodendrocyte Markers in ALS

This repository contains the complete computational workflow and Python pipeline used for the spatial data processing, differential gene expression profiling, and pathway-level analysis of white matter regions in ALS tissue sections.

## Repository Structure
* `scripts/`: Contains the 8 sequential execution scripts.
  * `01_napari_coordinate_extraction.py` to `03_...`: Handles interactive spatial selections and manual region segmentations in Napari.
  * `04_...` to `07_...`: Executes PyDESeq2 differential expression profiling.
  * `08_oligo_marker_analysis.py`: Computes regional category-level metrics and generates Figure 21 (Bar plot) and Figure 22 (Heatmap).

## System Requirements & Computational Environment
All analyses were executed using Python (version 3.13.9). The exact software environment and libraries required to fully reproduce this pipeline are detailed below:

* **Interactive Segmentation:** Napari (v0.7.0)
* **Differential Expression:** PyDESeq2 (v0.5.4)
* **Statistical Operations:** SciPy (v1.16.3), statsmodels (v0.14.6)
* **Data & Matrix Manipulation:** pandas (v2.3.3), NumPy (v2.3.5)
* **Data Visualization:** matplotlib (v3.10.6)

## Usage
1. Clone this repository or download the scripts.
2. Ensure your local environment matches the specific package versions listed above.
3. Execute the scripts sequentially from `01` to `08`.
