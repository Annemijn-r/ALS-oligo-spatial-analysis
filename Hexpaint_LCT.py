"""
Script 1 - Napari Region Selection: Lateral Corticospinal Tract (LCT)
-------------------------------------------------------------------------
Purpose:
  Manually select the LCT region per sample using Napari.
  Run this script once per sample by changing SAMPLE and CONDITION.

Workflow:
  1. Set SAMPLE and CONDITION at the top
  2. Run the script — Napari opens with the hexbin tissue map
  3. Draw a polygon around the LCT region using the Shapes layer
  4. Close Napari — selection is saved automatically
  5. A control plot is saved to verify the selection

Dependencies: napari, pandas, numpy, matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import napari
from matplotlib.path import Path

# ════════════════════════════════════════════════════════════════════════════════
# CHANGE THESE TWO LINES FOR EACH SAMPLE
# ════════════════════════════════════════════════════════════════════════════════
SAMPLE    = 'S09-055'    # Sample ID
CONDITION = 'ALS'        # 'ALS' or 'Control'
# ════════════════════════════════════════════════════════════════════════════════

# ── Paths (Universal & GitHub-Proof) ──────────────────────────────────────────
# Bepaal de hoofdmap waar dit notebook zich bevindt
BASE_DIR = os.getcwd()

# Relatieve paden die werken op élke computer
INPUT_PATH = os.path.join(BASE_DIR, "Hexbin_data")

if CONDITION == 'ALS':
    OUTPUT_PATH = os.path.join(BASE_DIR, "output", "Region_clusters", "Disease_hexpaint", "LCT")
    FILE_PREFIX = f"ALS_{SAMPLE}"
else:
    OUTPUT_PATH = os.path.join(BASE_DIR, "output", "Region_clusters", "Healthy_hexpaint", "LCT")
    FILE_PREFIX = f"healthy_{SAMPLE}"

# Maak de mappen automatisch aan als ze nog nu nog niet bestaan
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── Rotation angle per sample ─────────────────────────────────────────────────
ROTATION_ANGLES = {
    'S09-055': 15, 'S13-047': 10, 'S13-054': 175,
    'S18-070': -100, 'S19-061': 75,
    'S12-186': 170, 'S13-133': 70, 'S15-025': 190,
    'S15-051': 10, 'S17-062': -60,
}
ANGLE = ROTATION_ANGLES.get(SAMPLE, 75)

# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load hexbin data
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"  LCT Selection — {SAMPLE} ({CONDITION})")
print(f"{'='*55}")

raw_file = os.path.join(INPUT_PATH, f'{SAMPLE}_hexbin_50.csv.gz')
if not os.path.exists(raw_file):
    raise FileNotFoundError(f"Raw data not found: {raw_file}\nZorg ervoor dat de map 'Hexbin_data' in dezelfde map staat als dit notebook.")

df = pd.read_csv(raw_file, compression='gzip')
hexbins = df.drop_duplicates(subset='hexbin_id')[
    ['hexbin_id', 'hexbin_i', 'hexbin_j', 'L1_region_cluster']
].copy()
print(f"  Loaded {SAMPLE}: {len(hexbins):,} unique hexbins")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — Rotate and open Napari
# ════════════════════════════════════════════════════════════════════════════════
angle_rad = np.radians(ANGLE)
x = hexbins['hexbin_i'].values
y = hexbins['hexbin_j'].values

x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)

hexbins_display = hexbins.copy()
hexbins_display['hexbin_i_rot'] = x_rot
hexbins_display['hexbin_j_rot'] = y_rot

cmap     = plt.get_cmap('tab20')
clusters = hexbins_display['L1_region_cluster'].astype(int).values
colors   = np.array([cmap(c / 20) for c in clusters])

print(f"\n  Opening Napari — draw a polygon around the LCT region")
print(f"  then close the window to save the selection.")

viewer = napari.Viewer(title=f"LCT Selection — {SAMPLE} ({CONDITION})")
viewer.add_points(
    hexbins_display[['hexbin_j_rot', 'hexbin_i_rot']].values,
    name=f'hexbins_{SAMPLE}',
    size=2,
    face_color=colors,
)
napari.run()


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — Extract selected hexbins from drawn polygon
# ════════════════════════════════════════════════════════════════════════════════
shapes_layer = None
for layer in viewer.layers:
    if layer.__class__.__name__ == 'Shapes':
        shapes_layer = layer
        break

if shapes_layer is None or len(shapes_layer.data) == 0:
    print("\n  WARNING: No shapes drawn — nothing saved.")
else:
    angle_back_rad = np.radians(-ANGLE)
    selected_hexbins = []

    for shape in shapes_layer.data:
        y_rot_shape = shape[:, 0]
        x_rot_shape = shape[:, 1]

        x_orig = x_rot_shape * np.cos(angle_back_rad) - y_rot_shape * np.sin(angle_back_rad)
        y_orig = x_rot_shape * np.sin(angle_back_rad) + y_rot_shape * np.cos(angle_back_rad)

        rotated_shape = np.column_stack([x_orig, y_orig])
        path  = Path(rotated_shape)
        points = hexbins[['hexbin_i', 'hexbin_j']].values
        mask  = path.contains_points(points)
        selected_hexbins.append(hexbins[mask])

    selection = pd.concat(selected_hexbins).drop_duplicates(subset='hexbin_id')
    print(f"\n  Selected hexbins: {len(selection):,}")


    # ── STEP 4: Save full data for selected hexbins ───────────────────────────
    full_selection = df[df['hexbin_id'].isin(selection['hexbin_id'])].copy()
    output_file    = os.path.join(OUTPUT_PATH, f"{FILE_PREFIX}_selection.csv.gz")
    full_selection.to_csv(output_file, index=False, compression='gzip')
    print(f"  Saved: {output_file}")


    # ── STEP 5: Control plot ──────────────────────────────────────────────────
    hexbins_all = df.drop_duplicates(subset='hexbin_id')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'LCT Selection Control Plot — {SAMPLE} ({CONDITION})',
                 fontsize=12, fontweight='bold')

    # Full tissue with selection highlighted
    axes[0].scatter(hexbins_all['hexbin_i'], hexbins_all['hexbin_j'],
                    s=0.5, c='lightgrey', alpha=0.5, label='All hexbins')
    axes[0].scatter(selection['hexbin_i'], selection['hexbin_j'],
                    s=1, c='#6D597A', alpha=0.9, label='LCT selection')
    axes[0].set_aspect('equal')
    axes[0].set_title('LCT selection on full tissue')
    axes[0].set_xlabel('hexbin_i')
    axes[0].set_ylabel('hexbin_j')
    axes[0].legend(fontsize=8)

    # Cluster composition of selection
    cluster_counts = selection['L1_region_cluster'].value_counts().sort_index()
    axes[1].bar(cluster_counts.index.astype(str), cluster_counts.values,
                color='#6D597A', alpha=0.8)
    axes[1].set_title('Cluster composition of LCT selection')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Number of hexbins')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    control_plot = os.path.join(OUTPUT_PATH, f"control_{FILE_PREFIX}_LCT.png")
    plt.savefig(control_plot, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Control plot saved: control_{FILE_PREFIX}_LCT.png")

    print(f"\n  Done! Summary:")
    print(f"    Sample:     {SAMPLE} ({CONDITION})")
    print(f"    Region:     LCT")
    print(f"    Hexbins:    {len(selection):,}")
    print(f"    Output:     {output_file}")
