"""
Quick visual check: draw the BAG footprints on top of one aerial tile.
If the outlines don't sit on the roofs, something is wrong with the coordinates.
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

tile = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path("data/raw/imagery").glob("*.tif"))[0]
labels = gpd.read_file("data/raw/labels.gpkg")

with rasterio.open(tile) as src:
    b = src.bounds
    # zoom in on a 150 m square in the middle of the tile so individual buildings are visible
    cx, cy = (b.left + b.right) / 2, (b.bottom + b.top) / 2
    fig, ax = plt.subplots(figsize=(8, 8))
    show(src, ax=ax)
    labels.boundary.plot(ax=ax, color="yellow", linewidth=1)
    ax.set_xlim(cx - 75, cx + 75)
    ax.set_ylim(cy - 75, cy + 75)
    ax.set_title(f"{tile.name}: BAG footprints on 2025 aerial photo")

Path("docs").mkdir(exist_ok=True)
plt.savefig("docs/alignment_check.png", dpi=110, bbox_inches="tight")
print("saved docs/alignment_check.png")
