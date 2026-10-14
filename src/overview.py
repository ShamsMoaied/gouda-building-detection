"""Overview map of the whole study area with the four tiles labelled."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

labels = gpd.read_file("data/raw/labels.gpkg")
fig, ax = plt.subplots(figsize=(12, 8.5))
for tif in sorted(Path("data/raw/imagery").glob("luchtfoto_2025_*.tif")):
    with rasterio.open(tif) as src:
        show(src, ax=ax)
        b = src.bounds
        ax.plot([b.left, b.right, b.right, b.left, b.left], [b.bottom, b.bottom, b.top, b.top, b.bottom],
                color="white", linewidth=1.5)
        count = len(labels.cx[b.left:b.right, b.bottom:b.top])
        role = "TEST" if tif.stem.endswith("107500_446500") else "train"
        ax.text(b.left + 15, b.top - 45, f"{tif.stem.split('_', 2)[2]} ({role})\n{count} buildings",
                color="white", fontsize=10, weight="bold",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"))
labels.boundary.plot(ax=ax, color="yellow", linewidth=0.3)
ax.set_xlim(107000, 108500)
ax.set_ylim(446500, 447500)
ax.set_title("Study area, Gouda: 2025 aerial photo with BAG footprints")
Path("docs").mkdir(exist_ok=True)
plt.savefig("docs/study_area.png", dpi=100, bbox_inches="tight")
print("saved docs/study_area.png")
