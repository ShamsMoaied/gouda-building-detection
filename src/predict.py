"""
Step 5a: run the model over a full aerial tile and save the detected buildings as polygons.

The tile is processed in 256 px chips (same size as training). Every predicted mask is
converted to a polygon in RD New coordinates using the chip's own geotransform, so the
output lines up with any other Dutch dataset in QGIS or ArcGIS.

Usage:
    python src/predict.py data/raw/imagery/luchtfoto_2025_107500_446500.tif
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio import features
from rasterio.windows import Window
from shapely.geometry import shape
from shapely.strtree import STRtree

from model import build_model

CHIP = 256


def positions(size):
    pos = list(range(0, size - CHIP + 1, CHIP))
    if pos[-1] != size - CHIP:
        pos.append(size - CHIP)
    return pos


def mask_to_polygon(mask, transform):
    polys = [shape(geom) for geom, value in features.shapes(mask.astype(np.uint8), mask=mask,
                                                            transform=transform) if value == 1]
    if not polys:
        return None
    # a mask can break into a few pieces; keep the biggest, and smooth the pixel staircase a little
    return max(polys, key=lambda p: p.area).simplify(0.3)


def remove_duplicates(gdf, overlap=0.5):
    """The last chip row/column overlaps its neighbour, so the same building can be found twice.
    Keep the higher-scoring one when two polygons overlap by more than half of the smaller."""
    if gdf.empty:
        return gdf
    gdf = gdf.sort_values("score", ascending=False).reset_index(drop=True)
    geoms = list(gdf.geometry)
    tree = STRtree(geoms)
    kept = set()
    for i, geom in enumerate(geoms):          # highest score first
        neighbours = [j for j in tree.query(geom) if j in kept]
        if any(geom.intersection(geoms[j]).area > overlap * min(geom.area, geoms[j].area)
               for j in neighbours):
            continue
        kept.add(i)
    return gdf.loc[sorted(kept)].reset_index(drop=True)


@torch.no_grad()
def predict_tile(model, tif_path, score_thr=0.5):
    rows = []
    with rasterio.open(tif_path) as src:
        crs = src.crs
        for r in positions(src.height):
            for c in positions(src.width):
                win = Window(c, r, CHIP, CHIP)
                img = torch.from_numpy(src.read(window=win).astype(np.float32) / 255.0)
                out = model([img])[0]
                transform = src.window_transform(win)
                for m, s in zip(out["masks"], out["scores"]):
                    if s < score_thr:
                        continue
                    poly = mask_to_polygon((m[0] > 0.5).numpy(), transform)
                    if poly is not None and poly.area > 2:      # ignore anything under 2 m2
                        rows.append({"score": round(float(s), 3), "geometry": poly})
    if not rows:
        return gpd.GeoDataFrame({"score": [], "area_m2": []}, geometry=[], crs=crs)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    gdf = remove_duplicates(gdf)
    gdf["area_m2"] = gdf.geometry.area.round(1)
    return gdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tile")
    parser.add_argument("--weights", default="models/maskrcnn_gouda_best.pth")
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--out", default="data/predictions")
    args = parser.parse_args()

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    gdf = predict_tile(model, args.tile, args.score)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) / f"{Path(args.tile).stem}_buildings.gpkg"
    gdf.to_file(out_path, driver="GPKG")
    print(f"{len(gdf)} buildings -> {out_path}")


if __name__ == "__main__":
    main()
