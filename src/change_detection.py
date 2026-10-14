"""
Step 6: find new buildings by comparing model predictions on 2020 and 2025 imagery.

A 2025 detection counts as "new" when less than 30% of it is covered by any 2020 detection.
The check uses the BAG: buildings with bouwjaar 2021-2024 in the same area are the real new
buildings (2025 is left out because the 2025 photos are flown early in the year).

Usage:
    python src/change_detection.py
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
import torch
from rasterio.plot import show
from shapely.geometry import box

from model import build_model
from predict import predict_tile

MIN_AREA = 20        # m2; ignore small new objects (cars, containers, garden sheds)
MAX_OVERLAP = 0.3


def load_or_predict(model, tif, out_dir):
    path = out_dir / f"{tif.stem}_buildings.gpkg"
    if path.exists():
        return gpd.read_file(path)
    gdf = predict_tile(model, tif)
    gdf.to_file(path, driver="GPKG")
    return gdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", default="107500_446500")
    parser.add_argument("--before", default="2020")
    parser.add_argument("--after", default="2025")
    parser.add_argument("--weights", default="models/maskrcnn_gouda_best.pth")
    args = parser.parse_args()

    out_dir = Path("data/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    tif_before = Path(f"data/raw/imagery/luchtfoto_{args.before}_{args.tile}.tif")
    tif_after = Path(f"data/raw/imagery/luchtfoto_{args.after}_{args.tile}.tif")
    before = load_or_predict(model, tif_before, out_dir)
    after = load_or_predict(model, tif_after, out_dir)
    print(f"detected: {len(before)} buildings in {args.before}, {len(after)} in {args.after}")

    before_union = before.geometry.union_all()
    covered = after.geometry.intersection(before_union).area / after.geometry.area
    new = after[(covered < MAX_OVERLAP) & (after.area_m2 >= MIN_AREA)].copy()
    new["covered_by_before"] = covered[new.index].round(2)
    new.to_file(out_dir / f"changes_{args.before}_{args.after}.gpkg", driver="GPKG")
    print(f"flagged as new: {len(new)}")

    # Ground truth from the BAG register
    with rasterio.open(tif_after) as src:
        area = box(*src.bounds)
    bag = gpd.read_file("data/raw/bag_pand.gpkg")
    bag["bouwjaar"] = bag["bouwjaar"].astype(int)
    truth = bag[bag.within(area) & bag.bouwjaar.between(int(args.before) + 1, int(args.after) - 1)
                & (bag.status == "Pand in gebruik")]

    def hit(geom, others):
        return any(geom.intersection(o).area > 0.3 * min(geom.area, o.area) for o in others)

    found = [hit(g, new.geometry) for g in truth.geometry]
    correct_alerts = [hit(g, truth.geometry) for g in new.geometry]
    result = {
        "area": args.tile,
        "years": f"{args.before} -> {args.after}",
        "bag_new_buildings": len(truth),
        "bag_new_buildings_found": int(sum(found)),
        "recall": round(sum(found) / max(len(truth), 1), 3),
        "alerts": len(new),
        "alerts_matching_a_new_bag_building": int(sum(correct_alerts)),
        "alert_precision": round(sum(correct_alerts) / max(len(new), 1), 3),
    }
    print(json.dumps(result, indent=2))
    Path("docs/change_detection.json").write_text(json.dumps(result, indent=2))

    # Figure: zoom on the area with most new buildings
    if len(truth):
        # centre on the new building with the most other new buildings within 60 m
        centroids = truth.geometry.centroid
        busiest = max(centroids, key=lambda c: (centroids.distance(c) < 60).sum())
        cx, cy = busiest.x, busiest.y
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
        for ax, tif, year in ((axes[0], tif_before, args.before), (axes[1], tif_after, args.after)):
            with rasterio.open(tif) as src:
                show(src, ax=ax)
            truth.boundary.plot(ax=ax, color="yellow", linewidth=1.2)
            if year == args.after:
                new.boundary.plot(ax=ax, color="red", linewidth=1.5)
            ax.set_xlim(cx - 90, cx + 90)
            ax.set_ylim(cy - 90, cy + 90)
            ax.set_title(f"{year}" + ("  (red = flagged as new by the model)" if year == args.after else ""))
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Yellow: buildings the BAG says were built after " + args.before)
        plt.tight_layout()
        plt.savefig("docs/change_detection.png", dpi=100)
        print("saved docs/change_detection.png")


if __name__ == "__main__":
    main()
