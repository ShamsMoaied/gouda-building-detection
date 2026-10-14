"""
Step 2: cut the large aerial tiles into small training chips with instance masks.

For every chip we write two GeoTIFFs with the same georeference:
  images/<name>.tif  3-band RGB
  masks/<name>.tif   uint16, 0 = background, 1..N = one value per building

The split is spatial: one whole 500 m tile is held out as the test area, so no test building
(or a piece of it) is ever seen during training.

Usage:
    python src/make_chips.py
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box

CHIP = 256            # pixels; 256 * 0.25 m = 64 m on the ground
TRAIN_STRIDE = 192    # overlapping chips for training (25% overlap)
TEST_STRIDE = 256     # no overlap for the test area, so every building is counted once
MIN_PIXELS = 40       # drop tiny slivers of buildings cut off at the chip edge (40 px = 2.5 m2)


def chip_positions(size, stride):
    pos = list(range(0, size - CHIP + 1, stride))
    if pos[-1] != size - CHIP:      # make sure the last row/column reaches the edge
        pos.append(size - CHIP)
    return pos


def make_chips(tile_path, labels, out_dir, stride):
    n_chips, n_buildings = 0, 0
    with rasterio.open(tile_path) as src:
        cols = chip_positions(src.width, stride)
        rows = chip_positions(src.height, stride)
        for r in rows:
            for c in cols:
                win = Window(c, r, CHIP, CHIP)
                transform = src.window_transform(win)
                img = src.read(window=win)

                # only the buildings that touch this chip
                chip_box = box(*window_bounds(win, src.transform))
                hits = labels[labels.intersects(chip_box)]
                shapes = [(geom, i + 1) for i, geom in enumerate(hits.geometry)]

                mask = np.zeros((CHIP, CHIP), dtype=np.uint16)
                if shapes:
                    mask = features.rasterize(shapes, out_shape=(CHIP, CHIP), transform=transform,
                                              fill=0, dtype="uint16")
                    # remove slivers and renumber the remaining buildings 1..N
                    ids, counts = np.unique(mask[mask > 0], return_counts=True)
                    keep = ids[counts >= MIN_PIXELS]
                    clean = np.zeros_like(mask)
                    for new_id, old_id in enumerate(keep, start=1):
                        clean[mask == old_id] = new_id
                    mask = clean

                name = f"{tile_path.stem}_r{r:04d}_c{c:04d}"
                profile = dict(driver="GTiff", height=CHIP, width=CHIP, crs=src.crs,
                               transform=transform, compress="deflate")
                with rasterio.open(out_dir / "images" / f"{name}.tif", "w", count=3,
                                   dtype="uint8", **profile) as dst:
                    dst.write(img)
                with rasterio.open(out_dir / "masks" / f"{name}.tif", "w", count=1,
                                   dtype="uint16", **profile) as dst:
                    dst.write(mask, 1)

                n_chips += 1
                n_buildings += int(mask.max())
    return n_chips, n_buildings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/chips")
    parser.add_argument("--test-tile", default="luchtfoto_2025_107500_446500")
    parser.add_argument("--val-tile", default=None,
                        help="optional tile held out for choosing the best epoch, e.g. luchtfoto_2025_107000_446500")
    parser.add_argument("--dissolve", action="store_true",
                        help="merge touching BAG buildings into blocks (one terraced row = one object)")
    args = parser.parse_args()

    labels = gpd.read_file(Path(args.raw) / "labels.gpkg")
    if args.dissolve:
        # A BAG "pand" is a legal unit, not a visual one: a row of ten houses is ten panden
        # under one continuous roof. Grow every polygon by 10 cm, merge what touches, shrink back.
        merged = labels.geometry.buffer(0.1).union_all()
        labels = gpd.GeoDataFrame(geometry=[merged], crs=labels.crs).explode(index_parts=False)
        labels["geometry"] = labels.geometry.buffer(-0.1)
        labels = labels[~labels.geometry.is_empty].reset_index(drop=True)
        print(f"dissolved into {len(labels)} blocks")
    tiles = sorted((Path(args.raw) / "imagery").glob("luchtfoto_2025_*.tif"))

    for split in ("train", "val", "test"):
        for sub in ("images", "masks"):
            (Path(args.out) / split / sub).mkdir(parents=True, exist_ok=True)

    for tile in tiles:
        split = {args.test_tile: "test", args.val_tile: "val"}.get(tile.stem, "train")
        stride = TRAIN_STRIDE if split == "train" else TEST_STRIDE
        n, b = make_chips(tile, labels, Path(args.out) / split, stride)
        print(f"{tile.name:40s} {split:5s}  {n:4d} chips  {b:5d} building instances")


if __name__ == "__main__":
    main()
