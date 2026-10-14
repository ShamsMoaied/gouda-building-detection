"""
Step 1: download training data for a small area of Gouda.

- Building footprints from the BAG (Basisregistratie Adressen en Gebouwen) via the PDOK WFS
- 25 cm aerial imagery (Luchtfoto 2025) via the PDOK WMS, saved as GeoTIFF tiles

Everything is in EPSG:28992 (Amersfoort / RD New), the Dutch national grid, so coordinates are in metres.

Usage:
    python src/download_data.py
    python src/download_data.py --bbox 107000 446500 108000 447500 --year 2025
"""

import argparse
import io
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.transform import from_origin

BAG_WFS = "https://service.pdok.nl/lv/bag/wfs/v2_0"
LUCHTFOTO_WMS = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"

RESOLUTION = 0.25   # metres per pixel (the "ortho25" product)
TILE_SIZE_M = 500   # 500 m tile at 0.25 m = 2000 px, under the WMS limit of 2500 px

# BAG statuses that mean the building physically exists.
# "Bouwvergunning verleend" (permit granted) and "Bouw gestart" can be in the register
# while the photo still shows an empty plot, so those would be wrong labels.
EXISTING = {
    "Pand in gebruik",
    "Pand in gebruik (niet ingemeten)",
    "Verbouwing pand",
    "Sloopvergunning verleend",
}


def download_buildings(bbox, out_path):
    xmin, ymin, xmax, ymax = bbox
    features = []
    start = 0
    while True:
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": "bag:pand",
            "outputFormat": "application/json",
            "srsName": "EPSG:28992",
            "bbox": f"{xmin},{ymin},{xmax},{ymax},EPSG:28992",
            "count": 1000,          # the service returns max 1000 per request, so we page
            "startIndex": start,
        }
        r = requests.get(BAG_WFS, params=params, timeout=60)
        r.raise_for_status()
        page = r.json()["features"]
        features.extend(page)
        print(f"  buildings: {len(features)}")
        if len(page) < 1000:
            break
        start += 1000

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:28992")
    print(f"  statuses in this area: {gdf['status'].value_counts().to_dict()}")
    gdf.to_file(out_path, driver="GPKG")
    return gdf


def download_tile(x0, y0, year, out_path):
    """Download one TILE_SIZE_M x TILE_SIZE_M aerial photo tile with its lower-left corner at (x0, y0)."""
    px = int(TILE_SIZE_M / RESOLUTION)
    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": f"{year}_ortho25",
        "styles": "",
        "crs": "EPSG:28992",
        "bbox": f"{x0},{y0},{x0 + TILE_SIZE_M},{y0 + TILE_SIZE_M}",
        "width": px,
        "height": px,
        "format": "image/jpeg",
    }
    r = requests.get(LUCHTFOTO_WMS, params=params, timeout=120)
    r.raise_for_status()
    if not r.headers.get("Content-Type", "").startswith("image"):
        raise RuntimeError(f"WMS did not return an image: {r.text[:300]}")

    img = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
    # The WMS gives a plain JPEG. To make it a real geodata file we attach the coordinates:
    # top-left corner is (x0, y0 + size), each pixel is RESOLUTION metres.
    transform = from_origin(x0, y0 + TILE_SIZE_M, RESOLUTION, RESOLUTION)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=px, width=px, count=3, dtype="uint8",
        crs="EPSG:28992", transform=transform, compress="deflate",
    ) as dst:
        dst.write(img.transpose(2, 0, 1))


def main():
    parser = argparse.ArgumentParser()
    # Default: 1 x 1 km around the centre of Gouda
    parser.add_argument("--bbox", nargs=4, type=float, default=[107000, 446500, 108000, 447500])
    parser.add_argument("--year", default="2025")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--imagery-only", action="store_true",
                        help="skip the BAG download, e.g. when fetching an older year for change detection")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "imagery").mkdir(parents=True, exist_ok=True)

    if not args.imagery_only:
        print("Downloading BAG building footprints...")
        gdf = download_buildings(args.bbox, out / "bag_pand.gpkg")

        labels = gdf[gdf["status"].isin(EXISTING)]
        # A building with bouwjaar after the photo year can't be in the photo.
        labels = labels[labels["bouwjaar"].astype(int) <= int(args.year)]
        labels.to_file(out / "labels.gpkg", driver="GPKG")
        print(f"  kept {len(labels)} of {len(gdf)} as labels")

    print(f"Downloading {args.year} aerial imagery...")
    xmin, ymin, xmax, ymax = args.bbox
    for x0 in np.arange(xmin, xmax, TILE_SIZE_M):
        for y0 in np.arange(ymin, ymax, TILE_SIZE_M):
            name = out / "imagery" / f"luchtfoto_{args.year}_{int(x0)}_{int(y0)}.tif"
            if name.exists():
                continue
            download_tile(x0, y0, args.year, name)
            print(f"  {name.name}")

    print("Done.")


if __name__ == "__main__":
    main()
