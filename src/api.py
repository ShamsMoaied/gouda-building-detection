"""
Step 5b: a small REST API that serves the detected buildings as GeoJSON.

Run:
    uvicorn api:app --app-dir src --reload
Then open http://127.0.0.1:8000/docs for the interactive documentation.

Examples:
    /buildings?bbox=107600,446600,107700,446700
    /buildings?bbox=107600,446600,107700,446700&min_score=0.8&crs=4326
    /changes
"""

import json
import os
from pathlib import Path

import geopandas as gpd
from fastapi import FastAPI, HTTPException, Query
from shapely.geometry import box

PRED_FILE = Path(os.environ.get("PRED_FILE", "data/predictions/luchtfoto_2025_107500_446500_buildings.gpkg"))
CHANGE_FILE = Path(os.environ.get("CHANGE_FILE", "data/predictions/changes_2020_2025.gpkg"))

app = FastAPI(title="Gouda building detection API",
              description="Buildings detected by Mask R-CNN in 25 cm aerial imagery (EPSG:28992).")

buildings = gpd.read_file(PRED_FILE) if PRED_FILE.exists() else None


def to_geojson(gdf, crs):
    if crs not in ("28992", "4326"):
        raise HTTPException(400, "crs must be 28992 or 4326")
    if crs == "4326":
        gdf = gdf.to_crs(4326)
    return json.loads(gdf.to_json())


@app.get("/health")
def health():
    return {"status": "ok", "buildings_loaded": 0 if buildings is None else len(buildings)}


@app.get("/buildings")
def get_buildings(
    bbox: str = Query(..., description="xmin,ymin,xmax,ymax in EPSG:28992 (metres)"),
    min_score: float = Query(0.5, ge=0, le=1),
    crs: str = Query("28992", description="output CRS: 28992 or 4326"),
):
    if buildings is None:
        raise HTTPException(503, f"no predictions found at {PRED_FILE}; run src/predict.py first")
    try:
        xmin, ymin, xmax, ymax = map(float, bbox.split(","))
    except ValueError:
        raise HTTPException(400, "bbox must be four numbers: xmin,ymin,xmax,ymax")
    if (xmax - xmin) * (ymax - ymin) > 1_000_000:
        raise HTTPException(400, "bbox too large, max 1 km2 per request")

    hits = buildings[buildings.intersects(box(xmin, ymin, xmax, ymax)) & (buildings.score >= min_score)]
    return to_geojson(hits, crs)


@app.get("/changes")
def get_changes(crs: str = Query("28992")):
    """Buildings that appear in 2025 but not in 2020 (output of src/change_detection.py)."""
    if not CHANGE_FILE.exists():
        raise HTTPException(503, "run src/change_detection.py first")
    return to_geojson(gpd.read_file(CHANGE_FILE), crs)
