"""
Nasapower district polygon pipeline SOIL.py
============================================
Fetches root-zone soil wetness (GWETROOT) from NASA POWER for each of the 25
Sri Lankan districts, following the same centre-in-polygon masking rule as
the existing rainfall pipeline.

This is the sister file of `Nasapower district polygon pipeline.py`. Keep the
two scripts aligned. If you change the polygon masking rule in one, change it
in the other, otherwise the rainfall and soil-moisture series will describe
different footprints and the propagation-time analysis breaks.

Output layout (matches district_rainfall/):
    district_soil/<District>.xlsx
      columns: date, gwetroot  (fraction, 0 to 1)
      one row per month, no gaps, from 1984-01 to the last full month
"""
from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

# -------------------------------------------------------- configuration

GADM_SHAPEFILE = Path("gadm41_LKA/gadm41_LKA_2.shp")
OUTPUT_DIR = Path("district_soil")
START_YEAR = 1984
END_YEAR = 2025

# NASA POWER endpoint. GWETROOT is a MERRA-2 land surface parameter,
# served through the regional monthly endpoint.
POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/regional"
POWER_PARAM = "GWETROOT"
POWER_COMMUNITY = "AG"                          # agroclimatology community

# Grid spacing of NASA POWER regional service (identical to the rainfall pipeline)
CELL_DLAT = 0.5
CELL_DLON = 0.625


def load_districts() -> gpd.GeoDataFrame:
    """Load and dissolve GADM level-2 features by NAME_1 to yield 25 districts.

    This is the same reduction the dashboard performs at startup, so the
    polygons match one-for-one.
    """
    gdf = gpd.read_file(GADM_SHAPEFILE)
    gdf = gdf.dissolve(by="NAME_1", as_index=False)
    gdf = gdf.rename(columns={"NAME_1": "District"})[["District", "geometry"]]
    return gdf.to_crs(epsg=4326)


def bbox_padded(polygon, min_range_deg: float = 2.05) -> tuple[float, float, float, float]:
    """Bounding box guaranteed to span at least min_range_deg in each dimension.

    POWER's monthly regional endpoint rejects any bbox narrower than 2 degrees
    with HTTP 422: "Please provide at least a 2 degree range in latitude...".
    Sri Lankan districts are all smaller than that, so we symmetrically pad
    around the polygon's centroid until we hit the minimum. We add a small
    safety margin (2.05 rather than 2.0) so floating-point rounding doesn't
    trip the check at the boundary. The centre-in-polygon masking downstream
    discards every cell that isn't actually inside the district, so the
    larger query area doesn't contaminate the result.
    """
    minx, miny, maxx, maxy = polygon.bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    half = min_range_deg / 2
    x_half = max(half, (maxx - minx) / 2 + 0.1)
    y_half = max(half, (maxy - miny) / 2 + 0.1)
    return (cx - x_half, cy - y_half, cx + x_half, cy + y_half)


def fetch_power_range(lon_min: float, lat_min: float,
                      lon_max: float, lat_max: float,
                      start_year: int, end_year: int) -> pd.DataFrame:
    """Query POWER for a multi-year range over the padded bounding box.

    POWER's monthly regional endpoint rejects single-year queries with HTTP 422
    (Unprocessable Entity). It needs at least a two-year span. The nasapower R
    package documentation shows the correct shape:
        dates = c("1984", "1985")
    for monthly regional queries. We fetch the whole 1984-2025 record in one
    call, which is 25 API calls total for the whole island rather than
    25 x 42 = 1,050 calls.

    Returns a long DataFrame with columns [lat, lon, date, gwetroot].
    """
    params = {
        "start": f"{start_year}",
        "end": f"{end_year}",
        "latitude-min": f"{lat_min}",
        "latitude-max": f"{lat_max}",
        "longitude-min": f"{lon_min}",
        "longitude-max": f"{lon_max}",
        "community": POWER_COMMUNITY,
        "parameters": POWER_PARAM,
        "format": "JSON",
        "header": "true",
    }
    last_error = None
    for attempt in range(3):
        r = requests.get(POWER_URL, params=params, timeout=300)
        if r.status_code == 200:
            break
        last_error = f"{r.status_code}: {r.text[:200]}"
        if attempt == 0:
            # Log the actual server message on first failure, then retry
            print(f"      POWER returned {last_error}")
        time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"POWER API failed after 3 attempts: {last_error}")

    payload = r.json()
    features = payload.get("features", [])
    rows = []
    for feat in features:
        lon, lat = feat["geometry"]["coordinates"][:2]
        series = feat["properties"]["parameter"][POWER_PARAM]
        for date_key, value in series.items():
            # POWER month keys look like YYYYMM or YYYY13 for annual mean; skip annual
            if len(date_key) != 6 or date_key.endswith("13"):
                continue
            y = int(date_key[:4])
            m = int(date_key[4:6])
            rows.append({"lat": lat, "lon": lon,
                         "date": pd.Timestamp(y, m, 1),
                         "gwetroot": value})
    return pd.DataFrame(rows)


def mask_centre_in_polygon(cells: pd.DataFrame, polygon) -> pd.DataFrame:
    """Keep only cells whose centre lies inside the district polygon.

    Same rule as the rainfall pipeline so the two variables share a footprint.
    """
    if cells.empty:
        return cells
    keep = cells.apply(
        lambda r: polygon.contains(Point(r["lon"], r["lat"])), axis=1)
    return cells[keep]


def process_district(name: str, polygon) -> pd.DataFrame:
    """Fetch, mask and monthly-average one district's soil moisture.

    Single call for the whole 1984-2025 span. See fetch_power_range for why.
    """
    minx, miny, maxx, maxy = bbox_padded(polygon)
    try:
        raw = fetch_power_range(minx, miny, maxx, maxy, START_YEAR, END_YEAR)
    except Exception as e:
        print(f"  ! {name}: {e}")
        return pd.DataFrame(columns=["date", "gwetroot"])

    if raw.empty:
        return pd.DataFrame(columns=["date", "gwetroot"])

    # Drop the POWER fill value
    raw = raw[raw["gwetroot"] > -900]

    # Mask to the true polygon
    kept = mask_centre_in_polygon(raw, polygon)
    if kept.empty:
        # Tiny district (Colombo, Kalmunai): fall back to the nearest cell centre
        raw["dist"] = raw.apply(
            lambda r: polygon.centroid.distance(Point(r["lon"], r["lat"])), axis=1)
        keep_lat = raw.loc[raw["dist"].idxmin(), "lat"]
        keep_lon = raw.loc[raw["dist"].idxmin(), "lon"]
        kept = raw[(raw["lat"] == keep_lat) & (raw["lon"] == keep_lon)].copy()
        n_cells = 1
    else:
        n_cells = kept[["lat", "lon"]].drop_duplicates().shape[0]

    # Monthly average across the retained cells
    monthly = (kept.groupby("date", as_index=False)["gwetroot"].mean()
                    .sort_values("date").reset_index(drop=True))

    # Record the per-district cell count so we can report the resolution
    # limitation honestly (paper's honest-note principle)
    monthly["n_cells"] = n_cells
    return monthly


def main():
    """Process every district and write district_soil/<District>.xlsx."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    districts = load_districts()
    print(f"Processing {len(districts)} districts for GWETROOT, "
          f"{START_YEAR}-{END_YEAR}")

    summary = []
    for _, row in districts.iterrows():
        name = row["District"]
        out_file = OUTPUT_DIR / f"{name}.xlsx"
        if out_file.exists():
            print(f"  skip {name} (already exists)")
            continue

        print(f"  fetching {name}...")
        df = process_district(name, row["geometry"])
        if df.empty:
            print(f"  !! {name}: no data")
            continue

        df.to_excel(out_file, index=False)
        n_cells = int(df["n_cells"].iloc[0])
        print(f"     {len(df)} months, {n_cells} grid cells")
        summary.append({"district": name, "months": len(df), "cells": n_cells})

    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "_summary.csv", index=False)
    print(f"\nDone. Cell counts written to {OUTPUT_DIR / '_summary.csv'}")


if __name__ == "__main__":
    main()