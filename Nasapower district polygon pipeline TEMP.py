"""
Nasapower district polygon pipeline TEMP.py
============================================
Fetches monthly temperature from NASA POWER for each of the 25 Sri Lankan
districts, following the same centre-in-polygon masking rule as the rainfall
and soil-moisture pipelines. Temperature is the second variable of the MSDI
(Karunarathne, De Silva & Perera 2024).

Two parameters are fetched, because the paper says "ground temperature" but
does not name the NASA POWER variable:
    T2M  temperature at 2 metres (degC)
    TS   earth skin temperature  (degC)

Sister file of `Nasapower district polygon pipeline SOIL.py`. Keep the masking
rule aligned with it, otherwise rainfall and temperature describe different
footprints and the rainfall-temperature copula is fitted on mismatched data.

Output layout:
    district_temperature/<District>.xlsx
      columns: Year, Month, t2m, ts, n_cells
      one row per month, 1984-01 to 2025-12
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
OUTPUT_DIR = Path("district_temperature")
START_YEAR = 1984
END_YEAR = 2025

POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/regional"
POWER_PARAMS = {"T2M": "t2m", "TS": "ts"}       # POWER name -> output column
POWER_COMMUNITY = "AG"


def load_districts() -> gpd.GeoDataFrame:
    """Same 25-district reduction of GADM level 2 the dashboard uses."""
    gdf = gpd.read_file(GADM_SHAPEFILE)
    gdf = gdf.dissolve(by="NAME_1", as_index=False)
    gdf = gdf.rename(columns={"NAME_1": "District"})[["District", "geometry"]]
    return gdf.to_crs(epsg=4326)


def bbox_padded(polygon, min_range_deg: float = 2.05) -> tuple[float, float, float, float]:
    """Bounding box spanning at least 2 degrees (POWER rejects smaller boxes)."""
    minx, miny, maxx, maxy = polygon.bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    half = min_range_deg / 2
    x_half = max(half, (maxx - minx) / 2 + 0.1)
    y_half = max(half, (maxy - miny) / 2 + 0.1)
    return (cx - x_half, cy - y_half, cx + x_half, cy + y_half)


def fetch_power_param(param: str, lon_min: float, lat_min: float,
                      lon_max: float, lat_max: float) -> pd.DataFrame:
    """One regional query for one parameter over the whole 1984-2025 span.

    Returns a long DataFrame with columns [lat, lon, Year, Month, value].
    """
    params = {
        "start": f"{START_YEAR}", "end": f"{END_YEAR}",
        "latitude-min": f"{lat_min}", "latitude-max": f"{lat_max}",
        "longitude-min": f"{lon_min}", "longitude-max": f"{lon_max}",
        "community": POWER_COMMUNITY, "parameters": param,
        "format": "JSON", "header": "true",
    }
    last_error = None
    for attempt in range(3):
        try:
            r = requests.get(POWER_URL, params=params, timeout=300)
        except requests.RequestException as e:
            last_error = str(e)
        else:
            if r.status_code == 200:
                break
            last_error = f"{r.status_code}: {r.text[:200]}"
        print(f"      {param} attempt {attempt + 1} failed: {last_error}")
        time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"POWER API failed after 3 attempts: {last_error}")

    rows = []
    for feat in r.json().get("features", []):
        lon, lat = feat["geometry"]["coordinates"][:2]
        for key, value in feat["properties"]["parameter"][param].items():
            # Keys are YYYYMM; YYYY13 is the annual mean
            if len(key) != 6 or key.endswith("13"):
                continue
            rows.append({"lat": lat, "lon": lon, "Year": int(key[:4]),
                         "Month": int(key[4:]), "value": float(value)})
    return pd.DataFrame(rows)


def select_cells(cells: pd.DataFrame, polygon) -> pd.DataFrame:
    """Centre-in-polygon mask, with nearest-cell fallback for tiny districts.

    Same rule as the rainfall and soil pipelines.
    """
    grid = cells[["lat", "lon"]].drop_duplicates()
    inside = grid[grid.apply(lambda r: polygon.contains(Point(r["lon"], r["lat"])), axis=1)]
    if inside.empty:
        c = polygon.centroid
        d = grid.apply(lambda r: c.distance(Point(r["lon"], r["lat"])), axis=1)
        inside = grid.loc[[d.idxmin()]]
    return cells.merge(inside, on=["lat", "lon"])


def process_district(name: str, polygon) -> pd.DataFrame:
    """Fetch T2M and TS for one district and average the masked cells per month.

    Returns Year, Month, t2m, ts and n_cells (number of grid cells used).
    """
    minx, miny, maxx, maxy = bbox_padded(polygon)
    out = None
    n_cells = 0
    for param, col in POWER_PARAMS.items():
        raw = fetch_power_param(param, minx, miny, maxx, maxy)
        raw = raw[raw["value"] > -900]                      # POWER fill value
        kept = select_cells(raw, polygon)
        n_cells = kept[["lat", "lon"]].drop_duplicates().shape[0]
        monthly = (kept.groupby(["Year", "Month"], as_index=False)["value"].mean()
                        .rename(columns={"value": col}))
        out = monthly if out is None else out.merge(monthly, on=["Year", "Month"], how="outer")
        time.sleep(1)
    out = out.sort_values(["Year", "Month"]).reset_index(drop=True)
    out["n_cells"] = n_cells
    return out


def main():
    """Process every district and write district_temperature/<District>.xlsx."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    districts = load_districts()
    print(f"Processing {len(districts)} districts for "
          f"{', '.join(POWER_PARAMS)}, {START_YEAR}-{END_YEAR}")

    for _, row in districts.iterrows():
        name = row["District"]
        out_file = OUTPUT_DIR / f"{name}.xlsx"
        if out_file.exists():
            print(f"  skip {name} (already exists)")
            continue
        print(f"  fetching {name}...")
        try:
            df = process_district(name, row["geometry"])
        except Exception as e:
            print(f"  !! {name}: {e}")
            continue
        df.to_excel(out_file, index=False)
        print(f"     {len(df)} months, {int(df['n_cells'].iloc[0])} grid cells")

    # Rebuild the summary from every file on disk, so re-runs that skip
    # districts still produce a complete table
    summary = []
    for f in sorted(OUTPUT_DIR.glob("*.xlsx")):
        df = pd.read_excel(f)
        summary.append({"district": f.stem, "months": len(df),
                        "cells": int(df["n_cells"].iloc[0]),
                        "t2m_mean": round(df["t2m"].mean(), 2),
                        "ts_mean": round(df["ts"].mean(), 2)})
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "_summary.csv", index=False)
    print(f"\nDone. Summary written to {OUTPUT_DIR / '_summary.csv'}")


if __name__ == "__main__":
    main()
