"""
NASA POWER -> District-Averaged Monthly Rainfall
=================================================
PRODUCTION VERSION: uses TRUE district polygon boundaries (GADM level 2),
not bounding-box approximations.

Pipeline:
  1. First run only: auto-download GADM Sri Lanka district polygons.
  2. For each district:
        a. Read the district's polygon from GADM.
        b. Compute the polygon's bounding box.
        c. Query NASA POWER's REGIONAL endpoint for that bounding box
           (monthly PRECTOTCORR, 1984-2025, chunked by year).
        d. Convert each cell from mm/day -> monthly total (x days in month).
        e. Keep ONLY grid cells whose centre falls INSIDE the true polygon.
        f. Average the remaining cells per Year-Month.
  3. Save one Excel file per district (matches your existing
     Pre_processed_data_Anuradhapura.xlsx format) + one combined CSV.

Requirements:
    pip install requests pandas numpy openpyxl geopandas shapely pyogrio

First run downloads ~30MB from GADM. Subsequent runs skip the download.

WHY THIS DESIGN:
  Bounding boxes are quick but include grid cells that geographically
  belong to neighbouring districts, biasing the average -- especially bad
  for districts with irregular shapes (Trincomalee, Batticaloa,
  Hambantota). Masking to the true polygon ensures every averaged cell
  really is inside the district being reported.
"""

import os
import time
import zipfile
import io
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
 
START_YEAR = 1984
END_YEAR = 2025
YEARS_PER_CHUNK = 7
MIN_BOX_DEG = 2.2   # NASA POWER requires >=2 deg; 2.2 gives a safe margin
 
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_LKA_shp.zip"
GADM_DIR = "gadm41_LKA"
GADM_DISTRICTS_FILE = os.path.join(GADM_DIR, "gadm41_LKA_2.shp")
 
OUTPUT_DIR = "district_rainfall"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
 
# =============================================================================
# STEP 1: Download GADM shapefile (once)
# =============================================================================
def ensure_gadm():
    if os.path.isfile(GADM_DISTRICTS_FILE):
        print(f"GADM shapefile already present: {GADM_DISTRICTS_FILE}")
        return
    print(f"Downloading GADM Sri Lanka shapefile...")
    r = requests.get(GADM_URL, timeout=180)
    r.raise_for_status()
    os.makedirs(GADM_DIR, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(GADM_DIR)
    print(f"Extracted to {GADM_DIR}/")
 
 
# =============================================================================
# STEP 2: Load 25 Sri Lanka districts by dissolving DS divisions on NAME_1
# =============================================================================
def load_districts_gdf() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(GADM_DISTRICTS_FILE)
    # In GADM's Sri Lanka data, NAME_1 == District, NAME_2 == DS Division
    gdf = gdf.dissolve(by="NAME_1", as_index=False)
    gdf = gdf.rename(columns={"NAME_1": "District"})[["District", "geometry"]]
    gdf = gdf.to_crs(epsg=4326)  # WGS84 lat/lon
    return gdf
 
 
# =============================================================================
# STEP 3: Expand bounding box to satisfy NASA POWER's 2 deg minimum
# =============================================================================
def expand_bbox(lat_min, lat_max, lon_min, lon_max, min_deg=MIN_BOX_DEG):
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    if lat_range < min_deg:
        pad = (min_deg - lat_range) / 2
        lat_min -= pad
        lat_max += pad
    if lon_range < min_deg:
        pad = (min_deg - lon_range) / 2
        lon_min -= pad
        lon_max += pad
    return lat_min, lat_max, lon_min, lon_max
 
 
# =============================================================================
# STEP 4: Fetch NASA POWER regional data for a bounding box
# =============================================================================
def _fetch_regional_chunk(lat_min, lat_max, lon_min, lon_max, start_year, end_year,
                           max_retries=3):
    url = "https://power.larc.nasa.gov/api/temporal/monthly/regional"
    params = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "latitude-min": lat_min, "latitude-max": lat_max,
        "longitude-min": lon_min, "longitude-max": lon_max,
        "start": start_year, "end": end_year,
        "format": "JSON",
    }
    payload = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=180)
            if resp.status_code >= 400:
                print(f"      HTTP {resp.status_code}: {resp.text[:250]}")
            resp.raise_for_status()
            payload = resp.json()
            break
        except (requests.RequestException, ValueError) as e:
            if attempt == max_retries:
                raise
            print(f"      retry {attempt}/{max_retries} after error: {e}")
            time.sleep(5 * attempt)
 
    rows = []
    for feature in payload.get("features", []):
        lon, lat = feature["geometry"]["coordinates"][:2]
        monthly = feature["properties"]["parameter"]["PRECTOTCORR"]
        for key, value in monthly.items():
            month_num = int(key[-2:])
            if month_num == 13:  # skip annual-average pseudo-month
                continue
            year = int(key[:4])
            rows.append({"Lat": lat, "Lon": lon, "Year": year, "Month": month_num,
                          "Precip_mm_per_day": float(value)})
    return pd.DataFrame(rows)
 
 
def fetch_regional_precip(lat_min, lat_max, lon_min, lon_max, start_year, end_year):
    chunks = []
    y = start_year
    while y <= end_year:
        y_end = min(y + YEARS_PER_CHUNK - 1, end_year)
        print(f"      years {y}-{y_end}...")
        chunks.append(_fetch_regional_chunk(lat_min, lat_max, lon_min, lon_max, y, y_end))
        y = y_end + 1
        time.sleep(1)
    return pd.concat(chunks, ignore_index=True)
 
 
# =============================================================================
# STEP 5: Mask grid cells to a polygon, then average per Year-Month
# =============================================================================
def mask_and_average(raw: pd.DataFrame, polygon):
    days_in_month = raw.apply(
        lambda r: pd.Period(year=int(r.Year), month=int(r.Month), freq="M").days_in_month,
        axis=1,
    )
    raw = raw.copy()
    raw["precip_month_total"] = raw["Precip_mm_per_day"] * days_in_month
 
    unique_cells = raw[["Lat", "Lon"]].drop_duplicates().reset_index(drop=True)
    cells_gdf = gpd.GeoDataFrame(
        unique_cells,
        geometry=[Point(lon, lat) for lat, lon in zip(unique_cells["Lat"], unique_cells["Lon"])],
        crs="EPSG:4326",
    )
    inside_mask = cells_gdf.within(polygon)
    cells_inside = cells_gdf[inside_mask][["Lat", "Lon"]]
 
    n_total = len(unique_cells)
    n_inside = len(cells_inside)
 
    if n_inside == 0:
        # District polygon smaller than the NASA POWER grid resolution
        # (~0.5 deg); no grid cell centre lies inside. Fall back to the
        # single cell nearest the polygon centroid so we still get a value.
        centroid = polygon.centroid
        unique_cells["dist"] = np.hypot(
            unique_cells["Lat"] - centroid.y, unique_cells["Lon"] - centroid.x
        )
        cells_inside = unique_cells.nsmallest(1, "dist")[["Lat", "Lon"]]
        n_inside = 1
        print("      WARNING: 0 grid cells inside polygon; using nearest cell instead.")
 
    filtered = raw.merge(cells_inside, on=["Lat", "Lon"], how="inner")
    avg = (
        filtered.groupby(["Year", "Month"], as_index=False)["precip_month_total"]
        .mean()
        .rename(columns={"precip_month_total": "prcp"})
        .sort_values(["Year", "Month"])
        .reset_index(drop=True)
    )
    return avg, n_total, n_inside
 
 
# =============================================================================
# STEP 6: Run for every district
# =============================================================================
def run_all_districts():
    ensure_gadm()
    districts_gdf = load_districts_gdf()
    print(f"\nLoaded {len(districts_gdf)} districts from GADM.\n")
 
    combined = []
    audit_rows = []
 
    for i, row in districts_gdf.iterrows():
        district = row["District"]
        polygon = row["geometry"]
 
        lon_min, lat_min, lon_max, lat_max = polygon.bounds
        raw_bounds = (lat_min, lat_max, lon_min, lon_max)
        lat_min, lat_max, lon_min, lon_max = expand_bbox(lat_min, lat_max, lon_min, lon_max)
 
        print(f"[{i + 1}/{len(districts_gdf)}] {district}")
        print(f"    true polygon bounds:  lat {raw_bounds[0]:.2f}-{raw_bounds[1]:.2f}  "
              f"lon {raw_bounds[2]:.2f}-{raw_bounds[3]:.2f}  "
              f"({raw_bounds[1]-raw_bounds[0]:.2f}deg x {raw_bounds[3]-raw_bounds[2]:.2f}deg)")
        print(f"    expanded query bbox:  lat {lat_min:.2f}-{lat_max:.2f}  "
              f"lon {lon_min:.2f}-{lon_max:.2f}  "
              f"({lat_max-lat_min:.2f}deg x {lon_max-lon_min:.2f}deg)")
 
        raw = fetch_regional_precip(lat_min, lat_max, lon_min, lon_max, START_YEAR, END_YEAR)
        avg, n_total, n_inside = mask_and_average(raw, polygon)
        print(f"    grid cells: {n_inside} inside polygon (of {n_total} in bounding box)")
 
        avg.insert(0, "District", district)
        combined.append(avg)
 
        out_name = os.path.join(
            OUTPUT_DIR, f"Pre_processed_data_{district.replace(' ', '_')}.xlsx"
        )
        avg[["Year", "Month", "prcp"]].to_excel(out_name, index=False)
        print(f"    saved: {out_name}  ({len(avg)} rows)\n")
 
        audit_rows.append({"District": district, "cells_in_bbox": n_total,
                            "cells_inside_polygon": n_inside, "rows": len(avg)})
        time.sleep(1)
 
    combined_df = pd.concat(combined, ignore_index=True)
    combined_csv = os.path.join(OUTPUT_DIR, "Pre_processed_data_ALL_DISTRICTS.csv")
    combined_df.to_csv(combined_csv, index=False)
 
    audit_df = pd.DataFrame(audit_rows)
    audit_csv = os.path.join(OUTPUT_DIR, "grid_cell_audit.csv")
    audit_df.to_csv(audit_csv, index=False)
 
    print(f"\nSaved combined: {combined_csv}  ({len(combined_df)} rows)")
    print(f"Saved audit:    {audit_csv}")
    print("\n=== Row count per district (expect 504) ===")
    print(combined_df.groupby("District").size().sort_values())
    print("\n=== Grid-cell audit ===")
    print(audit_df.sort_values("cells_inside_polygon"))
 
    return combined_df, audit_df
 
 
if __name__ == "__main__":
    run_all_districts()