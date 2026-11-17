"""
common.py
=========
Shared paths, data readers and constants used by every other module.

Where it sits in the pipeline:

    NASA POWER pipelines  ->  district_rainfall/, district_soil/, district_temperature/
    common.py             ->  reads those files (this module)
    spi / ssi / msdi      ->  compute the indices
    index_data.py         ->  serves all indices to the dashboard
    dashboard.py          ->  the Streamlit app

Keeping the readers here means every module loads the data in exactly the
same way, so SPI, SSI and MSDI always start from the same numbers.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------- paths

RAIN_DIR = Path("district_rainfall")          # Pre_processed_data_<District>.xlsx, monthly mm
SOIL_DIR = Path("district_soil")              # <District>.xlsx, GWETROOT (0-1)
TEMP_DIR = Path("district_temperature")       # <District>.xlsx, T2M and TS (degC)
CACHE_DIR = Path("cache")                     # fitted models and index tables (pickles)
GADM_SHP = Path("gadm41_LKA") / "gadm41_LKA_2.shp"   # district boundaries

# ----------------------------------------------------------------------------- constants

SCALES = (1, 3, 6)                 # accumulation windows in months, same for every index
DRY_THRESHOLD = -1.0               # "dry month": index at or below -1 (moderately dry or worse)
EVENT_THRESHOLD = 0.0              # run theory: a drought event is a run of months below 0
START_YEAR, END_YEAR = 1984, 2025  # study period

# Documented North Central Province droughts (Karunarathne et al. 2024, sections 4-5).
# Documented droughts shaded on the time-series charts.
# 1982-83 is also listed in the paper but is before the 1984 start of this data.
DOCUMENTED_DROUGHTS = [("1986-87", (1986, 1987)), ("1988-89", (1988, 1989)), ("1992", (1992,)),
                       ("1995-96", (1995, 1996)), ("2001", (2001,)), ("2004", (2004,)),
                       ("2013-14", (2013, 2014)), ("2016-17", (2016, 2017))]
VALIDATION_DISTRICTS = ("Anuradhapura", "Polonnaruwa")
# The reference table reaches 2023, so 2024-2025 are not counted as
# "no drought" years: their status is unknown.
VALIDATION_YEARS = range(1984, 2024)
MIN_DRY_MONTHS = 3                 # a year is "flagged" when it has at least 3 dry months


# ----------------------------------------------------------------------------- readers

def list_districts() -> list[str]:
    """Names of the 25 districts, taken from the rainfall file names.

    Returns a sorted list such as ["Ampara", "Anuradhapura", ...]. The rainfall
    folder is the reference because every index needs rainfall or starts from it.
    """
    files = sorted(RAIN_DIR.glob("Pre_processed_data_*.xlsx"))
    return [f.stem.replace("Pre_processed_data_", "").replace("_", " ") for f in files]


def read_rainfall(district: str) -> pd.DataFrame:
    """Monthly rainfall for one district.

    Returns the pipeline's table with at least Year, Month and prcp (monthly
    total in mm). Used by SPI and MSDI.
    """
    return pd.read_excel(RAIN_DIR / f"Pre_processed_data_{district.replace(' ', '_')}.xlsx")


def read_soil(district: str) -> pd.DataFrame:
    """Monthly root-zone soil wetness (GWETROOT, 0 to 1) for one district.

    Returns a table with a datetime "date" column and "gwetroot". Used by SSI.
    """
    soil = pd.read_excel(SOIL_DIR / f"{district}.xlsx")
    soil["date"] = pd.to_datetime(soil["date"])
    return soil


def read_temperature(district: str) -> pd.DataFrame:
    """Monthly temperature for one district.

    Returns Year, Month, t2m (air at 2 m) and ts (earth skin), both in degC.
    MSDI uses ts, the ground temperature of the published method.
    """
    return pd.read_excel(TEMP_DIR / f"{district}.xlsx")


def shared_cells() -> dict[str, str]:
    """District -> the other district(s) whose rainfall series is identical.

    Small districts can fall back to the same NASA POWER grid cell as a
    neighbour, so their results are the same and not independent. Returns a
    dict like {"Colombo": "Gampaha", "Gampaha": "Colombo", ...}.
    """
    r = pd.read_csv(RAIN_DIR / "Pre_processed_data_ALL_DISTRICTS.csv")
    wide = r.pivot_table(index=["Year", "Month"], columns="District", values="prcp")
    out: dict[str, list[str]] = {}
    cols = list(wide.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            # Identical to numerical precision means the same grid cell
            if (wide[a] - wide[b]).abs().max() < 1e-9:
                out.setdefault(a, []).append(b)
                out.setdefault(b, []).append(a)
    return {k: ", ".join(v) for k, v in out.items()}
