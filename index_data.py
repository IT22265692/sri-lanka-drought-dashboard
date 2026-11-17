"""
index_data.py
=============
One place that serves every drought index to the dashboard, for all 25
districts, in the same long shape:

    District, Year, Month, value, Date, class

Where it sits in the pipeline:

    SPI   spi_copula_analysis.compute_spi    (rainfall)
    SSI   ssi_analysis.compute_ssi           (root-zone soil moisture)
    MSDI  cache/msdi_fits.pkl                (rainfall + TS, run_msdi_all_districts.py)
          -> index_data.build_tables()        (this module)
          -> cache/index_tables.pkl           (read by the dashboard and compare_indices.py)

The tables are built once and saved, so the dashboard opens quickly. They are
rebuilt automatically when a data file or the MSDI fits are newer:

    py -3.12 index_data.py          rebuild the tables now

SPI and SSI use the McKee classes and are dry at or below -1. The published
MSDI is not standardised, so it has no fixed classes here: thresholds.apply adds
them from the paper's Table 4 when a page reads the table.

Duration-severity analysis uses the same copula code
(spi_copula_analysis.model_from_events) for SPI and MSDI. The SSI page shows events
only. Events are runs below 0 for SPI and SSI, and below the dry cut-off for MSDI.
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common
import msdi_analysis as M
import thresholds
from common import list_districts                      # noqa: F401  (re-exported for the pages)
from spi_copula_analysis import compute_spi, extract_drought_events, model_from_events, spi_category
from ssi_analysis import compute_ssi
from theme import INDEX_COLOURS

MSDI_CACHE = common.CACHE_DIR / "msdi_fits.pkl"
TABLE_CACHE = common.CACHE_DIR / "index_tables.pkl"

MSDI_TEMP = M.TEMP_COL            # the paper's ground temperature (NASA POWER TS)
SCALES = common.SCALES            # SPI and SSI
MSDI_SCALES = M.ALL_SCALES        # the paper's T = 3, 6, 9, 12, plus the one-month MSDI they average
DOCUMENTED_DROUGHTS = common.DOCUMENTED_DROUGHTS

# Display name, short name and line colour of each index. Blue, vermilion and
# purple: checked for colour-blind separation and 3:1 contrast on white.
INDEX_INFO = {
    "SPI":  {"name": "Rainfall (SPI)", "short": "SPI", "colour": INDEX_COLOURS["SPI"],
             "what": "How unusual the rainfall was."},
    "SSI":  {"name": "Soil moisture (SSI)", "short": "SSI", "colour": INDEX_COLOURS["SSI"],
             "what": "How unusual the root-zone soil moisture was."},
    "MSDI": {"name": "Rainfall + heat (MSDI)", "short": "MSDI", "colour": INDEX_COLOURS["MSDI"],
             "what": "Rainfall and temperature together, by the published method."},
}


# =============================================================================
# 1. ONE DISTRICT, ONE INDEX  ->  Year, Month, value
# =============================================================================

def spi_series(district: str, scale: int) -> pd.DataFrame:
    """SPI-k for one district: Year, Month, value."""
    s = compute_spi(common.read_rainfall(district), scale=scale)
    return s[["Year", "Month", f"SPI{scale}"]].rename(columns={f"SPI{scale}": "value"})


def ssi_series(district: str, scale: int) -> pd.DataFrame:
    """SSI-k for one district: Year, Month, value."""
    s = compute_ssi(common.read_soil(district), scale).ssi
    return pd.DataFrame({"Year": s.index.year, "Month": s.index.month, "value": s.values})


def _load(path: Path):
    """Unpickle a file, or None if it does not exist yet."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def msdi_series(district: str, scale: int, cache: dict | None = None) -> pd.DataFrame | None:
    """MSDI-T for one district from the saved fits: Year, Month, value (None if not fitted)."""
    cache = cache if cache is not None else (_load(MSDI_CACHE) or {})
    m = cache.get(M.model_key(district, MSDI_TEMP))
    if m is None:
        return None
    return M.msdi_series(m, scale).rename(columns={"MSDI": "value"})


# =============================================================================
# 2. ALL DISTRICTS: BUILD AND CACHE
# =============================================================================

def _finish(df: pd.DataFrame, district: str, mckee: bool = True) -> pd.DataFrame:
    """Add District and Date to one district's series, and the McKee class and dry flag if it uses them."""
    df = df.copy()
    df.insert(0, "District", district)
    df["Date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))
    if mckee:
        df["class"] = [spi_category(v)[0] if pd.notna(v) else np.nan for v in df["value"]]
        df["dry"] = df["value"] <= common.DRY_THRESHOLD
    return df


def build_tables(verbose: bool = True) -> dict:
    """Compute every index for every district and scale.

    Returns {"tables": {(index, scale): long DataFrame}, "notes": [...],
    "built": time stamp, "sources_mtime": newest input file time}.
    """
    t0 = time.time()
    msdi_cache = _load(MSDI_CACHE) or {}
    tables, notes = {}, []
    for k in SCALES:
        for name, fn in [("SPI", spi_series), ("SSI", ssi_series)]:
            tables[(name, k)] = pd.concat([_finish(fn(d, k), d) for d in list_districts()],
                                          ignore_index=True)
            if verbose:
                print(f"  {name}-{k} done ({time.time() - t0:.0f}s)", flush=True)
    for k in MSDI_SCALES:
        parts, missing = [], []
        for d in list_districts():
            s = msdi_series(d, k, msdi_cache)
            if s is None:
                missing.append(d)
            else:
                parts.append(_finish(s, d, mckee=False))
        if parts:
            tables[("MSDI", k)] = pd.concat(parts, ignore_index=True)
        if missing:
            notes.append(f"MSDI-{k} ({MSDI_TEMP.upper()}) not fitted yet for: {', '.join(missing)}")
    return {"tables": tables, "notes": notes, "built": time.strftime("%Y-%m-%d %H:%M"),
            "sources_mtime": sources_mtime()}


def classified_table(table: pd.DataFrame, index: str, scale: int) -> pd.DataFrame:
    """Add the MSDI classes from Table 4. SPI and SSI already carry theirs."""
    return thresholds.apply(table, scale) if index == "MSDI" else table


def sources_mtime() -> float:
    """Modification time of the newest input (data files and MSDI fits)."""
    paths = [MSDI_CACHE, *common.RAIN_DIR.glob("*.xlsx"), *common.SOIL_DIR.glob("*.xlsx")]
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def save_tables(bundle: dict):
    """Write the tables to cache/index_tables.pkl via a temporary file (never half-written)."""
    TABLE_CACHE.parent.mkdir(exist_ok=True)
    tmp = TABLE_CACHE.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(bundle, f)
    os.replace(tmp, TABLE_CACHE)


def load_tables() -> dict:
    """Saved tables, rebuilt automatically when a data file or fit is newer."""
    bundle = _load(TABLE_CACHE)
    if bundle is None or bundle.get("sources_mtime", 0) < sources_mtime():
        bundle = build_tables(verbose=False)
        save_tables(bundle)
    return bundle


# =============================================================================
# 3. DROUGHT EVENTS AND DURATION-SEVERITY COPULA (any index)
# =============================================================================

def index_events(series: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Run-theory events (index < threshold) of one series, with no fitted model.

    series: Year, Month, value for one district. The SSI page stops here, because
    SSI has no copula or return periods at this stage.
    """
    s = series.dropna(subset=["value"]).sort_values(["Year", "Month"]).reset_index(drop=True)
    events = extract_drought_events(s.assign(IDX=s["value"] - threshold), "IDX")
    return events[events["Severity"] > 0].reset_index(drop=True)   # rounding can give 0


def events_and_model(series: pd.DataFrame, district: str, scale: int, threshold: float = 0.0):
    """Run-theory events (index < threshold) and the duration-severity copula model.

    series: Year, Month, value for one district. threshold: 0 for SPI and SSI,
    the dry cut-off for MSDI, where severity is then the sum of the distance
    below the cut-off. Uses the SPI module's extract_drought_events and
    model_from_events, so every index is analysed with the same code.
    Returns (DistrictModel or None, events).
    """
    events = index_events(series, threshold)
    try:
        model = model_from_events(events, district, scale, n_months=int(series["value"].notna().sum()))
    except ValueError:
        model = None                       # fewer than 10 events: no copula fit
    return model, events


if __name__ == "__main__":
    print("Building index tables for all districts...")
    b = build_tables()
    save_tables(b)
    for (name, k), t in sorted(b["tables"].items()):
        print(f"  {name:6s} scale {k}: {t['District'].nunique()} districts, {len(t)} rows")
    for n in b["notes"]:
        print("  NOTE:", n)
    print(f"Saved {TABLE_CACHE}")
