"""
thresholds.py
=============
Class cut-offs for the published MSDI. The index is not standardised (its mean
is about -1.1), so it cannot use the SPI classes. It uses the paper's Table 4
cut-offs instead, the same values for every district, one pair for each T.

Where it sits in the pipeline:

    run_msdi_all_districts.py -> cache/msdi_fits.pkl
        -> thresholds.py (this module)
        -> index_data, compare_indices and the MSDI page of the dashboard
        -> msdi_outputs/common_threshold_shares.csv

A month is Extremely dry at or below the extreme cut-off, Dry at or below the dry
cut-off, and Normal above it. The paper sets no classes for the one-month MSDI.

    py -3.12 thresholds.py      recompute and save the shares
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import common
import msdi_analysis as M

T_SCALES = M.T_SCALES
OUT_DIR = Path("msdi_outputs")

# Paper Table 4: T -> (extremely dry at or below, dry at or below). Normal is everything above.
TABLE4 = {3: (-3.2, -1.3), 6: (-2.5, -0.8), 9: (-2.1, -0.4), 12: (-1.7, -0.3)}
CLASSES = ("Extremely dry", "Dry", "Normal")

PROVINCES = {
    "Western": ("Colombo", "Gampaha", "Kalutara"),
    "Central": ("Kandy", "Matale", "Nuwara Eliya"),
    "Southern": ("Galle", "Matara", "Hambantota"),
    "Northern": ("Jaffna", "Kilinochchi", "Mannar", "Mullaitivu", "Vavuniya"),
    "Eastern": ("Batticaloa", "Ampara", "Trincomalee"),
    "North Western": ("Kurunegala", "Puttalam"),
    "North Central": ("Anuradhapura", "Polonnaruwa"),
    "Uva": ("Badulla", "Moneragala"),
    "Sabaragamuwa": ("Ratnapura", "Kegalle"),
}
PROVINCE_OF = {d: p for p, ds in PROVINCES.items() for d in ds}


def cutoff(T: int) -> tuple[float, float] | None:
    """(extreme, dry) for one T, or None for a window the paper gives no classes for."""
    return TABLE4.get(T)


def classify(values: pd.Series, extreme: float, dry: float) -> pd.Series:
    """Class of each value: Extremely dry, Dry or Normal. A missing value stays missing."""
    out = pd.Series(np.where(values <= extreme, "Extremely dry", np.where(values <= dry, "Dry", "Normal")),
                    index=values.index, dtype=object)
    return out.where(values.notna())


def apply(table: pd.DataFrame, T: int) -> pd.DataFrame:
    """Add class, dry, cut_extreme and cut_dry to a long MSDI table with a value column.

    T = 1 has no classes in the paper, so class and the cut-offs stay empty and
    dry stays False.
    """
    cut = cutoff(T)
    if cut is None:
        return table.assign(**{"class": np.nan, "dry": False, "cut_extreme": np.nan, "cut_dry": np.nan})
    return table.assign(**{"class": classify(table["value"], *cut), "dry": table["value"] <= cut[1],
                           "cut_extreme": cut[0], "cut_dry": cut[1]})


def drought_events(table: pd.DataFrame, T: int) -> pd.DataFrame:
    """Drought events of one district: unbroken runs of months at or below the dry cut-off.

    table: Year, Month and value, one row per month. Severity is the sum of the
    amount below the cut-off, so a month exactly on it adds nothing. Returns start,
    end, duration_months, severity and lowest, one row per event.
    """
    dry = cutoff(T)[1]
    s = table.dropna(subset=["value"]).sort_values(["Year", "Month"]).reset_index(drop=True)
    below = s["value"] <= dry
    run = (below != below.shift(fill_value=False)).cumsum()     # a new number each time the state changes
    rows = []
    for _, g in s[below].groupby(run[below]):
        first, last = g.iloc[0], g.iloc[-1]
        rows.append({"start": f"{int(first['Year'])}-{int(first['Month']):02d}", "end": f"{int(last['Year'])}-{int(last['Month']):02d}",
                     "duration_months": len(g), "severity": round(float((dry - g["value"]).sum()), 2),
                     "lowest": round(float(g["value"].min()), 2)})
    return pd.DataFrame(rows, columns=["start", "end", "duration_months", "severity", "lowest"])


def collect_series(cache: dict | None = None, temp_col: str = M.TEMP_COL) -> dict:
    """{(district, T): MSDI-T Series indexed by (Year, Month)} for every fitted district."""
    cache = cache if cache is not None else M.load_model_cache(common.CACHE_DIR / "msdi_fits.pkl")
    out = {}
    for d in common.list_districts():
        m = cache.get(M.model_key(d, temp_col))
        if m is None:
            continue
        s = m.series.set_index(["Year", "Month"])
        for T in T_SCALES:
            out[d, T] = s[f"MSDI{T}"].dropna()
    return out


def share_table(series: dict) -> pd.DataFrame:
    """For every district and T, the share of months flagged dry or worse and extremely dry."""
    rows = []
    for (district, T), s in series.items():
        extreme, dry = TABLE4[T]
        rows.append({"district": district, "province": PROVINCE_OF[district], "T": T, "extreme cut-off": extreme,
                     "dry cut-off": dry, "months": len(s), "share dry or worse": round(float((s <= dry).mean()), 4),
                     "share extremely dry": round(float((s <= extreme).mean()), 4)})
    return pd.DataFrame(rows)


def save_shares(series: dict, out_dir: Path = OUT_DIR) -> pd.DataFrame:
    """Write msdi_outputs/common_threshold_shares.csv and return the table."""
    out_dir.mkdir(exist_ok=True)
    table = share_table(series)
    table.to_csv(out_dir / "common_threshold_shares.csv", index=False)
    return table


def main():
    """Recompute the shares from the saved fits, save them and print the mean per T."""
    table = save_shares(collect_series())
    print(table.groupby("T")[["share dry or worse", "share extremely dry"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
