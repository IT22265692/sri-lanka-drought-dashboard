"""
compare_indices.py
==================
District-by-district comparison of SPI, SSI and MSDI, 1984-2025, all 25
districts, accumulation scales 1, 3 and 6 months.

Where it sits in the pipeline:

    index_data.spi_series / ssi_series / msdi_series   (the three indices)
        -> compare_indices.py                          (this module)
        -> comparison_outputs/index_comparison_summary.csv
        -> the Compare page of the dashboard (uses the same functions)

For each district and scale:
    * Pearson and Spearman correlation between each pair of indices
    * lag check: correlation of A(t) with B(t + lag), lag = 0..3 months, and
      the lag with the highest correlation
    * dry months per index, and how many months both / only one flag
    * run-theory drought events per index: number, mean duration, mean severity,
      longest duration

SPI and SSI are standardised: dry means <= -1 and an event is a run below 0. The
published MSDI is not, so both use the paper's Table 4 dry cut-off (thresholds.py).
The paper sets no cut-off for the one-month MSDI, so at scale 1 its dry and event
statistics are empty.

    py -3.12 compare_indices.py

Writes
    comparison_outputs/index_comparison_summary.csv   one row per district and scale
    comparison_outputs/anuradhapura_3month.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import common
import index_data
import thresholds
from ssi_analysis import extract_drought_events as run_theory_events

OUT_DIR = Path("comparison_outputs")

START_YEAR, END_YEAR = common.START_YEAR, common.END_YEAR
SCALES = common.SCALES
DRY = common.DRY_THRESHOLD                 # SPI and SSI: a month is "dry" when the index is <= -1
EVENT_THRESHOLD = common.EVENT_THRESHOLD   # SPI and SSI: an event is a continuous run below 0
LAGS = (0, 1, 2, 3)                        # second index shifted later by this many months



# =============================================================================
# 1. INDEX SERIES
#    Each loader returns a Series indexed by (Year, Month) for one district
#    and scale.
# =============================================================================

def _as_series(df: pd.DataFrame | None, name: str, district: str, scale: int) -> pd.Series:
    """Year/Month/value table -> Series indexed by (Year, Month)."""
    if df is None:
        raise FileNotFoundError(f"{name}-{scale} not available for {district}: run "
                                f"py -3.12 run_msdi_all_districts.py")
    return df.set_index(["Year", "Month"])["value"]


def load_spi(district: str, scale: int) -> pd.Series:
    """SPI-k for one district, indexed by (Year, Month)."""
    return _as_series(index_data.spi_series(district, scale), "SPI", district, scale)


def load_ssi(district: str, scale: int) -> pd.Series:
    """SSI-k for one district, indexed by (Year, Month)."""
    return _as_series(index_data.ssi_series(district, scale), "SSI", district, scale)


def load_msdi(district: str, scale: int) -> pd.Series:
    """MSDI-T for one district from cache/msdi_fits.pkl, indexed by (Year, Month)."""
    return _as_series(index_data.msdi_series(district, scale), "MSDI", district, scale)


INDICES = {"SPI": load_spi, "SSI": load_ssi, "MSDI": load_msdi}
PAIRS = [("SPI", "SSI"), ("SPI", "MSDI"), ("SSI", "MSDI")]


def load_indices(district: str, scale: int) -> pd.DataFrame:
    """All three indices for one district and scale, one column each, 1984-2025."""
    df = pd.concat({name: f(district, scale) for name, f in INDICES.items()}, axis=1).sort_index()
    years = df.index.get_level_values("Year")
    return df[(years >= START_YEAR) & (years <= END_YEAR)]


# =============================================================================
# 2. STATISTICS
# =============================================================================

def dry_cutoff(scale: int) -> float | None:
    """The MSDI dry cut-off for one scale, or None at scale 1, where the paper sets none."""
    cut = thresholds.cutoff(scale)
    return None if cut is None else cut[1]


def dry_mask(s: pd.Series, name: str, msdi_cut: float | None) -> pd.Series | None:
    """Booleans: dry months of one index. None for MSDI without a cut-off."""
    if name != "MSDI":
        return s <= DRY
    return None if msdi_cut is None else s <= msdi_cut


def pair_stats(df: pd.DataFrame, a: str, b: str, msdi_cut: float | None = None) -> dict:
    """Correlations, lag check and dry-month overlap for one pair of indices.

    df: one column per index, indexed by (Year, Month). msdi_cut: the MSDI dry
    cut-off, if MSDI is in the pair. Returns a flat dict whose keys name the
    pair, e.g. "pearson SPI-SSI".
    """
    both = df[[a, b]].dropna()
    out = {f"pearson {a}-{b}": stats.pearsonr(both[a], both[b]).statistic,
           f"spearman {a}-{b}": stats.spearmanr(both[a], both[b]).statistic}

    # Lag: b shifted later by k months, i.e. corr(a(t), b(t + k))
    lag_r = {}
    for k in LAGS:
        shifted = pd.concat([df[a], df[b].shift(-k)], axis=1).dropna()
        lag_r[k] = stats.pearsonr(shifted.iloc[:, 0], shifted.iloc[:, 1]).statistic
        out[f"r {a}-{b} lag {k}"] = lag_r[k]
    best = max(lag_r, key=lag_r.get)
    out[f"best lag {a}-{b} (months)"] = best
    out[f"best lag r {a}-{b}"] = lag_r[best]

    # Dry-month overlap on the months where both indices exist
    da, db = dry_mask(both[a], a, msdi_cut), dry_mask(both[b], b, msdi_cut)
    out[f"common months {a}-{b}"] = len(both)
    # "dry only X" is kept for the saved CSV, but each pair overwrites it, so it holds the last pair's count.
    # Read "dry only X in A-B" instead: that one belongs to this pair.
    if da is None or db is None:
        return out | {f"dry both {a}-{b}": np.nan, f"dry only {a}": np.nan, f"dry only {b}": np.nan,
                      f"dry only {a} in {a}-{b}": np.nan, f"dry only {b} in {a}-{b}": np.nan}
    out[f"dry both {a}-{b}"] = int((da & db).sum())
    out[f"dry only {a}"] = out[f"dry only {a} in {a}-{b}"] = int((da & ~db).sum())
    out[f"dry only {b}"] = out[f"dry only {b} in {a}-{b}"] = int((~da & db).sum())
    return out


def drought_events(s: pd.Series, threshold: float = EVENT_THRESHOLD) -> pd.DataFrame:
    """Run theory: continuous runs below threshold. Severity = sum of the distance below it.

    Uses the run-theory function of ssi_analysis (same rule as SPI, severity
    not rounded). Returns start, duration (months) and severity.
    """
    ev = run_theory_events(s - threshold, threshold=0.0)
    return pd.DataFrame({"start": ev["start"], "duration": ev["duration_months"],
                         "severity": ev["severity"]})


def index_stats(df: pd.DataFrame, name: str, msdi_cut: float | None = None) -> dict:
    """Dry months and drought-event statistics for one index."""
    s = df[name].dropna()
    dry = dry_mask(s, name, msdi_cut)
    if dry is None:
        keys = ("dry months", "% dry months", "events", "mean duration", "mean severity", "longest duration")
        return {f"months {name}": len(s), **{f"{k} {name}": np.nan for k in keys}}
    ev = drought_events(s, msdi_cut if name == "MSDI" else EVENT_THRESHOLD)
    return {f"months {name}": len(s),
            f"dry months {name}": int(dry.sum()),
            f"% dry months {name}": 100 * float(dry.mean()),
            f"events {name}": len(ev),
            f"mean duration {name}": ev["duration"].mean(),
            f"mean severity {name}": ev["severity"].mean(),
            f"longest duration {name}": int(ev["duration"].max()) if len(ev) else 0}


def district_scale_row(district: str, scale: int, df: pd.DataFrame) -> dict:
    """Every statistic for one district and scale, as one flat row."""
    cut = dry_cutoff(scale)
    row = {"district": district, "scale": scale, "MSDI dry cut-off": cut}
    for a, b in PAIRS:
        row.update(pair_stats(df, a, b, cut))
    for name in INDICES:
        row.update(index_stats(df, name, cut))
    return row


# =============================================================================
# 3. FIGURE: one district and scale (static PNG for the thesis)
# =============================================================================

def plot_district(district: str, scale: int, df: pd.DataFrame, row: dict, path: Path):
    """Time series, SPI-SSI scatter, lag bars and dry-month overlap in one PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {k: v["colour"] for k, v in index_data.INDEX_INFO.items()}
    t = pd.to_datetime(pd.DataFrame({"year": df.index.get_level_values("Year"),
                                     "month": df.index.get_level_values("Month"), "day": 1}))
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
    _panel_timeseries(fig.add_subplot(gs[0, :]), t, df, district, scale, colours)
    _panel_pair(fig, gs, df, row, colours)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _panel_timeseries(ax, t, df, district, scale, colours):
    """Top panel: the three indices over time, documented droughts shaded."""
    for _, years in common.DOCUMENTED_DROUGHTS:
        ax.axvspan(pd.Timestamp(years[0], 1, 1), pd.Timestamp(years[-1], 12, 31),
                   color="#F3E3B5", alpha=0.6, lw=0)
    for name in INDICES:
        ax.plot(t, df[name].values, color=colours.get(name, "k"), lw=1, label=f"{name}-{scale}")
    ax.axhline(DRY, color="#8C2D04", ls="--", lw=0.8, label="SPI, SSI dry (-1)")
    cut = dry_cutoff(scale)
    if cut is not None:
        ax.axhline(cut, color=colours["MSDI"], ls="--", lw=0.8, label=f"MSDI dry ({cut:g})")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylim(-4, 4)
    ax.set_ylabel("index value")
    ax.legend(ncol=5, fontsize=9, loc="upper left")
    ax.set_title(f"{district}, {scale}-month indices, {START_YEAR}-{END_YEAR} "
                 f"(shaded = documented North Central droughts)")


def _panel_pair(fig, gs, df, row, colours):
    """Bottom panels for SPI vs SSI: scatter, lag check and dry-month overlap."""
    # Bottom left: SPI vs SSI scatter
    a, b = PAIRS[0]
    both = df[[a, b]].dropna()
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(both[a], both[b], s=8, alpha=0.5, color="#5D6D7E")
    ax.axvline(DRY, color="#8C2D04", ls="--", lw=0.7)
    ax.axhline(DRY, color="#8C2D04", ls="--", lw=0.7)
    ax.plot([-4, 4], [-4, 4], "k:", lw=0.7)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_title(f"Pearson r = {row[f'pearson {a}-{b}']:.2f}, "
                 f"Spearman = {row[f'spearman {a}-{b}']:.2f}")

    # Bottom middle: lag check
    ax = fig.add_subplot(gs[1, 1])
    r = [row[f"r {a}-{b} lag {k}"] for k in LAGS]
    bars = ax.bar([str(k) for k in LAGS], r, color="#AAB7B8")
    bars[LAGS.index(row[f"best lag {a}-{b} (months)"])].set_color(colours[b])
    for k, v in zip(LAGS, r):
        ax.text(LAGS.index(k), v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"{b} shifted later by (months)")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Lag check: best lag = {row[f'best lag {a}-{b} (months)']} month(s)")

    # Bottom right: dry-month overlap plus event and validation numbers
    ax = fig.add_subplot(gs[1, 2])
    labels = ["both", f"only {a}", f"only {b}"]
    vals = [row[f"dry both {a}-{b}"], row[f"dry only {a} in {a}-{b}"], row[f"dry only {b} in {a}-{b}"]]
    ax.bar(labels, vals, color=["#7B241C", colours[a], colours[b]])
    for i, v in enumerate(vals):
        ax.text(i, v + 1, str(v), ha="center")
    txt = "\n".join(f"{n}: {row[f'events {n}']} events, mean {row[f'mean duration {n}']:.1f} mo, "
                    f"longest {row[f'longest duration {n}']} mo" for n in INDICES)
    if f"hit rate {a}" in row:
        txt += "\n" + "\n".join(f"{n}: hit rate {row[f'hit rate {n}']:.2f}, "
                                f"false alarms {row[f'false-alarm rate {n}']:.2f}" for n in INDICES)
    ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="#CCCCCC"))
    ax.set_ylabel("months")
    ax.set_title(f"Dry months (index <= {DRY:g}) of {row[f'common months {a}-{b}']}")


# =============================================================================
# 4. RUN
# =============================================================================

def _print_overview(summary: pd.DataFrame):
    """Console summary: means over districts and best lags."""
    pd.set_option("display.width", 220)
    print("\nAcross all 25 districts (mean over districts):")
    cols = ["pearson SPI-SSI", "spearman SPI-SSI", "best lag r SPI-SSI", "% dry months SPI",
            "% dry months SSI", "events SPI", "events SSI", "mean duration SPI", "mean duration SSI",
            "longest duration SPI", "longest duration SSI"]
    print(summary.groupby("scale")[cols].mean().round(2).T.to_string())
    print("\nBest lag (months) per scale, number of districts:")
    print(summary.groupby("scale")["best lag SPI-SSI (months)"].value_counts()
          .unstack(fill_value=0).to_string())
    print("(MSDI at scale 1 is empty: the paper sets no cut-off for the one-month MSDI.)")


def main():
    """Compute the comparison for every district and scale and write the outputs."""
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for d in common.list_districts():
        for k in SCALES:
            df = load_indices(d, k)
            row = district_scale_row(d, k, df)
            rows.append(row)
            if d == "Anuradhapura" and k == 3:
                plot_district(d, k, df, row, OUT_DIR / "anuradhapura_3month.png")
        print(f"  {d} done")
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "index_comparison_summary.csv", index=False)
    _print_overview(summary)
    print(f"\nWrote {OUT_DIR / 'index_comparison_summary.csv'} ({len(summary)} rows) "
          f"and {OUT_DIR / 'anuradhapura_3month.png'}")


if __name__ == "__main__":
    main()
