"""
nbtools.py
==========
Plotting and display helpers for the three analysis notebooks. No drought
maths lives here: every number comes from the project modules
(spi_copula_analysis, ssi_analysis, msdi_analysis, index_data, compare_indices,
model_checks). This file only draws and formats, in the dashboard's colours
(theme.py).

Where it sits in the pipeline: project modules -> notebooks/*.ipynb -> this file
for the look of each plot and table.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# The notebooks live in notebooks/; the data and modules are one folder up
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib as mpl  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from IPython.display import Markdown, display  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402

import common  # noqa: E402
import theme  # noqa: E402
from spi_copula_analysis import spi_category  # noqa: E402


def setup():
    """Work from the project folder (relative data paths) and apply the dashboard look."""
    os.chdir(PROJECT_ROOT)
    # The copula optimisers try parameter values where a log is undefined and
    # recover on their own; hide those harmless warnings from the output
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 110, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": theme.BORDER, "axes.labelcolor": theme.INK, "axes.titlesize": 12.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": theme.INK,
        "axes.grid": True, "grid.color": theme.GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": theme.MUTED,
        "ytick.color": theme.MUTED, "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 10.5, "font.family": "sans-serif",
    })
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)


# ----------------------------------------------------------------------------- text

def caption(text: str):
    """One-line caption under a plot."""
    display(Markdown(f"<span style='color:{theme.MUTED}'>*{text}*</span>"))


def interpret(text: str):
    """Interpretation text built from this run's numbers (rendered as Markdown)."""
    display(Markdown(text))


def show(df: pd.DataFrame, digits: int = 3, n: int | None = None):
    """Display a table rounded to a few digits."""
    df = df.head(n) if n else df
    display(df.round(digits))


def fmt(x: float, digits: int = 2) -> str:
    """Number to text with a fixed number of decimals."""
    return f"{x:.{digits}f}"


# ----------------------------------------------------------------------------- time series

def class_colour(v: float) -> str:
    """Dashboard colour of the McKee class of a value."""
    return spi_category(v)[2] if pd.notna(v) else theme.CLASS_COLOURS["No data"]


def shade_dry(ax, dates: pd.Series, values: pd.Series, threshold: float = common.DRY_THRESHOLD):
    """Shade consecutive months at or below the dry threshold."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    dry = (pd.Series(values).reset_index(drop=True) <= threshold).to_numpy()
    edges = np.diff(np.r_[0, dry.astype(int), 0])
    for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1):
        ax.axvspan(d[s], d[e] + pd.offsets.MonthBegin(1), color=theme.DRY_SHADE, lw=0)


def shade_documented(ax, label: bool = True):
    """Shade the documented North Central Province drought years."""
    for i, (_, ys) in enumerate(common.DOCUMENTED_DROUGHTS):
        ax.axvspan(pd.Timestamp(ys[0], 1, 1), pd.Timestamp(ys[-1], 12, 31), color=theme.EVENT_SHADE, lw=0,
                   label="documented drought" if (label and i == 0) else None)


def index_bars(ax, dates, values, label: str, title: str):
    """Monthly bars coloured by drought class, dry spells shaded, class lines at -1, -1.5, -2."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    values = pd.Series(values).reset_index(drop=True)
    ax.bar(dates, values, width=31, color=[class_colour(v) for v in values], linewidth=0)
    shade_dry(ax, dates, values)
    for y, txt in ((-1.0, "moderately dry"), (-1.5, "severely dry"), (-2.0, "extremely dry")):
        ax.axhline(y, color=theme.DRY_LINE, ls=":", lw=0.9, alpha=0.7)
        ax.text(dates.iloc[0], y, f" {txt}", va="bottom", fontsize=8, color=theme.MUTED)
    ax.axhline(0, color="#9A9A9A", lw=0.6)
    ax.set_title(title)
    ax.set_xlabel("month")
    ax.set_ylabel(f"{label} (standard deviations)")
    handles = [Patch(color=theme.CLASS_COLOURS[c], label=c) for c in theme.CLASS_ORDER]
    handles.append(Patch(color=theme.DRY_SHADE, label="dry spell (index <= -1)"))
    ax.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8)


def msdi_bars(ax, dates, values, classes, extreme: float, dry: float, title: str, label: str):
    """MSDI bars coloured by class, with the two cut-offs and the reference droughts shaded."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    colours = [theme.CLASS_COLOURS[c] for c in pd.Series(classes)]
    ax.bar(dates, pd.Series(values), width=31, color=colours, linewidth=0)
    shade_documented(ax)
    ax.axhline(extreme, color=theme.DRY_LINE, ls="--", lw=1.8, label=f"extremely dry below {extreme:.2f}")
    ax.axhline(dry, color=theme.DRY_LINE, ls="--", lw=1, label=f"dry below {dry:.2f}")
    ax.set_title(title)
    ax.set_xlabel("month")
    ax.set_ylabel(f"{label} (index value)")
    handles, _ = ax.get_legend_handles_labels()
    handles += [Patch(color=theme.CLASS_COLOURS[c], label=c) for c in theme.MSDI_CLASS_ORDER]
    ax.legend(handles=handles, ncol=6, loc="lower right", fontsize=8)


def class_frequency(ax, values: pd.Series, title: str):
    """Bar chart of how many months fall in each McKee class."""
    labels = pd.Series([spi_category(v)[0] for v in values.dropna()])
    counts = labels.value_counts().reindex(theme.CLASS_ORDER).fillna(0).astype(int)
    bars = ax.barh(counts.index, counts.values, color=[theme.CLASS_COLOURS[c] for c in counts.index],
                   edgecolor="white")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_width() + 2, b.get_y() + b.get_height() / 2, f"{v} ({100 * v / counts.sum():.1f}%)",
                va="center", fontsize=8.5, color=theme.INK)
    ax.invert_yaxis()
    ax.set_xlim(0, counts.max() * 1.3)
    ax.set_title(title)
    ax.set_xlabel("number of months")
    ax.legend(handles=[Patch(color=theme.CLASS_COLOURS["Moderately dry"], label="dry classes are browns"),
                       Patch(color=theme.CLASS_COLOURS["Very wet"], label="wet classes are blues")],
              loc="lower right")
    return counts


def class_table() -> pd.DataFrame:
    """McKee classes with their range and the dashboard colour."""
    from spi_copula_analysis import SPI_CATEGORIES
    rows = []
    for lo, hi, label, colour in SPI_CATEGORIES:
        rng = (f">= {lo:g}" if np.isinf(hi) else f"<= {hi:g}" if np.isinf(lo) else f"{lo:g} to {hi:g}")
        rows.append({"class": label, "index range": rng, "colour": colour})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- distributions

def hist_with_fits(ax, x, frozen: dict, names: list[str], title: str, xlabel: str, colours=None):
    """Histogram (density) with the fitted pdfs of the named families."""
    x = np.asarray(x)
    colours = colours or [theme.INDEX_COLOURS["SPI"], theme.INDEX_COLOURS["SSI"], theme.INDEX_COLOURS["MSDI"],
                          theme.DRY_LINE]
    ax.hist(x, bins="auto", density=True, color="#E9E2D6", edgecolor="white", label="data")
    lo, hi = x.min(), x.max()
    grid = np.linspace(max(lo - 0.1 * (hi - lo), 1e-6 if lo >= 0 else lo - 0.1 * (hi - lo)), hi + 0.1 * (hi - lo), 300)
    for name, c in zip(names, colours):
        ax.plot(grid, frozen[name].pdf(grid), color=c, lw=2, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend()


def qq_plot(ax, x, dists: dict, title: str, unit: str):
    """Quantile-quantile plot: sample quantiles against each fitted family's quantiles."""
    x = np.sort(np.asarray(x))
    p = (np.arange(1, len(x) + 1) - 0.5) / len(x)
    colours = [theme.INDEX_COLOURS["SPI"], theme.INDEX_COLOURS["SSI"]]
    for (name, d), c in zip(dists.items(), colours):
        ax.scatter(d.ppf(p), x, s=14, color=c, alpha=0.8, label=name)
    ax.plot([x.min(), x.max()], [x.min(), x.max()], color="#9A9A9A", ls="--", lw=1, label="1:1 line")
    ax.set_title(title)
    ax.set_xlabel(f"fitted quantile ({unit})")
    ax.set_ylabel(f"observed quantile ({unit})")
    ax.legend()


def pp_plot(ax, x, dists: dict, title: str):
    """Probability-probability plot: empirical CDF against each fitted CDF."""
    x = np.sort(np.asarray(x))
    p = (np.arange(1, len(x) + 1) - 0.5) / len(x)
    colours = [theme.INDEX_COLOURS["SPI"], theme.INDEX_COLOURS["SSI"]]
    for (name, d), c in zip(dists.items(), colours):
        ax.scatter(d.cdf(x), p, s=14, color=c, alpha=0.8, label=name)
    ax.plot([0, 1], [0, 1], color="#9A9A9A", ls="--", lw=1, label="1:1 line")
    ax.set_title(title)
    ax.set_xlabel("fitted probability F(x)")
    ax.set_ylabel("empirical probability")
    ax.legend()


def normal_check(axes, values: pd.Series, label: str):
    """Histogram with the standard normal curve, and a normal Q-Q plot."""
    v = values.dropna().to_numpy()
    ax = axes[0]
    ax.hist(v, bins=30, density=True, color="#E9E2D6", edgecolor="white", label=label)
    g = np.linspace(-4, 4, 300)
    ax.plot(g, stats.norm.pdf(g), color=theme.INDEX_COLOURS["SPI"], lw=2, label="standard normal N(0, 1)")
    ax.set_title(f"{label}: histogram and N(0, 1)")
    ax.set_xlabel(f"{label} (standard deviations)")
    ax.set_ylabel("density")
    ax.legend()
    ax = axes[1]
    q = stats.norm.ppf((np.arange(1, len(v) + 1) - 0.5) / len(v))
    ax.scatter(q, np.sort(v), s=8, color=theme.INDEX_COLOURS["MSDI"], alpha=0.6, label=label)
    ax.plot([-4, 4], [-4, 4], color="#9A9A9A", ls="--", lw=1, label="1:1 line")
    ax.set_title(f"{label}: normal Q-Q plot")
    ax.set_xlabel("standard normal quantile")
    ax.set_ylabel(f"{label} quantile")
    ax.legend()


# ----------------------------------------------------------------------------- maps

_GEO = None


def districts_geo():
    """District polygons (GADM level 2 dissolved to 25 districts), loaded once."""
    global _GEO
    if _GEO is None:
        import geopandas as gpd
        g = gpd.read_file(common.GADM_SHP).dissolve(by="NAME_1", as_index=False)
        _GEO = g.rename(columns={"NAME_1": "District"})[["District", "geometry"]].to_crs(epsg=4326)
    return _GEO


def district_map(ax, values: pd.Series, title: str, categorical: bool = False, colours: dict | None = None,
                 cmap: str = "YlOrBr", label: str = ""):
    """Choropleth of one value per district (index = district name)."""
    g = districts_geo().merge(values.rename("val").rename_axis("District").reset_index(), on="District", how="left")
    if categorical:
        colours = colours or {}
        g.plot(ax=ax, color=[colours.get(v, "#D0D0D0") for v in g["val"]], edgecolor="white", linewidth=0.6)
        ax.legend(handles=[Patch(color=c, label=k) for k, c in colours.items() if k in set(g["val"])],
                  loc="lower left", fontsize=8)
    else:
        g.plot(ax=ax, column="val", cmap=cmap, edgecolor="white", linewidth=0.6, legend=True,
               legend_kwds={"shrink": 0.6, "label": label})
    for _, r in g.iterrows():
        pt = r.geometry.representative_point()
        ax.text(pt.x, pt.y, r["District"], fontsize=6.5, ha="center", color=theme.INK)
    ax.set_title(title)
    ax.set_axis_off()
