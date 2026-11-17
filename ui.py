"""
ui.py
=====
Shared look and reusable pieces for every page of the dashboard.

Where it sits in the pipeline:

    index_data.py (numbers)  ->  ui.py (theme, sidebar, maps, charts)  ->  page modules

One palette is defined here and used everywhere:
    * drought classes: dry = warm browns / oranges, near normal = light
      neutral, wet = blues (a colour-blind safe diverging scheme, the same
      colours on every map, bar and legend)
    * index lines: SPI blue, SSI vermilion, MSDI purple (checked for
      colour-blind separation and contrast on white)
Every chart goes through style_fig(), so titles, axes, fonts, grid and
legends look the same on every page, and through show_chart(), which adds a
one-line plain-language caption under it.
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import common
from spi_copula_analysis import spi_category

# ----------------------------------------------------------------------------- theme tokens
# Colours live in theme.py so the notebooks use exactly the same ones
from theme import (ACCENT, BG, BORDER, CARD, CLASS_COLOURS, CLASS_ORDER, DRY_LINE, GRID,  # noqa: E402,F401
                   INK, MSDI_CLASS_ORDER, MUTED, SEQUENTIAL, rgba_css)
from theme import DRY_SHADE as _DRY_SHADE, EVENT_SHADE as _EVENT_SHADE  # noqa: E402

# Quoted, because CSS rejects a whole font list with an unquoted name ending in a number ("Source Sans 3"),
# and Plotly then sizes hover boxes for one font and draws another, which cuts off the values.
# "Source Sans" is the font Streamlit loads for the page.
FONT = '"Source Sans", "Source Sans 3", "Source Sans Pro", sans-serif'
DRY_CLASSES = ("Moderately dry", "Severely dry", "Extremely dry")

# Continuous version of the same scale for index values from -2.5 to +2.5
# (position = (value + 2.5) / 5, so -2, -1.5, -1, 0, 1, 1.5, 2 land on the class colours)
DIVERGING = [(0.00, "#7A2E0B"), (0.10, "#7A2E0B"), (0.20, "#C4561D"), (0.30, "#F4A259"),
             (0.50, "#EDE8DF"), (0.70, "#6BAED6"), (0.80, "#2171B5"), (0.90, "#08306B"),
             (1.00, "#08306B")]
VALUE_RANGE = (-2.5, 2.5)

DRY_SHADE = rgba_css(_DRY_SHADE)            # background shading of dry months (index <= -1)
EVENT_SHADE = rgba_css(_EVENT_SHADE)        # documented drought years
# Same hue family without the near-white end, for marks drawn on a white surface
SEQUENTIAL_MARKS = [[0, "#F4C27A"], [0.5, "#D9772B"], [1, "#7A2E0B"]]

SRI_LANKA_CENTER = {"lat": 7.85, "lon": 80.72}


# ----------------------------------------------------------------------------- page chrome

def inject_css():
    """One stylesheet for the whole app: background, cards, tabs, metrics, legend chips, tiles."""
    st.markdown(f"""
    <style>
    .stApp {{ background: {BG}; }}
    .block-container {{ padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1480px; }}
    h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.01em; }}
    h2 {{ font-size: 1.4rem !important; }}
    h3 {{ font-size: 1.12rem !important; margin-top: .6rem; }}
    [data-testid="stSidebar"] {{ background: #FBF9F5; border-right: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; background: #EFE9DF; padding: 5px; border-radius: 10px; }}
    .stTabs [data-baseweb="tab"] {{ height: 40px; padding: 0 16px; font-size: 14.5px; font-weight: 600;
        background: transparent; color: {INK}; border-radius: 8px; }}
    .stTabs [aria-selected="true"] {{ background: {ACCENT} !important; color: white !important; }}
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
    div[data-testid="stMetric"] {{ background: {CARD}; border: 1px solid {BORDER};
        border-radius: 10px; padding: 10px 14px; }}
    div[data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-weight: 600; font-size: 13px; }}
    div[data-testid="stMetricValue"] {{ color: {INK} !important; font-size: 1.5rem; }}
    .stExpander {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; }}
    .kpi {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px;
        border-top: 4px solid var(--accent); height: 100%; }}
    .kpi .lab {{ color: {MUTED}; font-size: 12.5px; font-weight: 700; text-transform: uppercase;
        letter-spacing: .04em; }}
    .kpi .val {{ color: {INK}; font-size: 30px; font-weight: 700; line-height: 1.15; margin-top: 4px; }}
    .kpi .sub {{ color: {INK}; font-size: 13px; margin-top: 4px; }}
    .legend-row {{ display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; margin: 4px 0 8px; }}
    .legend-row span.sw {{ display: inline-block; width: 14px; height: 14px; border-radius: 3px;
        margin-right: 5px; vertical-align: -2px; border: 1px solid rgba(0,0,0,.15); }}
    .legend-row .item {{ font-size: 13px; color: {INK}; white-space: nowrap; }}
    .note {{ background: {CARD}; border: 1px solid {BORDER}; border-left: 4px solid #2171B5;
        border-radius: 8px; padding: 10px 14px; color: {INK}; font-size: 14px; margin: 4px 0 10px; }}
    .note.method {{ border-left-color: {ACCENT}; }}
    .tiles {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 8px; }}
    .tile {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 8px 10px; }}
    .tile .nm {{ font-weight: 700; font-size: 13.5px; color: {INK}; }}
    .tile .row {{ display: flex; gap: 5px; margin-top: 5px; }}
    .dp {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; padding: 16px 18px; }}
    .dp-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 10px;
        padding-bottom: 12px; border-bottom: 1px solid {BORDER}; margin-bottom: 4px; }}
    .dp-name {{ font-size: 24px; font-weight: 800; color: {INK}; line-height: 1.1; }}
    .dp-sub {{ color: {MUTED}; font-size: 13px; margin-top: 3px; }}
    .dp-status {{ border-radius: 999px; padding: 5px 12px; font-size: 13px; font-weight: 700; white-space: nowrap; }}
    .dp-row {{ padding: 12px 0 10px 12px; border-left: 4px solid var(--accent); margin-top: 12px; }}
    .dp-top {{ display: flex; align-items: center; gap: 10px; }}
    .dp-lab {{ font-weight: 700; color: {INK}; font-size: 15px; flex: 1; }}
    .dp-lab small {{ color: {MUTED}; font-weight: 600; margin-left: 4px; }}
    .dp-pill {{ border-radius: 999px; padding: 2px 10px; font-size: 12.5px; font-weight: 700;
        border: 1px solid rgba(0,0,0,.08); }}
    .dp-val {{ font-size: 22px; font-weight: 800; color: {INK}; min-width: 64px; text-align: right; }}
    .dp-text {{ color: {INK}; font-size: 13.5px; margin: 4px 0 8px; }}
    .dp-bar {{ position: relative; height: 10px; border-radius: 6px; border: 1px solid rgba(0,0,0,.08); }}
    .dp-mark {{ position: absolute; top: -5px; width: 4px; height: 20px; margin-left: -2px; background: {INK};
        border-radius: 2px; box-shadow: 0 0 0 2px #fff; }}
    .dp-ends {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: 11.5px; margin-top: 3px; }}
    .dp-cells {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 3px; margin-top: 10px; }}
    .dp-cells span {{ height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,.08); }}
    .tile .chip {{ flex: 1; text-align: center; border-radius: 6px; font-size: 11.5px; font-weight: 700;
        padding: 3px 0; border: 1px solid rgba(0,0,0,.08); white-space: nowrap; }}
    </style>""", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    """Large page title with a muted one-line subtitle."""
    st.markdown(
        f"<div style='margin:0 0 12px'><h1 style='margin:0;font-size:1.85rem'>{title}</h1>"
        + (f"<div style='color:{MUTED};font-size:15px;margin-top:4px'>{subtitle}</div>" if subtitle else "")
        + "</div>", unsafe_allow_html=True)


def kpi_cards(items: list[dict]):
    """A row of headline cards. items: dicts with label, value, sub (optional), accent (optional)."""
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(
            f"<div class='kpi' style='--accent:{it.get('accent', ACCENT)}'>"
            f"<div class='lab'>{it['label']}</div><div class='val'>{it['value']}</div>"
            f"<div class='sub'>{it.get('sub', '')}</div></div>", unsafe_allow_html=True)


def class_legend(order=None, colours=None):
    """Colour chips for the drought classes (the key for every class-coloured map and bar)."""
    order = order or CLASS_ORDER
    colours = colours or CLASS_COLOURS
    items = "".join(f"<span class='item'><span class='sw' style='background:{colours[c]}'></span>{c}</span>"
                    for c in order)
    st.markdown(f"<div class='legend-row'>{items}</div>", unsafe_allow_html=True)


def note(text: str, method: bool = False):
    """Highlighted note box. method=True uses the brown accent for method statements."""
    st.markdown(f"<div class='note{' method' if method else ''}'>{text}</div>", unsafe_allow_html=True)


def month_label(year: int, month: int) -> str:
    """'December 2025'."""
    return f"{calendar.month_name[month]} {year}"


def text_colour_on(hex_colour: str) -> str:
    """White or dark text, whichever reads better on this background colour."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return "#FFFFFF" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else INK


def class_of(v: float) -> str:
    """McKee class name of an index value ("No data" for NaN)."""
    return spi_category(v)[0]


# ----------------------------------------------------------------------------- sidebar

INDEX_LABELS = {"SPI": "Rainfall (SPI)", "SSI": "Soil moisture (SSI)", "MSDI": "Rainfall + heat (MSDI)"}


def sidebar_controls(districts: list[str]):
    """The four global controls: district, index, time window and date range.

    Drawn on every run so their values persist across pages. Read them with
    selection(). Index is used by the Overview map; each index page shows its
    own index.
    """
    st.sidebar.selectbox("District", districts, index=districts.index("Anuradhapura"), key="g_district")
    st.sidebar.selectbox("Index", list(INDEX_LABELS), format_func=INDEX_LABELS.get, key="g_index",
                         help="Used on the Overview map. Each index page shows its own index.")
    st.sidebar.segmented_control("Time window", list(common.SCALES), default=3, key="g_scale",
                                 format_func=lambda k: f"{k} month" + ("s" if k > 1 else ""),
                                 help="1 month: this month alone. 3 months: one cultivation season. "
                                      "6 months: the last half-year.")
    st.sidebar.slider("Date range", common.START_YEAR, common.END_YEAR,
                      (common.START_YEAR, common.END_YEAR), key="g_years",
                      help="Years shown in the time-series charts.")


def selection() -> dict:
    """Current sidebar values: district, index, scale, years (start, end)."""
    ss = st.session_state
    return {"district": ss.get("g_district", "Anuradhapura"), "index": ss.get("g_index", "SPI"),
            "scale": ss.get("g_scale") or 3,
            "years": ss.get("g_years", (common.START_YEAR, common.END_YEAR))}


# ----------------------------------------------------------------------------- chart styling

def style_fig(fig, height=380, legend=True, x_title=None, y_title=None):
    """Apply the shared chart style: white surface, recessive grid, title top-left, legend top-right.

    x_title / y_title set the axis titles (always include units).
    """
    has_title = bool(fig.layout.title.text)
    fig.update_layout(height=height, plot_bgcolor=CARD, paper_bgcolor=CARD,
                      font=dict(family=FONT, color=INK, size=13),
                      margin=dict(l=10, r=10, t=60 if has_title else 34, b=10), showlegend=legend,
                      hoverlabel=dict(bgcolor="white", font_size=13, font_family=FONT),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                                  bgcolor="rgba(255,255,255,0)", font=dict(size=12)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=BORDER)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=BORDER)
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)
    if has_title:
        fig.update_layout(title=dict(x=0.01, xanchor="left", y=0.98, yanchor="top",
                                     font=dict(size=15, color=INK)))
    return fig


def show_chart(fig, caption: str = "", key: str | None = None):
    """Render a Plotly chart with the shared toolbar settings and a plain-language caption."""
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo": False})
    if caption:
        st.caption(caption)


# ----------------------------------------------------------------------------- GIS map

@st.cache_data(show_spinner=False)
def district_geojson() -> tuple[dict, pd.DataFrame]:
    """GADM level-2 polygons dissolved to the 25 districts, lightly simplified.

    Returns the GeoJSON (feature id = district name) and a table of label points.
    Simplifying to 0.003 degrees (about 300 m) keeps the map fast to draw.
    """
    import geopandas as gpd
    gdf = gpd.read_file(common.GADM_SHP).dissolve(by="NAME_1", as_index=False)
    gdf = gdf.rename(columns={"NAME_1": "District"})[["District", "geometry"]].to_crs(epsg=4326)
    gdf["geometry"] = gdf.geometry.simplify(0.003, preserve_topology=True)
    pts = gdf.geometry.representative_point()
    labels = pd.DataFrame({"District": gdf["District"], "lat": pts.y, "lon": pts.x})
    gj = gdf.set_index("District").__geo_interface__
    return gj, labels


def district_map(df: pd.DataFrame, value_col: str = "value", class_col: str = "class",
                 colour_by: str = "class", class_order=None, class_colours=None,
                 value_range=VALUE_RANGE, value_label: str = "Index", height: int = 560,
                 labels: bool = True, zoom: float = 6.35, title: str | None = None,
                 discrete_title: str = "Class", continuous_scale=None):
    """Interactive GIS map of the 25 districts on a light basemap.

    df needs a District column, plus value_col and/or class_col. Districts not
    in df are drawn grey ("No data"). colour_by = "class" (discrete drought
    classes) or "value" (continuous scale, diverging by default).
    """
    gj, lab = district_geojson()
    class_order = class_order or CLASS_ORDER
    class_colours = class_colours or CLASS_COLOURS
    d = lab[["District"]].merge(df, on="District", how="left")
    if class_col in d:
        d[class_col] = d[class_col].fillna("No data")
    hover = {"District": False}
    if value_col in d:
        hover[value_col] = ":.2f"
    common_args = dict(geojson=gj, locations="District", hover_name="District", hover_data=hover,
                       map_style="carto-positron", center=SRI_LANKA_CENTER, zoom=zoom, opacity=0.88)
    if colour_by == "class":
        fig = px.choropleth_map(d, color=class_col,
                                category_orders={class_col: [c for c in class_order + ["No data"]
                                                             if c in set(d[class_col])]},
                                color_discrete_map=class_colours,
                                labels={class_col: discrete_title, value_col: value_label}, **common_args)
    else:
        fig = px.choropleth_map(d, color=value_col, color_continuous_scale=continuous_scale or DIVERGING,
                                range_color=value_range, labels={value_col: value_label}, **common_args)
        fig.update_layout(coloraxis_colorbar=dict(title=value_label, thickness=12, len=0.6))
    fig.update_traces(marker_line_width=0.8, marker_line_color="white")
    if labels:
        fig.add_trace(go.Scattermap(lat=lab["lat"], lon=lab["lon"], mode="text", text=lab["District"],
                                    textfont=dict(size=10, color="#1F1A14"), hoverinfo="skip",
                                    showlegend=False))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=38 if title else 0, b=0),
                      paper_bgcolor=CARD, font=dict(family=FONT, color=INK),
                      legend=dict(title=None, bgcolor="rgba(255,255,255,.88)", x=0.01, y=0.99,
                                  font=dict(size=12)))
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=15, color=INK)))
    return fig


def show_map(fig, key: str | None = None, caption: str = ""):
    """Render a map with scroll-zoom enabled, plus an optional caption."""
    st.plotly_chart(fig, use_container_width=True, key=key,
                    config={"scrollZoom": True, "displaylogo": False})
    if caption:
        st.caption(caption)


# ----------------------------------------------------------------------------- time series

def dry_runs(dates: pd.Series, values: pd.Series, threshold: float = common.DRY_THRESHOLD, dry=None):
    """(start, end) date pairs of consecutive dry months.

    dry: optional boolean Series, for an index with its own cut-offs. Without
    it, a month is dry at or below threshold.
    """
    dry = (values <= threshold).to_numpy() if dry is None else np.asarray(dry, dtype=bool)
    edges = np.diff(np.r_[0, dry.astype(int), 0])
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    # Each box covers whole months: from the 1st of the first dry month to the
    # 1st of the month after the last one
    return [(d[s], d[e] + pd.offsets.MonthBegin(1)) for s, e in zip(starts, ends)]


def _shade_runs(fig, runs, row=None, col=None):
    """Light brown background boxes behind dry spells."""
    for x0, x1 in runs:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=DRY_SHADE, line_width=0, layer="below", row=row, col=col)


def _shade_documented(fig, years_range=None):
    """Background boxes for the documented North Central droughts."""
    for _, ys in common.DOCUMENTED_DROUGHTS:
        if years_range and (ys[-1] < years_range[0] or ys[0] > years_range[1]):
            continue
        fig.add_vrect(x0=f"{ys[0]}-01-01", x1=f"{ys[-1]}-12-31", fillcolor=EVENT_SHADE, line_width=0,
                      layer="below")


def _in_years(s: pd.DataFrame, years) -> pd.DataFrame:
    """Rows of a Date/value table inside the (start, end) year range."""
    if not years:
        return s
    return s[(s["Date"].dt.year >= years[0]) & (s["Date"].dt.year <= years[1])]


def index_bars(s: pd.DataFrame, label: str, title: str | None = None, height=360, years=None):
    """Monthly bars coloured by drought class, with dry spells shaded and the class lines.

    s: Date and value columns. label: e.g. "SPI-3". years: optional (start, end) filter.
    """
    s = _in_years(s.dropna(subset=["value"]), years)
    cls = s["value"].map(class_of)
    fig = go.Figure(go.Bar(x=s["Date"], y=s["value"], name=label, marker_line_width=0,
                           marker_color=[CLASS_COLOURS[c] for c in cls], customdata=cls,
                           hovertemplate="%{x|%b %Y}: %{y:.2f} (%{customdata})<extra></extra>"))
    _shade_runs(fig, dry_runs(s["Date"], s["value"]))
    for y, txt in ((-1.0, "moderately dry"), (-1.5, "severely dry"), (-2.0, "extremely dry")):
        fig.add_hline(y=y, line_dash="dot", line_color=DRY_LINE, opacity=0.55, line_width=1,
                      annotation_text=txt, annotation_position="bottom left", annotation_font_size=10,
                      annotation_font_color=MUTED)
    fig.update_layout(title=title or f"{label} by month", bargap=0)
    return style_fig(fig, height, legend=False, x_title="Year",
                     y_title=f"{label} (standard deviations)")


def index_timeseries(series: dict[str, pd.DataFrame], colours: dict[str, str], height=380,
                     title=None, years=None, documented=True):
    """Several indices as lines on one axis, documented droughts shaded.

    series: name -> DataFrame with Date and value.
    """
    fig = go.Figure()
    for name, s in series.items():
        s = _in_years(s, years)
        fig.add_trace(go.Scatter(x=s["Date"], y=s["value"], name=name, mode="lines",
                                 line=dict(color=colours.get(name), width=1.6),
                                 hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>"))
    if documented:
        _shade_documented(fig, years)
    fig.add_hline(y=-1, line_dash="dot", line_color=DRY_LINE, opacity=0.7)
    fig.add_hline(y=0, line_color="#9A9A9A", line_width=0.6)
    fig.update_layout(title=title, hovermode="x unified")
    return style_fig(fig, height, x_title="Year", y_title="index (standard deviations)")


def three_index_panel(series: dict[str, pd.DataFrame], years=None, label_suffix: str = "", height=640):
    """One row per index on a shared time axis, bars coloured by class and dry spells shaded.

    The stacked layout makes it easy to see which index flags a drought first
    and which one holds it longest.
    """
    names = list(series)
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=[f"{n}{label_suffix}" for n in names])
    for r, name in enumerate(names, 1):
        s = _in_years(series[name].dropna(subset=["value"]), years)
        # MSDI brings its own classes and dry flag; the others use the McKee classes
        cls = s["class"].fillna("No data") if "class" in s else s["value"].map(class_of)
        fig.add_trace(go.Bar(x=s["Date"], y=s["value"], name=name, marker_line_width=0,
                             marker_color=[CLASS_COLOURS[c] for c in cls], customdata=cls,
                             hovertemplate=f"{name} %{{x|%b %Y}}: %{{y:.2f}} (%{{customdata}})<extra></extra>"),
                      row=r, col=1)
        _shade_runs(fig, dry_runs(s["Date"], s["value"], dry=s["dry"] if "dry" in s else None), row=r, col=1)
        line = s["cut_dry"].iloc[0] if "cut_dry" in s and len(s) else -1
        if pd.notna(line):
            fig.add_hline(y=line, line_dash="dot", line_color=DRY_LINE, opacity=0.6, row=r, col=1)
        fig.update_yaxes(title_text="index", range=[-3.8, 3.8], row=r, col=1)
    fig.update_layout(bargap=0, title="The three indices month by month (shaded = dry spell, dotted = dry line)")
    fig.update_xaxes(title_text="Year", row=len(names), col=1)
    for a in fig.layout.annotations:                     # subplot titles in the shared style
        a.update(x=0.01, xanchor="left", font=dict(size=13, color=MUTED))
    return style_fig(fig, height, legend=False)


# ----------------------------------------------------------------------------- national charts

def drought_extent_chart(tables: dict[str, pd.DataFrame], colours: dict[str, str],
                         years: tuple[int, int], height=360, smooth: int = 1):
    """Share of the 25 districts in drought, one line per index.

    tables: index -> long table with a dry column. smooth > 1 shows a centred
    running mean over that many months.
    """
    fig = go.Figure()
    for name, t in tables.items():
        t = t[(t["Year"] >= years[0]) & (t["Year"] <= years[1])]
        share = t.groupby("Date")["dry"].mean() * 100
        if smooth > 1:
            share = share.rolling(smooth, center=True, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=share.index, y=share.values, name=name, mode="lines",
                                 line=dict(color=colours.get(name), width=2),
                                 hovertemplate=f"{name}: %{{y:.0f}}% of districts<extra></extra>"))
    _shade_documented(fig, years)
    fig.update_layout(title="Share of Sri Lanka's districts in drought", yaxis_range=[0, 100],
                      hovermode="x unified")
    return style_fig(fig, height, x_title="Year", y_title="districts in drought (%)")


def drought_calendar(t: pd.DataFrame, title: str, years=None, height=640):
    """District x year heat map: number of months in drought (the table's dry column)."""
    if years:
        t = t[(t["Year"] >= years[0]) & (t["Year"] <= years[1])]
    m = t.groupby(["District", "Year"])["dry"].sum().unstack("Year").sort_index(ascending=False)
    fig = go.Figure(go.Heatmap(z=m.values, x=m.columns, y=m.index, colorscale=SEQUENTIAL, zmin=0, zmax=12,
                               colorbar=dict(title="dry<br>months", thickness=12),
                               hovertemplate="%{y}, %{x}: %{z} dry months<extra></extra>",
                               xgap=1, ygap=1))
    fig.update_layout(title=title, xaxis=dict(dtick=2, tickangle=-90))
    return style_fig(fig, height, legend=False, x_title="Year", y_title="")


def status_tiles(snaps: dict[str, pd.DataFrame]):
    """One small tile per district with a coloured chip for each index's value and class.

    snaps: index name -> one month's table (District, value, class, dry).
    Districts are ordered by how many indices put them in drought, then by the
    lowest value.
    """
    names = list(snaps)
    table = pd.DataFrame({n: snaps[n].set_index("District")["value"] for n in names})
    classes = {n: snaps[n].set_index("District")["class"] for n in names}
    votes = pd.DataFrame({n: snaps[n].set_index("District")["dry"] for n in names}).fillna(False).astype(bool).sum(axis=1)
    order = table.assign(v=votes, m=table.min(axis=1)).sort_values(["v", "m"], ascending=[False, True]).index
    tiles = []
    for d in order:
        chips = []
        for n in names:
            v = table.at[d, n]
            name = classes[n].get(d)
            name = name if isinstance(name, str) else "No data"
            c = CLASS_COLOURS[name]
            txt = f"{n} n/a" if pd.isna(v) else f"{n} {v:+.1f}"
            chips.append(f"<span class='chip' style='background:{c};color:{text_colour_on(c)}' "
                         f"title='{n}: {name}'>{txt}</span>")
        tiles.append(f"<div class='tile'><div class='nm'>{d}</div><div class='row'>{''.join(chips)}</div></div>")
    st.markdown(f"<div class='tiles'>{''.join(tiles)}</div>", unsafe_allow_html=True)
