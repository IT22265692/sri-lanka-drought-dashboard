"""
msdi_tabs.py
============
Streamlit page for the combined rainfall + temperature drought index: the
published MSDI of Karunarathne et al. (2024), laid out like the SSI page.

Where it sits in the pipeline:

    run_msdi_all_districts.py -> cache/msdi_fits.pkl -> msdi_tabs (this module)
    thresholds.py (the paper's Table 4)              -> msdi_tabs
                                                     -> dashboard.py "Rainfall + heat (MSDI)"

A control at the top of the page chooses the prolonged period T. The district and
the years come from the sidebar. A model missing from the cache is fitted on demand
with a smaller bootstrap and added to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import common
import index_data as D
import msdi_analysis as M
import pages_extra as PX
import thresholds as TH
import ui

CACHE_FILE = common.CACHE_DIR / "msdi_fits.pkl"
ON_DEMAND_BOOT = 200     # smaller bootstrap when a model has to be fitted inside the app
FLAG_NOTE_ABOVE = 60     # above this share of months, a data note appears under the tiles
TOP_EVENTS = 15


# ---------------------------------------------------------------- data access

@st.cache_resource(show_spinner="Loading MSDI fits...")
def _load_cache_version(mtime: float) -> dict:
    """The model cache for one file version (reloaded when the file changes)."""
    return M.load_model_cache(CACHE_FILE)


def _load_cache() -> dict:
    """Cached by file modification time, so a batch re-run is picked up."""
    mtime = CACHE_FILE.stat().st_mtime if CACHE_FILE.exists() else 0.0
    return _load_cache_version(mtime)


def get_model(district: str) -> M.MSDIModel:
    """Fitted MSDI model for one district, fitting and caching it if missing."""
    cache = _load_cache()
    key = M.model_key(district)
    if key in cache:
        return cache[key]
    with st.spinner(f"Fitting MSDI for {district}. A few seconds, then cached."):
        model = M.fit_msdi_model(M.load_inputs(district), district, n_boot=ON_DEMAND_BOOT)
    # Merge into the file on disk rather than overwriting it with this
    # session's copy, which may be older than the file
    M.update_model_cache(CACHE_FILE, {key: model})
    return model


def _district_table(index: str, scale: int, district: str) -> pd.DataFrame:
    """One index and window for one district, with its dates, values and classes."""
    t = PX.table(index, scale)
    return t[t["District"] == district].dropna(subset=["value"])


def _in_years(s: pd.DataFrame, years) -> pd.DataFrame:
    """Rows of a table inside the sidebar's (start, end) years."""
    return s[(s["Year"] >= years[0]) & (s["Year"] <= years[1])]


# ---------------------------------------------------------------- page entry

def _controls() -> int:
    """The prolonged period T."""
    return st.segmented_control("Prolonged period T", list(M.T_SCALES), default=3, key="msdi_T",
                                format_func=lambda k: f"{k} months") or 3


def render_msdi_layer(districts: list[str]):
    """The MSDI page: the T control, the cut-off sentence, then the tabs."""
    sel = ui.selection()
    district = sel["district"]
    ui.page_header("Rainfall + heat drought (MSDI)",
                   "Multivariate Standardized Drought Index from rainfall and ground temperature (TS). "
                   f"District and years come from the sidebar ({district}).")
    T = _controls()
    extreme, dry = TH.cutoff(T)
    st.markdown(f"For T = {T}, a month is Dry at or below {dry} and Extremely dry at or below {extreme}. These are the "
                "paper's Table 4 values, the same for every district.")
    t_district, t_map = st.tabs(["District analysis", "National map"])
    with t_district:
        _tab_district(district, T, sel["years"])
    with t_map:
        PX.national_map_tab("MSDI", "msdi_nat", scale=T)


# ---------------------------------------------------------------- district tab

def _tiles(model, s: pd.DataFrame, events: pd.DataFrame, T: int):
    """Four tiles: drought months, longest event, Kendall's tau and the selected copula."""
    months, share = int(s["dry"].sum()), 100 * float(s["dry"].mean())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"MSDI-{T} drought months", f"{months} ({share:.1f}%)",
              help="Months at or below the dry cut-off, and their share of all months.")
    c2.metric("Longest drought event", f"{events['duration_months'].max()} months" if len(events) else "none",
              help="The longest unbroken run of months at or below the dry cut-off.")
    c3.metric("Kendall's tau", f"{model.kendall_tau:+.3f}",
              help="Rank correlation of rainfall and temperature, with months paired. Negative means hot months tend "
                   "to be dry months.")
    c4.metric("Selected copula", model.best_family,
              help="The copula family that joins rainfall and temperature. It has the lowest AIC of the families tried.")
    if share > FLAG_NOTE_ABOVE:
        ui.note(f"These cut-offs flag {share:.0f}% of months.")


def history_chart(district: str, T: int, years):
    """MSDI-T bars by class, with the two cut-off lines and the dry spells shaded."""
    whole = _district_table("MSDI", T, district)
    s = _in_years(whole, years)
    extreme, dry = TH.cutoff(T)
    fig = go.Figure(go.Bar(x=s["Date"], y=s["value"], name=f"MSDI-{T}", marker_line_width=0,
                           marker_color=[ui.CLASS_COLOURS[c] for c in s["class"]], customdata=s["class"],
                           hovertemplate="%{x|%b %Y}: %{y:.2f} (%{customdata})<extra></extra>"))
    ui._shade_runs(fig, ui.dry_runs(s["Date"], s["value"], dry=s["dry"]))
    for y, label in ((dry, "dry"), (extreme, "extremely dry")):
        fig.add_hline(y=y, line_dash="dash", line_color=ui.DRY_LINE, line_width=1.3, annotation_text=f"{label} {y:g}",
                      annotation_position="bottom right", annotation_font_size=11, annotation_font_color=ui.INK,
                      annotation_bgcolor="white")
    # Lines do not stretch the axis by themselves, and the extreme line can sit below every bar
    low, high = min(float(s["value"].min()), extreme), max(float(s["value"].max()), 0.0)
    fig.update_layout(title=f"MSDI-{T} by month, {district}", bargap=0, yaxis=dict(range=[low - 0.35, high + 0.2]))
    return ui.style_fig(fig, 430, legend=False, x_title="Year", y_title=f"MSDI-{T} (index value)")


def _comparison_window(T: int) -> int:
    """The SPI and SSI window beside MSDI-T: T itself, or 6, the longest that exist."""
    return T if T in common.SCALES else max(common.SCALES)


def comparison_chart(district: str, T: int, years):
    """Two stacked panels on one Year axis: MSDI-T on top, SPI and SSI below, each with its own lines."""
    w = _comparison_window(T)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=[f"MSDI-{T} (index value)", f"SPI-{w} and SSI-{w} (standard deviations)"])
    m = _in_years(_district_table("MSDI", T, district), years)
    fig.add_trace(go.Scatter(x=m["Date"], y=m["value"], name=f"MSDI-{T}", mode="lines",
                             line=dict(color=D.INDEX_INFO["MSDI"]["colour"], width=1.5),
                             hovertemplate=f"MSDI-{T} %{{x|%b %Y}}: %{{y:.2f}}<extra></extra>"), row=1, col=1)
    extreme, dry = TH.cutoff(T)
    for y, label in ((dry, "dry"), (extreme, "extremely dry")):
        fig.add_hline(y=y, line_dash="dash", line_color=ui.DRY_LINE, line_width=1.1, row=1, col=1, annotation_text=f"{label} {y:g}",
                      annotation_position="bottom right", annotation_font_size=10, annotation_font_color=ui.INK,
                      annotation_bgcolor="white")
    fig.update_yaxes(range=[min(float(m["value"].min()), extreme) - 0.35, max(float(m["value"].max()), 0.0) + 0.2], row=1, col=1)
    for index in ("SPI", "SSI"):
        t = _in_years(_district_table(index, w, district), years)
        fig.add_trace(go.Scatter(x=t["Date"], y=t["value"], name=f"{index}-{w}", mode="lines",
                                 line=dict(color=D.INDEX_INFO[index]["colour"], width=1.2),
                                 hovertemplate=f"{index}-{w} %{{x|%b %Y}}: %{{y:.2f}}<extra></extra>"), row=2, col=1)
    for y in (-1, -1.5, -2):
        fig.add_hline(y=y, line_dash="dot", line_color=ui.DRY_LINE, line_width=1, opacity=0.6, row=2, col=1)
    fig.update_layout(title=f"MSDI, SPI and SSI, {district}")
    fig.update_xaxes(title_text="Year", row=2, col=1)
    for a in fig.layout.annotations:
        if a.text.startswith(("MSDI-", "SPI-")):       # the panel titles, in the shared style
            a.update(x=0.01, xanchor="left", font=dict(size=13, color=ui.MUTED))
    return ui.style_fig(fig, 560)


def _rain_table(model) -> pd.DataFrame:
    """Rainfall fits, as in the paper's Table 2."""
    return pd.DataFrame([{"Distribution": f.family, "Parameters": f.describe(), "k": f.k, "Log-likelihood": round(f.loglik, 2),
                          "AIC": round(f.aic, 2), "Used": "yes" if f.family == model.rain.family else ""}
                         for f in sorted(model.rain_fits, key=lambda f: f.aic)])


def _temperature_table(model) -> pd.DataFrame:
    """The two-component mixture against one normal, by BIC."""
    return pd.DataFrame([{"Model": "Two-component mixture", "Fit": model.gmm.describe(), "BIC": round(model.gmm.bic, 1), "Used": "yes"},
                         {"Model": "One normal", "Fit": model.normal.describe(), "BIC": round(model.normal.bic, 1), "Used": ""}])


def _copula_table(model) -> pd.DataFrame:
    """One row per copula family: parameter, implied tau, AIC and whether it was selected."""
    return pd.DataFrame([{
        "Family": f.family, "theta": None if pd.isna(f.theta) else round(f.theta, 4),
        "tau implied": None if pd.isna(f.tau_implied) else round(f.tau_implied, 3),
        "AIC": None if pd.isna(f.aic) else round(f.aic, 2),
        "Selected": "yes" if f.family == model.best_family else "",
        "Note": f.note} for f in model.copula_fits])


def _copula_contour(model):
    """Contours of the fitted C(u, v) with every month as a dot."""
    s = model.series
    g = np.linspace(0.01, 0.99, 80)
    U, V = np.meshgrid(g, g)
    C = M.copula_cdf(model.best_family, U.ravel(), V.ravel(), model.best_theta).reshape(U.shape)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=g, y=g, z=C, contours=dict(start=0.1, end=0.9, size=0.1, showlabels=True,
                                                          coloring="lines", labelfont=dict(color=ui.MUTED)),
                             showscale=False, line=dict(width=1.2), hoverinfo="skip",
                             colorscale=[[0, "#C4561D"], [1, "#2171B5"]]))
    fig.add_trace(go.Scatter(x=s["u"], y=s["v"], mode="markers", name="months",
                             marker=dict(color=ui.INK, size=5, opacity=0.45),
                             customdata=np.c_[s["Year"], s["Month"]],
                             hovertemplate="%{customdata[0]}-%{customdata[1]:02d}: u %{x:.2f}, v %{y:.2f}<extra></extra>"))
    fig.update_layout(title=f"{model.best_family} copula C(u, v) and observed months")
    return ui.style_fig(fig, 430, legend=False, x_title="u = F(rain)  (low = dry)", y_title="v = F(temp)  (low = cool)")


def _temperature_fit(model):
    """Histogram of all months' temperature with the two-component mixture."""
    x = model.series["temp"]
    grid = np.linspace(x.min() - 0.5, x.max() + 0.5, 240)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x, histnorm="probability density", marker_color="#E9E2D6", nbinsx=35,
                               marker_line=dict(color="white", width=1), name="data",
                               hovertemplate="%{x:.1f} degC: density %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=grid, y=model.gmm.pdf(grid), mode="lines", name="mixture",
                             line=dict(color=D.INDEX_INFO["MSDI"]["colour"], width=2), hoverinfo="skip"))
    fig.update_layout(title=f"Temperature ({model.temp_col.upper()}), all months pooled")
    return ui.style_fig(fig, 430, legend=False, x_title="temperature (degC)", y_title="density")


def _expert_view(model):
    """The three fit tables, with the copula contour and the temperature histogram."""
    st.markdown("**Rainfall distributions**")
    st.dataframe(_rain_table(model), use_container_width=True, hide_index=True)
    st.markdown("**Temperature distribution**")
    st.dataframe(_temperature_table(model), use_container_width=True, hide_index=True)
    st.markdown("**Copula families**")
    st.dataframe(_copula_table(model), use_container_width=True, hide_index=True)
    st.caption("The lowest AIC is used for rainfall. The copula has the lowest AIC among the families whose range includes "
               "the sample tau.")
    c1, c2 = st.columns(2)
    with c1:
        ui.show_chart(_copula_contour(model), "Each dot is a month.")
    with c2:
        ui.show_chart(_temperature_fit(model), "The curve is the fitted mixture.")


def _tab_district(district: str, T: int, years):
    """District analysis: tiles, class chart, comparison chart, events table and expert view."""
    model = get_model(district)
    s = _district_table("MSDI", T, district)
    events = TH.drought_events(s, T)
    _tiles(model, s, events, T)
    ui.show_chart(history_chart(district, T, years),
                  "Brown bars are Dry or Extremely dry months, and light bars are Normal. Shaded spans are dry spells. "
                  "The dashed lines are the cut-offs.")
    ui.class_legend(ui.MSDI_CLASS_ORDER)
    w = _comparison_window(T)
    ui.show_chart(comparison_chart(district, T, years),
                  "The panels have different scales. Read each against its own lines."
                  + ("" if w == T else f" SPI and SSI go up to 6 months, so the lower panel shows SPI-{w} and SSI-{w}."))
    if len(events):
        with st.expander(f"Drought events found in MSDI-{T} ({TOP_EVENTS} most severe)"):
            st.dataframe(events.sort_values("severity", ascending=False).head(TOP_EVENTS),
                         use_container_width=True, hide_index=True)
    with st.expander("Expert view: rainfall, temperature and copula fits"):
        _expert_view(model)
