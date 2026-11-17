"""
soil_tabs.py
============
Streamlit renderers for the soil-moisture (SSI) district and validation tabs.

Where it sits in the pipeline:

    common.read_soil -> ssi_analysis.compute_ssi -> soil_tabs (this module)
                                                 -> pages_extra.page_ssi

    render_soil_moisture_tab   headline numbers, SSI history, events, marginals
    render_validation_tab      SPI vs SSI correlation by lag and propagation time
                               (the checks of Sanchez Alcalde & Escorihuela 2025)
"""
from __future__ import annotations

import calendar

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common
import index_data as D
import ui
from ssi_analysis import compute_ssi, cross_validate_ssi_spi, propagation_time


@st.cache_data(show_spinner="Loading soil moisture data...")
def load_soil_data(district: str) -> pd.DataFrame:
    """GWETROOT table for one district, or an empty table if the file is missing."""
    try:
        return common.read_soil(district)
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data(show_spinner="Computing SSI...")
def cached_ssi(district: str, scale: int):
    """SSIResult for one district and scale (None without soil data)."""
    df = load_soil_data(district)
    if df.empty:
        return None
    return compute_ssi(df, scale)


def _ssi_frame(r) -> pd.DataFrame:
    """Date / value table of an SSI series, for the shared chart functions."""
    s = r.ssi.dropna()
    return pd.DataFrame({"Date": s.index, "value": s.values})


# ---------------------------------------------------------------------------- district tab

def _headline(district: str, r, spi_by_scale: dict, scale: int):
    """Four metrics: events, longest event, robustness r and soil memory."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"SSI-{scale} drought events", len(r.events))
    col2.metric("Longest event", f"{r.events['duration_months'].max()} months" if len(r.events) else "n/a")
    col3.metric("Robustness r", f"{r.robustness_r:.3f}",
                help="Correlation between the fitted-distribution SSI and the distribution-free (Gringorten) "
                     "SSI. Above 0.95 means the choice of distribution is not driving the result.")
    best_k, best_r = propagation_time(spi_by_scale, cached_ssi(district, 1).ssi)
    col4.metric("Soil memory (SPI window)", f"SPI-{best_k}",
                help=f"The SPI accumulation window (1, 3 or 6 months) that best matches SSI-1 (r = {best_r:.3f}). "
                     "It shows how many months of rain the soil remembers. It is not a delay.")


def _spi_vs_ssi_chart(r, spi_by_scale: dict, scale: int, district: str, years):
    """SPI-k and SSI-k as lines on one axis."""
    ssi = _ssi_frame(r)
    spi = spi_by_scale[scale]
    series = {f"SSI-{scale}": ssi,
              f"SPI-{scale}": pd.DataFrame({"Date": spi.index, "value": spi.values})}
    colours = {f"SSI-{scale}": D.INDEX_INFO["SSI"]["colour"], f"SPI-{scale}": D.INDEX_INFO["SPI"]["colour"]}
    return ui.index_timeseries(series, colours, 340, f"Rainfall and soil moisture, {district}", years)


def _marginal_table(r) -> pd.DataFrame:
    """Selected distribution per calendar month with its AIC and KS test."""
    rows = [{"Month": calendar.month_abbr[m], "Best family": f.family, "AIC": round(f.aic, 2),
             "KS statistic": round(f.ks_stat, 4), "KS p-value": round(f.ks_pvalue, 4), "n": f.n}
            for m, f in sorted(r.marginals.items())]
    return pd.DataFrame(rows)


def render_soil_moisture_tab(district: str, spi_by_scale: dict, scale: int = 3, years=None):
    """District analysis of SSI for the sidebar district and time window."""
    r = cached_ssi(district, scale)
    if r is None:
        st.warning(f"No soil moisture data for {district}. "
                   "Run `Nasapower district polygon pipeline SOIL.py` first.")
        return
    st.subheader(f"Soil moisture drought, {district}, SSI-{scale}")
    _headline(district, r, spi_by_scale, scale)

    ui.show_chart(ui.index_bars(_ssi_frame(r), f"SSI-{scale}", f"SSI-{scale} by month, {district}", years=years),
                  "Brown bars are drier soil than normal for that month, blue wetter. Shaded spans are "
                  "dry spells (SSI ≤ -1).")
    ui.show_chart(_spi_vs_ssi_chart(r, spi_by_scale, scale, district, years),
                  "Soil moisture follows rainfall with a short delay and smooths out single dry months.")

    if len(r.events):
        with st.expander(f"Drought events found in SSI-{scale} (15 most severe)"):
            ev = r.events.copy()
            ev["start"] = ev["start"].dt.strftime("%Y-%m")
            ev["end"] = ev["end"].dt.strftime("%Y-%m")
            st.dataframe(ev.sort_values("severity", ascending=False).head(15),
                         use_container_width=True, hide_index=True)
    with st.expander("Expert view: distribution selected per calendar month"):
        st.dataframe(_marginal_table(r), use_container_width=True, hide_index=True)
        st.caption("AIC selection over Beta, Gamma, Log-normal, Log-logistic, GEV and Normal, per calendar "
                   "month, because 42 years is a small sample and one family may not win everywhere.")


# ---------------------------------------------------------------------------- validation tab

def _lag_table(district: str, spi_by_scale: dict) -> pd.DataFrame:
    """Correlation of SSI-k with SPI-k at lags -1..2 months (Alcalde et al. Table 1)."""
    rows = []
    for ssi_k in common.SCALES:
        r = cached_ssi(district, ssi_k)
        if r is None or ssi_k not in spi_by_scale:
            continue
        cv = cross_validate_ssi_spi(spi_by_scale[ssi_k], r.ssi)
        for _, c in cv.iterrows():
            rows.append({"pair": f"SSI-{ssi_k} vs SPI-{ssi_k}", "lag": int(c["lag"]),
                         "r": c["correlation"]})
    return pd.DataFrame(rows)


def _lag_chart(lag_df: pd.DataFrame):
    """Grouped bars: correlation for each lag, one group per scale."""
    fig = go.Figure()
    shades = {-1: "#C6DBEF", 0: "#6BAED6", 1: "#2171B5", 2: "#08306B"}
    for lag, sub in lag_df.groupby("lag"):
        fig.add_trace(go.Bar(x=sub["pair"], y=sub["r"], name=f"lag {lag:+d} month(s)",
                             marker_color=shades.get(lag, "#999"), marker_line=dict(color="white", width=2),
                             hovertemplate="%{x}, lag " + f"{lag:+d}" + ": r = %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group", title="Correlation of SPI(t) with SSI(t + lag)",
                      yaxis=dict(range=[0, 1]))
    return ui.style_fig(fig, 380, x_title="scale", y_title="Pearson r")


def render_validation_tab(district: str, spi_by_scale: dict):
    """SPI vs SSI at different lags, and which SPI window best matches SSI-1."""
    st.subheader(f"Does soil moisture agree with rainfall? {district}")
    st.markdown("Sanchez Alcalde and Escorihuela (2025) checked their satellite SSI against SPI at several "
                "time windows and lags. They found the best match at lag 0 for short windows and at lag 1 "
                "for longer ones, with correlations of about 0.60 to 0.67. The same check for this district:")
    lag_df = _lag_table(district, spi_by_scale)
    if len(lag_df):
        ui.show_chart(_lag_chart(lag_df),
                      "The tallest bar in each group is the lag at which soil moisture matches rainfall best. "
                      "A positive lag means the soil responds later than the rain.")
        wide = lag_df.pivot(index="pair", columns="lag", values="r").round(3)
        wide.columns = [f"lag {c:+d}" for c in wide.columns]
        st.dataframe(wide, use_container_width=True)

    ssi_1 = cached_ssi(district, 1)
    if ssi_1:
        rs = []
        for k, spi_k in spi_by_scale.items():
            idx = spi_k.index.intersection(ssi_1.ssi.index)
            if len(idx) >= 24:
                rs.append({"SPI window (months)": k, "r": round(spi_k.loc[idx].corr(ssi_1.ssi.loc[idx]), 3)})
        rs = pd.DataFrame(rs)
        fig = go.Figure(go.Bar(x=rs["SPI window (months)"].astype(str), y=rs["r"],
                               marker_color=D.INDEX_INFO["SPI"]["colour"], text=rs["r"], textposition="outside",
                               hovertemplate="SPI-%{x} vs SSI-1: r = %{y:.3f}<extra></extra>"))
        fig.update_layout(title="Propagation time: which SPI window best matches SSI-1", yaxis=dict(range=[0, 1]))
        ui.show_chart(ui.style_fig(fig, 320, legend=False, x_title="SPI window (months)", y_title="Pearson r"),
                      "The tallest bar is the district's rainfall-to-soil propagation time: how many months "
                      "of rain the root zone remembers.")
