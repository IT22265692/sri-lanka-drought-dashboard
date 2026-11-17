"""
events_panel.py
===============
One panel for historical drought events and duration-severity risk, used for
SPI and MSDI so the two are shown the same way. The SSI page uses render_events,
which shows the events only: SSI has no copula or return periods at this stage.

Where it sits in the pipeline:

    spi_copula_analysis.model_from_events  (D ~ Exponential, S ~ Gamma, copula by AIC)
        -> events_panel.render              (this module)
        -> SPI deep-dive and MSDI "Drought events" tabs
    index_data.index_events
        -> events_panel.render_events       (this module)
        -> SSI "Drought events" tab

Shows
    * event KPIs and the fitted duration-severity copula
    * event timeline (when, how long, how severe)
    * duration vs severity scatter with the fitted copula's joint-CDF contours
    * return-period curves: T_AND and T_OR of design droughts
    * return-period matrix for a grid of durations and severities
    * a "how rare is this drought?" calculator and the worst events
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common
import ui
from spi_copula_analysis import _copula_cdf, T_and, T_or, return_period_table

RECORD_YEARS = common.END_YEAR - common.START_YEAR + 1     # 42 years, 1984-2025
SEVERITY_PERCENTILES = (25, 50, 75, 90, 95, 99)
LIGHT_CELL = "#F3F0EA"                                     # cells beyond the record
EXTRAPOLATION = "model extrapolation, not a prediction"


def _event_dates(ev: pd.DataFrame) -> pd.DataFrame:
    """Add Start, End and a readable Period ("Jan 2016 to Sep 2017") to an events table."""
    ev = ev.copy()
    ev["Start"] = pd.to_datetime(dict(year=ev["StartYear"], month=ev["StartMonth"], day=1))
    ev["End"] = pd.to_datetime(dict(year=ev["EndYear"], month=ev["EndMonth"], day=1))
    ev["Period"] = ev["Start"].dt.strftime("%b %Y") + " to " + ev["End"].dt.strftime("%b %Y")
    return ev


def timeline(ev: pd.DataFrame, label: str):
    """Bubble chart: x = start date, y = duration, bubble size and colour = severity."""
    fig = go.Figure(go.Scatter(
        x=ev["Start"], y=ev["Duration"], mode="markers",
        marker=dict(size=7 + 30 * np.sqrt(ev["Severity"] / ev["Severity"].max()),
                    color=ev["Severity"], colorscale=ui.SEQUENTIAL_MARKS, cmin=0, showscale=True,
                    colorbar=dict(title="severity", thickness=12),
                    line=dict(color="white", width=1.5), opacity=0.9),
        customdata=np.c_[ev["Period"], ev["Severity"].round(2)],
        hovertemplate="%{customdata[0]}<br>%{y} months, severity %{customdata[1]}<extra></extra>"))
    fig.update_layout(title=f"Every {label} drought event (bubble size = severity)")
    return ui.style_fig(fig, 340, legend=False, x_title="Year",
                        y_title="duration (months)")


def ds_chart(model, ev: pd.DataFrame):
    """Duration vs severity of each event, over contours of the fitted joint CDF.

    The contour labelled p encloses the (d, s) pairs with P(D <= d, S <= s) = p,
    computed as C(F_D(d), F_S(s)) with the selected copula (Sklar's theorem).
    """
    m = model.marginals
    # Log axes: most events are short and mild, a few very long, so a linear
    # axis squeezes almost every event into one corner
    dmax = max(ev["Duration"].max() * 1.3, 6)
    smax = max(ev["Severity"].max() * 1.3, 4)
    smin = max(ev["Severity"].min() * 0.7, 0.01)
    dg, sg = np.geomspace(0.8, dmax, 90), np.geomspace(smin, smax, 90)
    DD, SS = np.meshgrid(dg, sg)
    C = np.asarray(_copula_cdf(model.best_family, m.F_D(DD.ravel()), m.F_S(SS.ravel()),
                               model.best_theta)).reshape(DD.shape)
    fig = go.Figure(go.Contour(
        x=dg, y=sg, z=C, showscale=False, hoverinfo="skip",
        contours=dict(coloring="lines", showlabels=True, start=0.5, end=0.95, size=0.15,
                      labelfont=dict(size=10, color=ui.MUTED)),
        line=dict(width=1.3), colorscale=[[0, "#F4A259"], [1, "#7A2E0B"]]))
    fig.add_trace(go.Scatter(x=ev["Duration"], y=ev["Severity"], mode="markers", name="events",
                             marker=dict(size=9, color=ui.INK, opacity=0.65, line=dict(color="white", width=1)),
                             customdata=ev["Period"],
                             hovertemplate="%{customdata}<br>%{x} months, severity %{y:.2f}<extra></extra>"))
    fig.update_layout(title=f"Events and the fitted {model.best_family} copula (contours: P(D ≤ d, S ≤ s))")
    fig.update_xaxes(type="log", tickvals=[1, 2, 3, 6, 12, 24, 48])
    fig.update_yaxes(type="log", tickvals=[0.1, 0.3, 1, 3, 10, 30, 100])
    return ui.style_fig(fig, 420, legend=False, x_title="duration D (months, log scale)",
                        y_title="severity S (sum of |index|, log scale)")


def rp_curves(model):
    """Joint return periods of design droughts against their own univariate return period.

    For each T, the design drought has the duration and the severity whose
    single-variable return period is T. T_AND (both exceeded) is longer than
    T and T_OR (either exceeded) shorter; the gap shows how strongly duration
    and severity are linked.
    """
    Ts = np.geomspace(2, 100, 40)
    rows = []
    for T in Ts:
        p_exc = min(model.EL_months / (T * 12), 0.999)
        d, s = model.marginals.q_D(1 - p_exc), model.marginals.q_S(1 - p_exc)
        rows.append((T, T_and(model, d, s), T_or(model, d, s), d, s))
    r = pd.DataFrame(rows, columns=["T", "AND", "OR", "d", "s"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r["T"], y=r["T"], name="one variable alone", mode="lines",
                             line=dict(color="#9A9A9A", dash="dot", width=1.5), hoverinfo="skip"))
    for kind, colour in (("AND", "#7A2E0B"), ("OR", "#2171B5")):
        fig.add_trace(go.Scatter(
            x=r["T"], y=r[kind], name=f"T_{kind}", mode="lines", line=dict(color=colour, width=2),
            customdata=np.c_[r["d"].round(1), r["s"].round(1)],
            hovertemplate=f"T = %{{x:.0f}} yr design drought (%{{customdata[0]}} months, severity "
                          f"%{{customdata[1]}}): T_{kind} = %{{y:.1f}} years<extra></extra>"))
    fig.update_xaxes(type="log", tickvals=[2, 5, 10, 20, 50, 100])
    fig.update_yaxes(type="log", tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    fig.update_layout(title="Return-period curves of design droughts")
    return ui.style_fig(fig, 420, x_title="return period of duration or severity alone (years)",
                        y_title="joint return period (years)")


def _matrix_axes(ev: pd.DataFrame) -> tuple[list[int], list[tuple[int, float]]]:
    """Durations of the rows, and (percentile, severity) of the columns, from this district's events."""
    durations = [d for d in (1, 2, 3, 4, 6, 9, 12, 18, 24, 36) if d <= max(ev["Duration"].max() * 1.5, 6)]
    columns, seen = [], set()
    for pct in SEVERITY_PERCENTILES:
        value = round(float(np.quantile(ev["Severity"], pct / 100)), 1)
        if value > 0 and value not in seen:
            seen.add(value)
            columns.append((pct, value))
    return durations, columns


def rp_matrix(model, ev: pd.DataFrame, kind: str = "AND"):
    """Heat map of T_AND or T_OR (years) for a grid of durations and severity percentiles.

    Cells longer than the record are drawn in a light style with a star: they are
    model extrapolation, not a prediction.
    """
    durations, columns = _matrix_axes(ev)
    f = T_and if kind == "AND" else T_or
    z = np.array([[f(model, d, s) for _, s in columns] for d in durations])
    text = [[f"{v:.0f}" if v >= 10 else f"{v:.1f}" for v in row] for row in z]
    inside = z <= RECORD_YEARS
    x = [f"≥ {s:g}<br>{pct}th percentile" for pct, s in columns]
    y = [f"≥ {d} mo" for d in durations]
    # Colour on a log scale (1 to 1000 years) so short and long periods both show
    fig = go.Figure(go.Heatmap(
        z=np.where(inside, np.log10(np.clip(z, 1, 1e4)), np.nan), x=x, y=y,
        text=[[t if ok else "" for t, ok in zip(row, flags)] for row, flags in zip(text, inside)],
        texttemplate="%{text}", textfont=dict(size=12), colorscale=ui.SEQUENTIAL,
        zmin=0, zmax=np.log10(RECORD_YEARS), showscale=False, xgap=2, ygap=2,
        hovertemplate="duration %{y}, severity %{x}: once every %{text} years<extra></extra>"))
    fig.add_trace(go.Heatmap(
        z=np.where(inside, np.nan, 1.0), x=x, y=y,
        text=[["" if ok else f"{t}*" for t, ok in zip(row, flags)] for row, flags in zip(text, inside)],
        texttemplate="%{text}", textfont=dict(size=11, color=ui.MUTED),
        colorscale=[[0, LIGHT_CELL], [1, LIGHT_CELL]], showscale=False, xgap=2, ygap=2,
        hovertemplate="duration %{y}, severity %{x}: %{text} years, model extrapolation, not a prediction<extra></extra>"))
    fig.update_layout(title=f"Return-period matrix, T_{kind} (years)", yaxis=dict(autorange="reversed"))
    return ui.style_fig(fig, 460, legend=False, x_title="severity at least (sum of |index|), with its percentile among this district's events",
                        y_title="duration at least")


def matrix_example(model, ev: pd.DataFrame, kind: str) -> str:
    """One real cell in words: the longest return period that is still inside the record."""
    durations, columns = _matrix_axes(ev)
    f = T_and if kind == "AND" else T_or
    cells = [(f(model, d, s), d, s) for d in durations for _, s in columns]
    inside = [c for c in cells if c[0] <= RECORD_YEARS]
    t, d, s = max(inside) if inside else min(cells)
    joiner = "and" if kind == "AND" else "or"
    return (f"**Reading one cell:** a drought of at least {d} months {joiner} severity at least {s:g} happens about "
            f"once every {t:.0f} years.")


def matrix_columns(ev: pd.DataFrame) -> str:
    """Which percentile each severity column is."""
    _, columns = _matrix_axes(ev)
    listed = ", ".join(f"{s:g} is the {pct}th percentile" for pct, s in columns)
    return (f"The severity columns are percentiles of this district's own {len(ev)} events, so they differ between "
            f"districts: {listed}.")


def _kpis(model, ev: pd.DataFrame):
    """Headline cards: number of events, average length and severity, selected copula."""
    worst = ev.loc[ev["Severity"].idxmax()]
    ui.kpi_cards([
        {"label": "Drought events", "value": f"{model.n_events}",
         "sub": f"one every {model.EL_months / 12:.1f} years on average"},
        {"label": "Average length", "value": f"{model.marginals.mean_D:.1f} mo",
         "sub": f"longest {int(ev['Duration'].max())} months"},
        {"label": "Average severity", "value": f"{model.marginals.mean_S:.2f}",
         "sub": f"worst: {worst['Period']}"},
        {"label": "Duration-severity copula", "value": model.best_family,
         "sub": f"θ = {model.best_theta:.2f}, Kendall τ = {model.kendall_tau:.2f} (lowest AIC)"},
    ])


def _calculator(model, ev: pd.DataFrame, key: str):
    """Sliders for a chosen drought size and its T_AND / T_OR, plus the standard table."""
    st.markdown("Pick a drought length and severity. This tells you how many years usually pass between droughts that bad.")
    cc1, cc2 = st.columns(2)
    d_pick = cc1.slider("Length (months)", 1, int(max(ev["Duration"].max() * 1.3, 12)),
                        int(ev["Duration"].median()) + 2, key=f"{key}_d")
    s_pick = cc2.slider("Severity score", 0.5, float(max(ev["Severity"].max() * 1.3, 8)),
                        float(round(ev["Severity"].quantile(0.75), 1)), 0.5, key=f"{key}_s")
    t_and, t_or = T_and(model, d_pick, s_pick), T_or(model, d_pick, s_pick)
    r1, r2 = st.columns(2)
    r1.metric("Both reached (AND): once every", f"{t_and:.1f} years")
    r1.caption(f"A drought at least {d_pick} months long and also at least {s_pick:g} severe.")
    r2.metric("At least one reached (OR): once every", f"{t_or:.1f} years")
    r2.caption(f"A drought at least {d_pick} months long, or at least {s_pick:g} severe, or both.")
    longer = [n for n, t in (("AND", t_and), ("OR", t_or)) if t > RECORD_YEARS]
    if longer:
        st.warning(f"The {' and '.join(longer)} result is longer than the {RECORD_YEARS}-year record: {EXTRAPOLATION}.")
    st.dataframe(return_period_table(model), use_container_width=True, hide_index=True)
    st.caption("Each row is a design drought whose length and severity each have the stated "
               f"return period on their own. Values above {RECORD_YEARS} years go beyond the record: {EXTRAPOLATION}.")


def render_events(ev: pd.DataFrame, label: str, key: str, below: str = "0"):
    """Drought events on their own: no fitted model, no copula, no return periods.

    below: the line an event runs below, as text for the captions.
    """
    if ev.empty:
        st.info(f"No {label} drought events in the record.")
        return
    ev = _event_dates(ev)
    worst = ev.loc[ev["Severity"].idxmax()]
    ui.kpi_cards([
        {"label": "Drought events", "value": f"{len(ev)}", "sub": f"{common.START_YEAR}-{common.END_YEAR}"},
        {"label": "Average length", "value": f"{ev['Duration'].mean():.1f} mo",
         "sub": f"longest {int(ev['Duration'].max())} months"},
        {"label": "Average severity", "value": f"{ev['Severity'].mean():.2f}", "sub": f"worst: {worst['Period']}"},
    ])
    st.write("")
    ui.show_chart(timeline(ev, label),
                  f"Each bubble is one run of months with {label} below {below}. Look for big, high bubbles: "
                  "long and severe droughts.", key=f"{key}_tl")
    st.markdown("**All events, most severe first**")
    st.dataframe(ev.sort_values("Severity", ascending=False)[["Period", "Duration", "Severity"]].rename(
        columns={"Duration": "Length (months)", "Severity": "Severity score"}),
        use_container_width=True, hide_index=True)


def render(model, ev: pd.DataFrame, label: str, key: str, below: str = "0"):
    """Full events-and-risk panel for one index, district and scale.

    below: the line an event runs below, as text for the captions.
    """
    if model is None or len(ev) < 10:
        st.info(f"Too few {label} drought events ({len(ev)}) to fit the duration-severity copula "
                f"(needs at least 10).")
        return
    ev = _event_dates(ev)
    _kpis(model, ev)
    st.write("")
    ui.show_chart(timeline(ev, label),
                  f"Each bubble is one run of months with {label} below {below}. Look for big, high bubbles: "
                  "long and severe droughts.", key=f"{key}_tl")
    c1, c2 = st.columns(2)
    with c1:
        ui.show_chart(ds_chart(model, ev),
                      "Long droughts are also severe ones, so the dots follow a rising band. Events "
                      "outside the 0.95 contour are the rare, extreme ones.", key=f"{key}_ds")
    with c2:
        ui.show_chart(rp_curves(model),
                      "T_AND (brown) sits above the dotted line and T_OR (blue) below it. The further "
                      "apart they are, the weaker the link between length and severity.", key=f"{key}_rc")
    kind = st.segmented_control("Matrix", ["AND", "OR"], default="AND", key=f"{key}_kind",
                                help="AND: both the duration and the severity are reached "
                                     "(rarer). OR: at least one of them is.") or "AND"
    ui.show_chart(rp_matrix(model, ev, kind),
                  "Read a cell as: a drought at least this long and (AND) / or (OR) at least this severe "
                  "happens once every N years on average.", key=f"{key}_m")
    st.markdown(matrix_example(model, ev, kind))
    st.caption(matrix_columns(ev))
    st.caption(rf"\* Lighter cells are longer than the {RECORD_YEARS}-year record: {EXTRAPOLATION}.")

    with st.expander("How rare would a drought of my chosen size be?"):
        _calculator(model, ev, key)
    with st.expander("Worst 10 events"):
        top = ev.sort_values("Severity", ascending=False).head(10)
        st.dataframe(top[["Period", "Duration", "Severity"]].rename(
            columns={"Duration": "Length (months)", "Severity": "Severity score"}),
            use_container_width=True, hide_index=True)
