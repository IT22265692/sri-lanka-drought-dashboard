"""
dashboard.py
============
Streamlit entry point of the Sri Lanka drought dashboard, and the rainfall
(SPI) page.

    py -3.12 -m streamlit run dashboard.py

Where it sits in the pipeline: the last step. It reads the index tables
(index_data.py), the MSDI fits (cache/msdi_fits.pkl) and the SPI copula fits
(cache/copula_fits.pkl, fitted on first use), and lays out the pages:

    Overview                 pages_extra.page_overview
    Rainfall (SPI)           this module
    Soil moisture (SSI)      pages_extra.page_ssi + soil_tabs
    Rainfall + heat (MSDI)   msdi_tabs
    Compare indices          pages_extra.page_compare
    Learn and how to use     learn_page

The sidebar holds the four global controls (district, index, time window,
date range) and the drought-class legend.
"""
from __future__ import annotations

import calendar
import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import common
from spi_copula_analysis import (
    _copula_log_density, build_district_model, compute_spi, estimate_spi_from_input,
    return_period_table, spi_category, spi_plain_english, tail_dependence_coefs,
)

st.set_page_config(page_title="Sri Lanka Drought Dashboard", page_icon=":material/water_drop:",
                   layout="wide")

import events_panel                          # noqa: E402  (Streamlit modules after set_page_config)
import learn_page                            # noqa: E402
import pages_extra as PX                     # noqa: E402
import ui                                    # noqa: E402
from msdi_tabs import render_msdi_layer      # noqa: E402
from soil_tabs import render_soil_moisture_tab, render_validation_tab   # noqa: E402

ui.inject_css()

SPI_FIT_CACHE = common.CACHE_DIR / "copula_fits.pkl"
SPI_COLOUR = "#0072B2"


# ============================================================================= data and cache

@st.cache_data
def list_districts() -> list[str]:
    """The 25 district names."""
    return common.list_districts()


@st.cache_data
def load_precip(district: str) -> pd.DataFrame:
    """Monthly rainfall table for one district."""
    return common.read_rainfall(district)


@st.cache_data
def compute_spi_series(district: str, scale: int) -> pd.DataFrame:
    """SPI table (Year, Month, prcp, SPI<k>) for one district and scale."""
    return compute_spi(load_precip(district), scale=scale)


def _load_fit_cache() -> dict:
    """Saved SPI duration-severity fits, {"<district>__SPI<k>": {...}}."""
    if SPI_FIT_CACHE.exists():
        with open(SPI_FIT_CACHE, "rb") as f:
            return pickle.load(f)
    return {}


def _save_fit_cache(cache: dict):
    """Write the SPI fits via a temporary file so a reader never sees half a pickle."""
    SPI_FIT_CACHE.parent.mkdir(exist_ok=True)
    tmp = SPI_FIT_CACHE.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(cache, f)
    os.replace(tmp, SPI_FIT_CACHE)


def get_or_fit_model(district: str, scale: int) -> dict:
    """SPI duration-severity model for one district and scale, fitted once and saved.

    Returns {"model": DistrictModel, "events": events table, "return_periods": table}.
    """
    cache = _load_fit_cache()
    key = f"{district}__SPI{scale}"
    # Fits from an old library version may name a family no longer supported; refit those
    if key in cache and cache[key]["model"].best_family in ("Gumbel", "Clayton", "Frank"):
        return cache[key]
    with st.spinner(f"First analysis for {district} SPI-{scale}, fitting 3 copulas "
                    f"(about 15 seconds). Saved for later visits."):
        model, _, events = build_district_model(load_precip(district), district, scale)
    cache[key] = {"model": model, "events": events, "return_periods": return_period_table(model)}
    _save_fit_cache(cache)
    return cache[key]


# ============================================================================= SPI deep-dive

def _spi_frame(district: str, scale: int) -> pd.DataFrame:
    """Date / value table of SPI-k, for the shared chart functions."""
    s = compute_spi_series(district, scale).dropna(subset=[f"SPI{scale}"])
    return pd.DataFrame({"Date": pd.to_datetime(s[["Year", "Month"]].assign(day=1)), "value": s[f"SPI{scale}"]})


def _rainfall_calculator(district: str, scale: int):
    """Form: type a rainfall total for a month and see the SPI it would give."""
    precip_df = load_precip(district)
    with st.form("rainfall_input_form"):
        rc1, rc2, rc3 = st.columns([1, 1, 1.4])
        in_year = rc1.number_input("Year", min_value=1984, max_value=2035, value=int(precip_df["Year"].max()))
        in_month = rc2.selectbox("Month", list(range(1, 13)), format_func=lambda m: calendar.month_name[m],
                                 key="rain_input_month")
        in_rain = rc3.number_input("Rainfall this month (mm)", min_value=0.0, max_value=2000.0,
                                   value=100.0, step=5.0)
        submitted = st.form_submit_button("Calculate my SPI")
    if not submitted:
        return
    est = estimate_spi_from_input(precip_df, scale, in_year, in_month, in_rain)
    if est is None:
        st.warning("Not enough surrounding data to compute SPI for that month at this time window.")
        return
    label, _, colour = spi_category(est)
    st.markdown(
        f"<div style='background:{colour}22;border-left:5px solid {colour};padding:14px 18px;"
        f"border-radius:8px;margin-top:8px'><span style='font-size:24px;font-weight:700;color:{ui.INK}'>"
        f"SPI-{scale} = {est:.2f}</span>&nbsp;&nbsp;<b>{label}</b><br>"
        f"<span style='color:{ui.INK}'>{spi_plain_english(est)}</span></div>", unsafe_allow_html=True)
    st.caption(f"Based on {in_rain:.0f} mm in {calendar.month_name[in_month]} {in_year}"
               + (f", together with {district}'s real rainfall in the {scale - 1} month(s) before it"
                  if scale > 1 else "") + f", compared with {district}'s 1984-2025 record.")


def tab_district_deepdive():
    """SPI deep-dive for the sidebar district and time window."""
    sel = ui.selection()
    district, scale = sel["district"], sel["scale"]
    result = get_or_fit_model(district, scale)
    model, events = result["model"], result["events"]

    st.subheader(f"{district}, SPI-{scale}")
    ui.show_chart(ui.index_bars(_spi_frame(district, scale), f"SPI-{scale}",
                                f"SPI-{scale} by month, {district}", 380, sel["years"]),
                  "Brown bars are drier than normal for that month of the year, blue wetter. Shaded spans "
                  "are dry spells (SPI ≤ -1).")
    with st.expander("Try your own rainfall reading"):
        _rainfall_calculator(district, scale)

    st.subheader("Drought events and return periods")
    events_panel.render(model, events, f"SPI-{scale}", f"spi_ev_{district}_{scale}")

    if st.toggle("Expert view: copula fit, tail dependence and 3D surfaces", value=False, key="spi_expert"):
        expert_district_view(model, events)


# ============================================================================= expert view

JOINT_TITLE = "Joint density f(d, s): where droughts cluster (per month, per unit severity)"
COPULA_TITLE = "Copula density c(u, v): how much more likely than independence (flat = 1)"
JOINT_CAP, COPULA_CAP = 97, 98           # percentile at which the top of each plot is capped for display


def _fits_table(model) -> pd.DataFrame:
    """Gumbel, Clayton and Frank fits sorted by AIC, with the selected family marked."""
    df = pd.DataFrame([{"Family": f.family, "Parameter": round(f.parameter, 4), "Log-Lik": round(f.loglik, 2),
                        "AIC": round(f.aic, 2), "Upper tail": "yes" if f.tail_upper else "no",
                        "Lower tail": "yes" if f.tail_lower else "no"} for f in model.all_fits])
    df = df.sort_values("AIC").reset_index(drop=True)
    df["Selected"] = df["Family"].apply(lambda f: "yes" if f == model.best_family else "")
    return df


def _tail_text(up: float, low: float) -> str:
    """Plain-language reading of the tail dependence coefficients."""
    if up > 0.05:
        return (f"lambda_U = {up:.3f}: when duration is in its top few percent, severity has roughly a "
                f"{up * 100:.0f}% chance of also being in its top few percent.")
    if low > 0.05:
        return f"lambda_L = {low:.3f}: the strongest coupling is between short, mild events."
    return "No tail dependence: extreme droughts are no more linked than typical ones under this copula."


def _nice_ticks(top: float, count: int = 5) -> np.ndarray:
    """About count round tick values from 0 to top."""
    step = 10.0 ** np.floor(np.log10(top / count))
    for m in (1, 2, 5, 10):
        if top / (m * step) <= count:
            step *= m
            break
    return np.arange(0, top + step * 1e-6, step)


def _label_3d(x, y, z, text, xshift=0, yshift=0, angle=0, bold=False) -> dict:
    """A text label fixed to a point in a 3D scene.

    Plotly draws its own 3D axis text in WebGL, where a font size of 14 comes out at
    about half that. Scene annotations are ordinary page text, so 14 really is 14,
    and they still follow the plot when it rotates.
    """
    return dict(x=x, y=y, z=z, text=f"<b>{text}</b>" if bold else text, showarrow=False, xshift=xshift, yshift=yshift,
                textangle=angle, font=dict(size=14, color=ui.INK))


def _surface(x, y, z, x_title, y_title, z_title, x_top, y_top):
    """3D surface with readable labels. Axes run from 0 to x_top and y_top."""
    z_top = float(np.max(z))
    fig = go.Figure(go.Surface(x=x, y=y, z=z, colorscale=ui.SEQUENTIAL, showscale=False))
    bare = dict(showticklabels=False, title=dict(text=""))
    labels = [_label_3d(t, y_top, 0, f"{t:g}", xshift=16, yshift=-12) for t in _nice_ticks(x_top)]
    labels += [_label_3d(x_top, t, 0, f"{t:g}", xshift=-16, yshift=-12) for t in _nice_ticks(y_top)]
    labels += [_label_3d(0, y_top, t, f"{t:.2g}", xshift=-26) for t in _nice_ticks(z_top, 4)[1:]]
    labels += [_label_3d(x_top / 2, y_top, 0, x_title, xshift=62, yshift=-40, angle=-33, bold=True),
               _label_3d(x_top, y_top / 2, 0, y_title, xshift=-62, yshift=-40, angle=33, bold=True),
               _label_3d(0, y_top, z_top, z_title, xshift=-40, yshift=26, bold=True)]
    fig.update_layout(scene=dict(xaxis=dict(range=[0, x_top], **bare), yaxis=dict(range=[0, y_top], **bare),
                                 zaxis=dict(range=[0, z_top], **bare), annotations=labels,
                                 domain=dict(x=[0.06, 0.94], y=[0.16, 1]),
                                 camera=dict(eye=dict(x=1.45, y=1.45, z=1.0))),
                      title=dict(text="3D view (drag to rotate)", x=0.01, font=dict(size=14, color=ui.INK)),
                      height=560, margin=dict(l=0, r=0, t=36, b=0), paper_bgcolor=ui.CARD,
                      font=dict(family=ui.FONT, color=ui.INK))
    return fig


def _flat_view(x, y, z, x_title, y_title, z_title, x_range, y_range, hover):
    """The same surface as a flat heat map, readable without rotating."""
    fig = go.Figure(go.Heatmap(x=x, y=y, z=z, colorscale=ui.SEQUENTIAL, hovertemplate=hover + "<extra></extra>",
                               colorbar=dict(title=z_title, thickness=12)))
    fig.update_layout(title="Flat view", xaxis=dict(range=x_range), yaxis=dict(range=y_range))
    return ui.style_fig(fig, 560, legend=False, x_title=x_title, y_title=y_title)


def _joint_grid(model, events: pd.DataFrame):
    """f(d, s) = c(F_D(d), F_S(s)) f_D(d) f_S(s) from 0 to the largest observed duration and severity.

    Returns durations, severities, the density (rows = severity) and the display cap.
    """
    m = model.marginals
    d_seq = np.linspace(0.1, float(events["Duration"].max()), 60)
    s_seq = np.linspace(0.1, float(events["Severity"].max()), 60)
    DD, SS = np.meshgrid(d_seq, s_seq)
    c = np.exp(_copula_log_density(model.best_family, m.F_D(DD).ravel(), m.F_S(SS).ravel(), model.best_theta)).reshape(DD.shape)
    joint = c * m.f_D(DD) * m.f_S(SS)
    joint[~np.isfinite(joint)] = 0
    cap = np.percentile(joint[joint > 0], JOINT_CAP) if (joint > 0).any() else joint.max()
    return d_seq, s_seq, joint, cap


def _copula_grid(model):
    """c(u, v) on the unit square (rows = v) and its display cap."""
    g = np.linspace(0.02, 0.98, 60)
    UU, VV = np.meshgrid(g, g)
    c = np.exp(_copula_log_density(model.best_family, UU.ravel(), VV.ravel(), model.best_theta)).reshape(UU.shape)
    c[~np.isfinite(c)] = 0
    cap = np.percentile(c[c > 0], COPULA_CAP) if (c > 0).any() else c.max()
    return g, c, cap


def _copula_at(model, u: float, v: float) -> float:
    """c(u, v) at one point: how many times more likely than independence."""
    return float(np.exp(_copula_log_density(model.best_family, np.array([u]), np.array([v]), model.best_theta))[0])


def _joint_density_view(model, events: pd.DataFrame, label: str):
    """3D and flat joint density side by side, with the events as dots and a plain caption."""
    d_seq, s_seq, joint, cap = _joint_grid(model, events)
    shown = np.minimum(joint, cap)
    d_max, s_max = float(events["Duration"].max()), float(events["Severity"].max())
    x_t, y_t = "Duration D (months)", "Severity S (sum of |index|)"
    st.markdown(f"**{JOINT_TITLE}**")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(_surface(d_seq, s_seq, shown, x_t, "Severity S", "f(d, s), capped", d_max, s_max),
                        use_container_width=True, key="spi_joint_3d")
    with right:
        fig = _flat_view(d_seq, s_seq, shown, x_t, y_t, "f(d, s)", [0, d_max], [0, s_max],
                         "%{x:.1f} months, severity %{y:.1f}: density %{z:.4f}")
        ev = events_panel._event_dates(events)
        fig.add_trace(go.Scatter(x=ev["Duration"], y=ev["Severity"], mode="markers", name="observed events", cliponaxis=False,
                                 marker=dict(size=8, color=ui.INK, opacity=0.75, line=dict(color="white", width=1)),
                                 customdata=ev["Period"], hovertemplate="%{customdata}<br>%{x} months, severity %{y:.2f}<extra></extra>"))
        fig.update_layout(title="Flat view (dots are the observed events)")
        st.plotly_chart(fig, use_container_width=True, key="spi_joint_flat")
    row, col = np.unravel_index(np.argmax(joint), joint.shape)
    st.caption(f"Axes: drought length in months across, severity (the sum of |index| over the drought) along the other side, "
               f"and how common that combination is as height or colour. A peak means droughts of that size are common. "
               f"For {label}, the {len(events)} observed events are the dots: the density is highest near "
               f"{d_seq[col]:.1f} months and severity {s_seq[row]:.1f}, so most droughts are short and mild, and it falls away "
               f"along a rising band because length and severity go together (Kendall's tau {model.kendall_tau:.2f}, "
               f"{model.best_family} copula). The top is capped at the {JOINT_CAP}th percentile for display, so the true peak "
               f"({joint.max():.4f}) is higher than shown ({cap:.4f}).")


def _copula_density_view(model, label: str):
    """3D and flat copula density side by side, with a plain caption."""
    g, c, cap = _copula_grid(model)
    shown = np.minimum(c, cap)
    x_t, y_t = "u = F_D(D) (duration percentile, 0 to 1)", "v = F_S(S) (severity percentile, 0 to 1)"
    st.markdown(f"**{COPULA_TITLE}**")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(_surface(g, g, shown, "u (duration percentile)", "v (severity percentile)", "c(u, v), capped", 1.0, 1.0),
                        use_container_width=True, key="spi_copula_3d")
    with right:
        st.plotly_chart(_flat_view(g, g, shown, x_t, y_t, "c(u, v)", [0, 1], [0, 1], "u %{x:.2f}, v %{y:.2f}: c = %{z:.2f}"),
                        use_container_width=True, key="spi_copula_flat")
    high, low, mixed = _copula_at(model, 0.9, 0.9), _copula_at(model, 0.1, 0.1), _copula_at(model, 0.9, 0.1)
    st.caption(f"Axes: u and v are the percentiles of a drought's length and severity among this district's droughts, both "
               f"from 0 to 1. Height or colour is how many times more likely that pair is than if length and severity were "
               f"unrelated, so a flat surface at 1 would mean no link, and a peak means that pair occurs together more often. "
               f"For {label} (Kendall's tau {model.kendall_tau:.2f}, {model.best_family} copula), a long and severe drought "
               f"(both at the 90th percentile) is {high:.1f} times as likely as under no link, a short and mild one {low:.1f} "
               f"times, and a long but mild one {mixed:.2f} times. The top is capped at the {COPULA_CAP}th percentile for "
               f"display, at {cap:.1f}.")


def _pair_counts(x: np.ndarray, y: np.ndarray) -> dict:
    """Concordant, discordant and tied pairs of two series."""
    dx = np.sign(x[:, None] - x[None, :])[np.triu_indices(len(x), 1)]
    dy = np.sign(y[:, None] - y[None, :])[np.triu_indices(len(y), 1)]
    return {"pairs": len(dx), "concordant": int((dx * dy > 0).sum()), "discordant": int((dx * dy < 0).sum()),
            "tied duration": int((dx == 0).sum()), "tied severity": int((dy == 0).sum())}


def _dependence_help(model, events: pd.DataFrame):
    """How Kendall's tau and Spearman's rho are calculated, with this district's numbers."""
    x, y = events["Duration"].to_numpy(float), events["Severity"].to_numpy(float)
    c = _pair_counts(x, y)
    simple = (c["concordant"] - c["discordant"]) / c["pairs"]
    corrected = (c["concordant"] - c["discordant"]) / np.sqrt((c["pairs"] - c["tied duration"]) * (c["pairs"] - c["tied severity"]))
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    worst = int(np.argmax(y))
    st.markdown("**Kendall's tau** compares every pair of droughts. A pair is concordant when the longer drought is also the "
                "more severe one, and discordant when it is the less severe one.")
    st.latex(r"\tau = \frac{\text{concordant} - \text{discordant}}{\text{total pairs}}")
    st.markdown("Durations are whole months, so many pairs tie, and a tied pair is neither concordant nor discordant. The "
                "dashboard therefore uses the tie-corrected version, tau-b, which shrinks the denominator by the tied pairs:")
    st.latex(r"\tau_b = \frac{\text{concordant} - \text{discordant}}{\sqrt{(\text{pairs} - \text{tied D})(\text{pairs} - \text{tied S})}}")
    st.markdown(f"**Example:** {len(x)} droughts give {c['pairs']} pairs: {c['concordant']} concordant, {c['discordant']} discordant, "
                f"{c['tied duration']} tied on duration and {c['tied severity']} tied on severity. The simple formula gives "
                f"({c['concordant']} - {c['discordant']}) / {c['pairs']} = {simple:.3f}. With the tie correction it gives "
                f"**{corrected:.3f}**, the value shown above ({model.kendall_tau:.3f}).")
    st.markdown("**Spearman's rho** is the Pearson correlation of the ranks: rank the droughts by duration, rank them by severity "
                "(ties share an average rank), and correlate the two lists of ranks.")
    st.latex(r"\rho = \operatorname{corr}\big(\operatorname{rank}(D),\ \operatorname{rank}(S)\big)")
    st.markdown(f"**Example:** the most severe drought ranks {ry[worst]:.0f} of {len(y)} on severity and {rx[worst]:.1f} on duration. "
                f"Correlating all {len(x)} pairs of ranks gives **{rho:.3f}**, the value shown above ({model.spearman_rho:.3f}).")


def _selection_help(model):
    """How the duration-severity copula is selected, with this district's numbers."""
    fits = sorted(model.all_fits, key=lambda f: f.aic)
    st.markdown("1. **Percentiles.** Each drought becomes a pair of percentiles: u = F_D(D) for its duration and v = F_S(S) for "
                "its severity, both between 0 and 1.\n"
                "2. **Maximum likelihood.** For each family (Gumbel, Clayton, Frank), find the theta that makes the observed (u, v) "
                "pairs most probable, which gives the log-likelihood ln L.\n"
                "3. **AIC.** Each family has one parameter, so its score is the formula below.\n"
                "4. **Lowest AIC wins.**")
    st.latex(r"\text{AIC} = -2\ln L + 2")
    st.markdown("**This district:** " + "; ".join(f"{f.family}: -2 x {f.loglik:.2f} + 2 = {f.aic:.2f}" for f in fits)
                + f". Lowest: **{fits[0].family}**.")
    st.markdown("**Tail dependence** says how strongly the extremes occur together. Frank has none.")
    st.latex(r"\lambda_U^{\text{Gumbel}} = 2 - 2^{1/\theta},\qquad \lambda_L^{\text{Clayton}} = 2^{-1/\theta},\qquad "
             r"\lambda_U^{\text{Frank}} = \lambda_L^{\text{Frank}} = 0")


def expert_district_view(model, events: pd.DataFrame):
    """Copula comparison, tail dependence, marginals and the two density views for the SPI model."""
    sel = ui.selection()
    label = f"{sel['district']}, SPI-{sel['scale']}"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kendall's tau (D, S)", f"{model.kendall_tau:.3f}")
    c2.metric("Spearman's rho (D, S)", f"{model.spearman_rho:.3f}")
    c3.metric("Selected copula", model.best_family)
    c4.metric("theta", f"{model.best_theta:.4f}")
    with st.expander("How these are calculated"):
        _dependence_help(model, events)
    st.dataframe(_fits_table(model), use_container_width=True, hide_index=True)
    st.caption("Three families fitted by maximum likelihood on u = F_D(D), v = F_S(S); the lowest AIC is selected.")
    with st.expander("How the copula is selected"):
        _selection_help(model)
    up, low = tail_dependence_coefs(model.best_family, model.best_theta)
    t1, t2 = st.columns(2)
    t1.metric("Upper-tail coefficient lambda_U", f"{up:.3f}")
    t2.metric("Lower-tail coefficient lambda_L", f"{low:.3f}")
    ui.note(_tail_text(up, low))
    m = model.marginals
    st.dataframe(pd.DataFrame([
        {"Variable": "Duration D", "Family": "Exponential", "Parameter": f"lambda = {m.lambda_d:.4f}",
         "Mean": f"{1 / m.lambda_d:.2f}"},
        {"Variable": "Severity S", "Family": "Gamma", "Parameter": f"alpha = {m.alpha_s:.4f}, beta = {m.beta_s:.4f}",
         "Mean": f"{m.alpha_s * m.beta_s:.2f}"}]), use_container_width=True, hide_index=True)
    tab_pdf, tab_density = st.tabs(["Joint density f(d, s)", "Copula density c(u, v)"])
    with tab_pdf:
        _joint_density_view(model, events, label)
    with tab_density:
        _copula_density_view(model, label)


# ============================================================================= compare districts

def tab_compare_districts():
    """Side-by-side SPI drought statistics for two districts (A defaults to the sidebar district)."""
    sel = ui.selection()
    districts = list_districts()
    c1, c2 = st.columns(2)
    d1 = c1.selectbox("District A", districts, index=districts.index(sel["district"]), key="cmp_a")
    d2 = c2.selectbox("District B", districts, index=districts.index("Colombo"), key="cmp_b")
    if d1 == d2:
        st.warning("Pick two different districts to compare.")
        return
    scale = sel["scale"]
    series = {d1: _spi_frame(d1, scale), d2: _spi_frame(d2, scale)}
    ui.show_chart(ui.index_timeseries(series, {d1: SPI_COLOUR, d2: "#D55E00"}, 360,
                                      f"SPI-{scale}: {d1} and {d2}", sel["years"], documented=False),
                  "Where the two lines move together, both districts shared the same dry or wet spell.")
    rows = []
    for d in (d1, d2):
        r = get_or_fit_model(d, scale)
        rows.append({"District": d, "Drought events": r["model"].n_events,
                     "Average length (months)": round(r["model"].marginals.mean_D, 1),
                     "Average severity": round(r["model"].marginals.mean_S, 2),
                     "Longest (months)": int(r["events"]["Duration"].max()),
                     "Most severe": round(r["events"]["Severity"].max(), 1),
                     "Copula": r["model"].best_family})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================================================================= pages

def page_spi():
    """Rainfall page: national map, district deep-dive, compare two districts."""
    sel = ui.selection()
    ui.page_header("Rainfall drought (SPI)",
                   "Standardized Precipitation Index from NASA POWER rainfall (PRECTOTCORR), Gamma "
                   f"distribution per calendar month. District and time window from the sidebar "
                   f"({sel['district']}, {sel['scale']} months).")
    tab1, tab2, tab3 = st.tabs(["National map", "District deep-dive", "Compare districts"])
    with tab1:
        PX.national_map_tab("SPI", "spi_nat",
                            "Brown means drier than normal for this district and this time of year, "
                            "blue wetter. A 3-month window follows a cultivation season.")
    with tab2:
        tab_district_deepdive()
    with tab3:
        tab_compare_districts()


def page_ssi():
    """Soil-moisture page (pages_extra.page_ssi)."""
    PX.page_ssi(render_soil_moisture_tab, render_validation_tab, compute_spi_series)


def page_msdi():
    """Rainfall + heat page (msdi_tabs.render_msdi_layer)."""
    render_msdi_layer(list_districts())


def _sidebar():
    """Title, the four global controls, the class legend and data notes."""
    with st.sidebar:
        st.markdown(
            f"<div style='padding:2px 0 6px'><div style='font-size:21px;font-weight:800;color:{ui.INK}'>"
            f"Sri Lanka Drought</div><div style='color:{ui.MUTED};font-size:13px'>Monitoring for paddy "
            f"cultivation, 25 districts, 1984-2025</div></div>", unsafe_allow_html=True)
    ui.sidebar_controls(list_districts())
    with st.sidebar:
        st.divider()
        st.caption("Drought classes, SPI and SSI (MSDI has its own, on its page)")
        ui.class_legend()
        b = PX.bundle()
        st.caption(f"Data: NASA POWER (PRECTOTCORR, TS, GWETROOT), GADM district boundaries. "
                   f"Index tables built {b['built']}.")
        for n in b["notes"]:
            st.caption(f"Note: {n}")


def main():
    """Build the navigation and the sidebar, then run the selected page."""
    nav = st.navigation([
        st.Page(PX.page_overview, title="Overview", icon=":material/dashboard:", default=True),
        st.Page(page_spi, title="Rainfall (SPI)", icon=":material/rainy:", url_path="spi"),
        st.Page(page_ssi, title="Soil moisture (SSI)", icon=":material/grass:", url_path="ssi"),
        st.Page(page_msdi, title="Rainfall + heat (MSDI)", icon=":material/thermostat:", url_path="msdi"),
        st.Page(PX.page_compare, title="Compare indices", icon=":material/compare_arrows:",
                url_path="compare"),
        st.Page(learn_page.page_learn, title="Learn and how to use", icon=":material/school:",
                url_path="learn"),
    ], position="sidebar")
    _sidebar()
    nav.run()


if __name__ == "__main__":
    main()
