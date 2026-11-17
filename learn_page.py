"""
learn_page.py
=============
"Learn and how to use" page: every step of the method, in order, with the
equation, a worked example using real numbers from the sidebar district, and
a small plot of that step.

Where it sits in the pipeline: it reads the same functions and fits the other
pages use (spi_copula_analysis, ssi_analysis, msdi_analysis via msdi_tabs,
compare_indices via pages_extra), so every number shown here is the number
the dashboard uses.

Steps (one expander each): data, SPI, run theory, marginals, copulas, joint
return periods, SSI, MSDI, MSDI classes, comparing the indices, limitations, glossary.
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

import common
import index_data as D
import msdi_analysis as M
import pages_extra as PX
import thresholds as TH
import ui
from spi_copula_analysis import (
    SPI_CATEGORIES, T_and, T_or, _copula_cdf, build_district_model, compute_spi,
    tail_dependence_coefs,
)
from ssi_analysis import compute_ssi

COLOURS = {k: v["colour"] for k, v in D.INDEX_INFO.items()}


def _fmt_month(year: int, month: int) -> str:
    """'Mar 2017'."""
    return f"{calendar.month_abbr[month]} {year}"


# ============================================================================= cached calculations

@st.cache_data(show_spinner=False)
def _inputs(district: str) -> pd.DataFrame:
    """Rainfall, soil wetness and ground temperature for one district on one monthly table."""
    rain = common.read_rainfall(district)[["Year", "Month", "prcp"]]
    soil = common.read_soil(district)
    soil = pd.DataFrame({"Year": soil["date"].dt.year, "Month": soil["date"].dt.month, "gwetroot": soil["gwetroot"]})
    temp = common.read_temperature(district)[["Year", "Month", "ts"]]
    df = rain.merge(soil, on=["Year", "Month"], how="left").merge(temp, on=["Year", "Month"], how="left")
    df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(day=1))
    return df


@st.cache_data(show_spinner=False)
def _spi_example(district: str, scale: int) -> dict:
    """The driest SPI-k month on record and every number behind its SPI value.

    Repeats compute_spi's own steps for that one calendar month, so the
    worked SPI equals the dashboard's SPI exactly.
    """
    rain = common.read_rainfall(district)
    spi = compute_spi(rain, scale=scale)
    col = f"SPI{scale}"
    row = spi.loc[spi[col].idxmin()]
    y, m = int(row["Year"]), int(row["Month"])
    rolled = spi["prcp"].rolling(scale, min_periods=scale).sum()
    same_month = rolled[(spi["Month"] == m) & rolled.notna()].to_numpy().copy()
    n_zero = int((same_month <= 0).sum())
    same_month[same_month <= 0] = 0.001
    shape, _, scale_g = stats.gamma.fit(same_month, floc=0)
    x = float(rolled[(spi["Year"] == y) & (spi["Month"] == m)].iloc[0])
    F = float(np.clip(stats.gamma.cdf(max(x, 0.001), a=shape, scale=scale_g), 1e-4, 1 - 1e-4))
    A = np.log(same_month.mean()) - np.log(same_month).mean()
    return {"year": y, "month": m, "x": x, "sample": same_month, "alpha": shape, "beta": scale_g,
            "F": F, "spi": float(stats.norm.ppf(F)), "table_spi": float(row[col]), "n": len(same_month),
            "n_zero": n_zero, "A": A, "n_zero_all": int((rain["prcp"] <= 0).sum())}


@st.cache_data(show_spinner=False)
def _event_model(district: str, scale: int):
    """SPI events and the duration-severity model (same fit as the SPI page)."""
    model, spi_df, events = build_district_model(common.read_rainfall(district), district, scale)
    return model, spi_df, events


@st.cache_data(show_spinner=False)
def _ssi_example(district: str, scale: int, year: int, month: int) -> dict:
    """SSI-k at one month, with the family selected for that calendar month."""
    r = compute_ssi(common.read_soil(district), scale)
    when = pd.Timestamp(year, month, 1)
    soil = common.read_soil(district).set_index("date")["gwetroot"].rolling(scale, min_periods=scale).mean()
    sample = soil[soil.index.month == month].dropna()
    fit = r.marginals[month]
    x = float(soil.loc[when])
    return {"x": x, "family": fit.family, "params": fit.params, "aic": fit.aic, "F": float(fit.cdf(np.array([x]))[0]),
            "ssi": float(r.ssi.loc[when]), "sample": sample.to_numpy(), "robust": r.robustness_r}


# ============================================================================= small plots

def _mini(fig, height=300, x_title=None, y_title=None, legend=False):
    """Shared style for the small step plots."""
    return ui.style_fig(fig, height, legend=legend, x_title=x_title, y_title=y_title)


def _data_plot(df: pd.DataFrame, years):
    """Rainfall, soil wetness and temperature, one row each."""
    d = df[(df["Year"] >= years[0]) & (df["Year"] <= years[1])]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["Rainfall (mm per month)", "Root-zone soil wetness GWETROOT (0 to 1)",
                                        "Ground temperature TS (degC)"])
    for r, (col, colour) in enumerate([("prcp", COLOURS["SPI"]), ("gwetroot", COLOURS["SSI"]),
                                       ("ts", COLOURS["MSDI"])], 1):
        fig.add_trace(go.Scatter(x=d["Date"], y=d[col], mode="lines", line=dict(color=colour, width=1.3),
                                 hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>"), row=r, col=1)
    for a in fig.layout.annotations:
        a.update(x=0.01, xanchor="left", font=dict(size=12, color=ui.MUTED))
    fig.update_layout(title="The three inputs, month by month")
    fig.update_xaxes(title_text="Year", row=3, col=1)
    return _mini(fig, 520, None, None)


def _spi_plot(ex: dict, scale: int):
    """Histogram of one calendar month's k-month totals, the fitted Gamma, and the example."""
    grid = np.linspace(0.001, ex["sample"].max() * 1.15, 300)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ex["sample"], histnorm="probability density", marker_color="#E9E2D6",
                               marker_line=dict(color="white", width=1), name="42 years",
                               hovertemplate="%{x:.0f} mm: density %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=grid, y=stats.gamma.pdf(grid, ex["alpha"], scale=ex["beta"]), mode="lines",
                             name="fitted Gamma", line=dict(color=COLOURS["SPI"], width=2), hoverinfo="skip"))
    fig.add_vline(x=ex["x"], line_color=ui.DRY_LINE, line_width=2,
                  annotation_text=f"{_fmt_month(ex['year'], ex['month'])}: {ex['x']:.0f} mm",
                  annotation_position="top right")
    fig.update_layout(title=f"{scale}-month rainfall ending in {calendar.month_name[ex['month']]}, every year")
    return _mini(fig, 320, f"{scale}-month rainfall total (mm)", "density")


def _event_plot(spi_df: pd.DataFrame, ev_row: pd.Series, scale: int):
    """SPI bars around the worst event, with the event shaded."""
    col = f"SPI{scale}"
    s = spi_df.dropna(subset=[col]).copy()
    s["Date"] = pd.to_datetime(s[["Year", "Month"]].assign(day=1))
    start = pd.Timestamp(int(ev_row["StartYear"]), int(ev_row["StartMonth"]), 1)
    end = pd.Timestamp(int(ev_row["EndYear"]), int(ev_row["EndMonth"]), 1)
    w = s[(s["Date"] >= start - pd.DateOffset(months=12)) & (s["Date"] <= end + pd.DateOffset(months=12))]
    fig = go.Figure(go.Bar(x=w["Date"], y=w[col], marker_color=np.where(w[col] < 0, "#C4561D", "#2171B5"),
                           hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>"))
    fig.add_vrect(x0=start, x1=end + pd.offsets.MonthBegin(1), fillcolor=ui.DRY_SHADE, line_width=0, layer="below",
                  annotation_text="the event", annotation_position="top left")
    fig.add_hline(y=0, line_color="#9A9A9A")
    fig.update_layout(title="The worst event on record, with a year either side", bargap=0.05)
    return _mini(fig, 300, "Year", f"SPI-{scale}")


def _marginal_plot(model, events: pd.DataFrame):
    """Histograms of D and S with the fitted Exponential and Gamma densities."""
    m = model.marginals
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Duration D ~ Exponential", "Severity S ~ Gamma"])
    for c, (vals, pdf) in enumerate([(events["Duration"], m.f_D), (events["Severity"], m.f_S)], 1):
        grid = np.linspace(0.01, vals.max() * 1.1, 200)
        fig.add_trace(go.Histogram(x=vals, histnorm="probability density", marker_color="#E9E2D6",
                                   marker_line=dict(color="white", width=1)), row=1, col=c)
        fig.add_trace(go.Scatter(x=grid, y=pdf(grid), mode="lines", line=dict(color=ui.ACCENT, width=2),
                                 hoverinfo="skip"), row=1, col=c)
    fig.update_xaxes(title_text="months", row=1, col=1)
    fig.update_xaxes(title_text="sum of |SPI|", row=1, col=2)
    fig.update_yaxes(title_text="density", row=1, col=1)
    for a in fig.layout.annotations:
        a.update(font=dict(size=12, color=ui.MUTED))
    return _mini(fig, 300)


def _copula_plot(model, events: pd.DataFrame):
    """(u, v) of each event over the selected copula's CDF contours."""
    m = model.marginals
    u, v = m.F_D(events["Duration"].values), m.F_S(events["Severity"].values)
    g = np.linspace(0.01, 0.99, 70)
    U, V = np.meshgrid(g, g)
    C = np.asarray(_copula_cdf(model.best_family, U.ravel(), V.ravel(), model.best_theta)).reshape(U.shape)
    fig = go.Figure(go.Contour(x=g, y=g, z=C, showscale=False, hoverinfo="skip", line=dict(width=1.1),
                               contours=dict(coloring="lines", showlabels=True, start=0.1, end=0.9, size=0.2),
                               colorscale=[[0, "#F4A259"], [1, "#7A2E0B"]]))
    fig.add_trace(go.Scatter(x=u, y=v, mode="markers", marker=dict(size=7, color=ui.INK, opacity=0.6),
                             hovertemplate="u %{x:.2f}, v %{y:.2f}<extra></extra>"))
    fig.update_layout(title=f"Events on the unit square, {model.best_family} copula contours")
    return _mini(fig, 340, "u = F_D(D)", "v = F_S(S)")


def _ssi_plot(ex: dict, scale: int, month: int):
    """Histogram of one calendar month's soil wetness, the selected distribution and the example."""
    dist = {"loglogistic": "fisk", "gev": "genextreme"}.get(ex["family"], ex["family"])
    grid = np.linspace(max(ex["sample"].min() * 0.9, 1e-4), min(ex["sample"].max() * 1.1, 0.9999), 300)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ex["sample"], histnorm="probability density", marker_color="#E9E2D6",
                               marker_line=dict(color="white", width=1)))
    fig.add_trace(go.Scatter(x=grid, y=getattr(stats, dist).pdf(grid, *ex["params"]), mode="lines",
                             line=dict(color=COLOURS["SSI"], width=2), hoverinfo="skip"))
    fig.add_vline(x=ex["x"], line_color=ui.DRY_LINE, line_width=2)
    fig.update_layout(title=f"{scale}-month mean soil wetness in {calendar.month_name[month]}, every year "
                            f"({ex['family']} selected)")
    return _mini(fig, 300, "GWETROOT (0 to 1)", "density")


def _lag_plot(row: pd.Series):
    """Correlation of SPI(t) with SSI(t + k) for k = 0..3."""
    lags = [0, 1, 2, 3]
    r = [row[f"r SPI-SSI lag {k}"] for k in lags]
    best = int(row["best lag SPI-SSI (months)"])
    fig = go.Figure(go.Bar(x=[str(k) for k in lags], y=r, text=[f"{v:.2f}" for v in r], textposition="outside",
                           marker_color=[COLOURS["SSI"] if k == best else "#D9D2C5" for k in lags],
                           hovertemplate="lag %{x}: r = %{y:.3f}<extra></extra>"))
    fig.update_layout(title="Lag check: SSI shifted later by k months", yaxis=dict(range=[0, 1.05]),
                      xaxis=dict(type="category"))
    return _mini(fig, 300, "lag k (months)", "Pearson r")


# ============================================================================= steps

def step_data(district: str, years):
    """Step 1: the three NASA POWER inputs."""
    df = _inputs(district)
    st.markdown("**What it does.** Collects monthly rainfall, root-zone soil wetness and air temperature "
                "for every district, averaged over the NASA POWER grid cells inside the district boundary.")
    st.markdown("- **Rainfall**: PRECTOTCORR, converted from mm/day to a monthly total in mm.\n"
                "- **Soil moisture**: GWETROOT, the wet fraction of the root zone (0 = dry, 1 = saturated).\n"
                "- **Temperature**: TS, earth skin temperature in degC. This is the paper's \"ground temperature\": "
                "its range of 22.98 to 33.74 degC matches TS at the paper's point, while T2M (air at 2 m) stops at 31.41.\n"
                "- **Period**: January 1984 to December 2025, 504 months. District polygons: GADM 4.1.")
    st.latex(r"\bar{x}_{d,t} = \frac{1}{|G_d|}\sum_{g \in G_d} x_{g,t}, \qquad "
             r"G_d = \{\text{grid cells whose centre lies inside district } d\}")
    last = df.dropna().iloc[-1]
    ui.note(f"<b>Worked example, {district}, {_fmt_month(int(last['Year']), int(last['Month']))}:</b> rainfall "
            f"{last['prcp']:.1f} mm, soil wetness {last['gwetroot']:.3f}, TS {last['ts']:.2f} degC.")
    ui.show_chart(_data_plot(df, years), "Rainfall comes in two monsoon bursts each year; soil wetness follows "
                                         "it more smoothly; temperature has its own seasonal cycle.")


def step_spi(district: str, scale: int):
    """Step 2: SPI from k-month rainfall totals."""
    ex = _spi_example(district, scale)
    st.markdown("**What it does.** Says how unusual the rainfall of the last k months was, compared with "
                "the same calendar month in every other year, on a standard normal scale.")
    st.markdown("**(a) Aggregate** over k = 1, 3 or 6 months, then group by calendar month m:")
    st.latex(r"x_t^{(k)} = \sum_{j=0}^{k-1} P_{t-j}")
    st.markdown("**(b) Fit a Gamma distribution** to the ~42 values of each calendar month:")
    st.latex(r"g(x;\alpha,\beta) = \frac{x^{\alpha-1} e^{-x/\beta}}{\beta^{\alpha}\,\Gamma(\alpha)},\qquad "
             r"G(x) = \int_0^x g(s)\,ds")
    st.markdown("**(c) Maximum likelihood.** alpha solves the first equation; then beta = mean / alpha:")
    st.latex(r"\ln\hat\alpha - \psi(\hat\alpha) = A = \ln\bar{x} - \overline{\ln x},\qquad \hat\beta = \bar{x}/\hat\alpha")
    st.markdown("**(d) Zeros.** A month with no rain cannot come from a Gamma distribution, so the standard "
                "SPI uses a mixed distribution with q = share of zero months:")
    st.latex(r"H(x) = q + (1 - q)\,G(x)")
    ui.note(f"What this project's code does: it does not use the mixed form. Any zero total is replaced by "
            f"0.001 mm before the Gamma fit, so H(x) = G(x). District-mean totals are rarely zero "
            f"({ex['n_zero_all']} of {len(common.read_rainfall(district))} monthly totals in {district}; "
            f"{ex['n_zero']} of the {ex['n']} values in the worked example below), so the difference is small.",
            method=True)
    st.markdown("**(e) Standardise**, then classify (McKee et al. 1993):")
    st.latex(r"\text{SPI} = \Phi^{-1}\big(H(x)\big)")
    ui.note(f"<b>Worked example, {district}, SPI-{scale} in {_fmt_month(ex['year'], ex['month'])} (the driest on "
            f"record):</b> {scale}-month total x = {ex['x']:.1f} mm. The {ex['n']} values for "
            f"{calendar.month_name[ex['month']]} give A = {ex['A']:.4f}, alpha = {ex['alpha']:.3f}, "
            f"beta = {ex['beta']:.2f} mm. G(x) = {ex['F']:.5f}, so SPI = Phi^-1({ex['F']:.5f}) = "
            f"<b>{ex['spi']:.2f}</b> (dashboard value {ex['table_spi']:.2f}): {ui.class_of(ex['spi'])}.")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        ui.show_chart(_spi_plot(ex, scale), "The brown line sits far in the left tail: very few years were "
                                            "this dry in this season.")
    with c2:
        st.dataframe(pd.DataFrame([{"Index value": (f"{lo:g} to {hi:g}" if np.isfinite(lo) and np.isfinite(hi)
                                                    else (f">= {lo:g}" if np.isfinite(lo) else f"<= {hi:g}")),
                                    "Class": lab} for lo, hi, lab, _ in SPI_CATEGORIES]),
                     use_container_width=True, hide_index=True)
        st.caption("McKee classes, used for SPI, SSI and MSDI alike.")


def step_run_theory(district: str, scale: int):
    """Step 3: events, duration, severity and inter-arrival time."""
    model, spi_df, events = _event_model(district, scale)
    st.markdown("**What it does.** Cuts the index series into separate drought events: a drought starts when "
                "the index drops below the threshold X0 = 0 and ends when it comes back to 0 or above.")
    st.latex(r"D = t_e - t_s + 1 \ \text{(months)},\qquad S = \sum_{t=t_s}^{t_e} \lvert \text{SPI}_t \rvert")
    st.markdown("The **inter-arrival time** L is the time from the start of one drought to the start of the "
                "next. Its mean, used in the return periods, is estimated from the record as")
    st.latex(r"E(L) = \frac{\text{months in the record}}{\text{number of events}}")
    worst = events.loc[events["Severity"].idxmax()]
    starts = pd.to_datetime(dict(year=events["StartYear"], month=events["StartMonth"], day=1))
    gap = starts.diff().dt.days.dropna().mean() / 30.44
    ui.note(f"<b>Worked example, {district}, SPI-{scale}:</b> {model.n_events} events. The worst ran from "
            f"{_fmt_month(int(worst['StartYear']), int(worst['StartMonth']))} to "
            f"{_fmt_month(int(worst['EndYear']), int(worst['EndMonth']))}: D = {int(worst['Duration'])} months, "
            f"S = {worst['Severity']:.2f}. E(L) = {len(spi_df)} / {model.n_events} = {model.EL_months:.2f} months "
            f"(the average gap between event starts is {gap:.2f} months).")
    ui.show_chart(_event_plot(spi_df, worst, scale), "Every brown bar inside the shaded span adds its size to "
                                                     "the severity; the number of bars is the duration.")


def step_marginals(district: str, scale: int):
    """Step 4: Exponential duration and Gamma severity, with a goodness-of-fit check."""
    model, _, events = _event_model(district, scale)
    m = model.marginals
    st.markdown("**What it does.** Describes duration and severity each with a probability distribution "
                "(Shiau 2006): most droughts are short and mild, a few are long and severe.")
    st.latex(r"f_D(d) = \lambda e^{-\lambda d},\quad \hat\lambda = 1/\bar D \qquad\qquad "
             r"f_S(s) = \frac{s^{\alpha-1}e^{-s/\beta}}{\beta^\alpha\Gamma(\alpha)}")
    st.markdown("**Goodness of fit.** The Kolmogorov-Smirnov statistic is the largest gap between the "
                "empirical and the fitted CDF; a large p-value means the fit is not rejected.")
    st.latex(r"D_n = \sup_x \lvert F_n(x) - F(x) \rvert")
    ks_d = stats.kstest(events["Duration"], lambda x: stats.expon.cdf(x, scale=1 / m.lambda_d))
    ks_s = stats.kstest(events["Severity"], lambda x: stats.gamma.cdf(x, m.alpha_s, scale=m.beta_s))
    ui.note(f"<b>Worked example, {district}, SPI-{scale}:</b> mean duration {m.mean_D:.2f} months, so "
            f"lambda = {m.lambda_d:.4f} per month. Severity: alpha = {m.alpha_s:.3f}, beta = {m.beta_s:.3f}. "
            f"KS: duration D_n = {ks_d.statistic:.3f} (p = {ks_d.pvalue:.3f}), severity D_n = "
            f"{ks_s.statistic:.3f} (p = {ks_s.pvalue:.3f}). Duration is counted in whole months, so its KS "
            "test is only approximate.")
    ui.show_chart(_marginal_plot(model, events), "The curves should follow the bars: many short, mild events and "
                                                 "a long thin tail of big ones.")


def step_copulas(district: str, scale: int):
    """Step 5: Sklar's theorem, the three copulas, MLE, AIC, tau and tail dependence."""
    model, _, events = _event_model(district, scale)
    st.markdown("**What it does.** Joins the two distributions so that long droughts can also be severe. "
                "Sklar's theorem splits any joint distribution into its marginals and a copula C:")
    st.latex(r"F(d, s) = C\big(F_D(d), F_S(s)\big) = C(u, v)")
    st.latex(r"\text{Gumbel: } C = \exp\!\Big(-\big[(-\ln u)^\theta + (-\ln v)^\theta\big]^{1/\theta}\Big),\ \theta \ge 1")
    st.latex(r"\text{Clayton: } C = \big(u^{-\theta} + v^{-\theta} - 1\big)^{-1/\theta},\ \theta > 0")
    st.latex(r"\text{Frank: } C = -\tfrac{1}{\theta}\ln\!\Big(1 + \tfrac{(e^{-\theta u}-1)(e^{-\theta v}-1)}{e^{-\theta}-1}\Big)")
    st.markdown("**Fit and choose.** theta maximises the log-likelihood; the family with the lowest AIC wins. "
                "Kendall's tau measures the dependence without assuming any distribution; tail dependence "
                "says whether the extremes are linked more strongly than the middle.")
    st.latex(r"\ell(\theta) = \sum_i \ln c(u_i, v_i; \theta),\qquad \text{AIC} = 2k - 2\ell(\hat\theta)")
    st.latex(r"\tau = \frac{n_c - n_d}{\binom{n}{2}},\quad \tau_{\text{Gumbel}} = 1 - \tfrac{1}{\theta},\quad "
             r"\tau_{\text{Clayton}} = \tfrac{\theta}{\theta + 2},\quad \lambda_U^{\text{Gumbel}} = 2 - 2^{1/\theta},"
             r"\quad \lambda_L^{\text{Clayton}} = 2^{-1/\theta}")
    up, low = tail_dependence_coefs(model.best_family, model.best_theta)
    fits = ", ".join(f"{f.family} theta {f.parameter:.3f} AIC {f.aic:.1f}" for f in model.all_fits)
    ui.note(f"<b>Worked example, {district}, SPI-{scale}:</b> sample Kendall's tau = {model.kendall_tau:.3f}. "
            f"Fits: {fits}. Selected: <b>{model.best_family}</b> (lowest AIC), with upper-tail coefficient "
            f"{up:.3f} and lower-tail coefficient {low:.3f}.")
    ui.show_chart(_copula_plot(model, events), "The dots rise from bottom left to top right: longer events "
                                               "are more severe. The contours are the fitted copula.")


def step_return_periods(district: str, scale: int):
    """Step 6: T_AND and T_OR."""
    import events_panel
    model, _, events = _event_model(district, scale)
    m = model.marginals
    st.markdown("**What it does.** Turns the joint distribution into \"once every N years\".")
    st.latex(r"T_{\text{AND}}(d, s) = \frac{E(L)}{1 - F_D(d) - F_S(s) + C(F_D(d), F_S(s))},\qquad "
             r"T_{\text{OR}}(d, s) = \frac{E(L)}{1 - C(F_D(d), F_S(s))}")
    st.markdown("E(L) is in months, so the result is divided by 12 to give years. **T_AND** is for a drought "
                "at least d months long *and* at least s severe (rarer, the one that matters for damage); "
                "**T_OR** for one that reaches either threshold.")
    d, s = 6, round(float(np.quantile(events["Severity"], 0.9)), 1)
    u, v = float(m.F_D(d)), float(m.F_S(s))
    C = float(_copula_cdf(model.best_family, u, v, model.best_theta))
    ui.note(f"<b>Worked example, {district}, SPI-{scale}:</b> d = {d} months, s = {s} (90th percentile of "
            f"severity). u = F_D({d}) = {u:.4f}, v = F_S({s}) = {v:.4f}, C(u, v) = {C:.4f}. "
            f"T_AND = {model.EL_months:.2f} / {1 - u - v + C:.4f} / 12 = <b>{T_and(model, d, s):.1f} years</b>; "
            f"T_OR = {model.EL_months:.2f} / {1 - C:.4f} / 12 = <b>{T_or(model, d, s):.1f} years</b>.")
    ui.show_chart(events_panel.rp_curves(model), "Read across from the x-axis: a drought whose length and "
                                                 "severity are each a T-year event is rarer (T_AND) or more "
                                                 "common (T_OR) as a pair.")


def step_ssi(district: str, scale: int):
    """Step 7: SSI, the same standardisation on soil moisture."""
    spi_ex = _spi_example(district, scale)
    y, mth = spi_ex["year"], spi_ex["month"]
    ex = _ssi_example(district, scale, y, mth)
    st.markdown("**What it does.** The same standardisation as SPI, applied to root-zone soil wetness "
                "(Sanchez Alcalde & Escorihuela 2025). Two differences: soil moisture is a state, so the "
                "k months are **averaged**, not summed; and the distribution is chosen per calendar month "
                "from Beta, Gamma, Log-normal, Log-logistic, GEV and Normal by AIC.")
    st.latex(r"\bar\theta_t^{(k)} = \frac{1}{k}\sum_{j=0}^{k-1}\theta_{t-j},\qquad "
             r"\text{SSI} = \Phi^{-1}\big(F_m(\bar\theta_t^{(k)})\big)")
    st.markdown("A check that the distribution choice does not drive the result: the empirical SSI from "
                "Gringorten plotting positions (rank i of n) should correlate closely with the fitted one.")
    st.latex(r"p_i = \frac{i - 0.44}{n + 0.12},\qquad \text{SSI}^{\text{emp}} = \Phi^{-1}(p_i)")
    ui.note(f"<b>Worked example, {district}, SSI-{scale} in {_fmt_month(y, mth)}</b> (the month of the SPI "
            f"example): mean wetness {ex['x']:.3f}. The {ex['family']} distribution won for "
            f"{calendar.month_name[mth]} (AIC {ex['aic']:.1f}); F = {ex['F']:.4f}, so SSI = <b>{ex['ssi']:.2f}</b> "
            f"({ui.class_of(ex['ssi'])}). Robustness r = {ex['robust']:.3f}.")
    ui.show_chart(_ssi_plot(ex, scale, mth), "Same idea as the SPI plot: the further into the left tail, the "
                                             "drier the soil was for that season.")


def step_msdi(district: str, scale: int):
    """Step 8: MSDI step by step from the cached model, with the paper's equations."""
    import msdi_tabs
    model = msdi_tabs.get_model(district)
    T = scale if scale in M.T_SCALES else 3
    ex = _spi_example(district, scale)
    s = model.series
    i = s.index[(s["Year"] == ex["year"]) & (s["Month"] == ex["month"])][0]
    r = s.loc[i]
    last = s["MSDI1"].iloc[i - T + 1:i + 1]
    extreme, dry = TH.cutoff(T)
    st.markdown("**What it does.** Joins rainfall X and ground temperature Y into one index by the published method "
                "(Karunarathne, De Silva and Perera 2024, following Hao and AghaKouchak 2013).")
    st.markdown("**(a) Joint probability.** Rainfall and temperature of the same month go through a copula C:")
    st.latex(r"p = P(X \le x,\ Y \le y) = C\big(F_X(x), F_Y(y)\big)")
    st.markdown("**(b) MSDI**, capped at +/-5 as the paper does:")
    st.latex(r"\text{MSDI} = \Phi^{-1}(p)")
    st.markdown("**(c) Prolonged periods.** The T-month MSDI averages the last T one-month values:")
    st.latex(r"\text{T-month MSDI} = \frac{1}{T}\sum_{k=1}^{T} \text{1-month MSDI}_k,\qquad T \in \{3, 6, 9, 12\}")
    st.markdown("**Marginals, one distribution for all months.** X is the monthly average rainfall in mm/day, zero months "
                "set to 1e-10: the lowest AIC of four-parameter Beta, Gamma and Half-Gaussian (paper Table 2). Y is a "
                "two-component Gaussian mixture, fitted by EM from a k-means start:")
    st.latex(r"F_Y(y) = \sum_{i=1}^{2} w_i\,\Phi\!\Big(\frac{y-\mu_i}{\sigma_i}\Big)")
    st.markdown("**Copula.** The lowest AIC of Frank, FGM and AMH (paper Table 3), plus Gaussian, Gumbel and Clayton as "
                "extra candidates, among the families whose range includes the sample Kendall's tau:")
    st.latex(r"\text{FGM: } C = uv\,[1 + \theta(1-u)(1-v)],\qquad \text{AMH: } C = \frac{uv}{1-\theta(1-u)(1-v)}")
    ui.note(f"<b>Worked example, {district}, {_fmt_month(ex['year'], ex['month'])}:</b> rain x = {r['rain']:.2f} mm/day "
            f"({model.rain.family}), u = F_X(x) = {r['u']:.4f}; temperature y = {r['temp']:.2f} degC, v = F_Y(y) = "
            f"{r['v']:.4f}. {model.best_family} copula (theta {model.best_theta:.3f}): p = C(u, v) = {r['p']:.4f}, so "
            f"MSDI = Phi^-1({r['p']:.4f}) = <b>{r['MSDI1']:.2f}</b>. The last {T} one-month values are "
            f"{', '.join(f'{v:.2f}' for v in last)}, so MSDI-{T} = <b>{r[f'MSDI{T}']:.2f}</b>. "
            f"With extremely dry at or below {extreme:g} and dry at or below {dry:g}, this month is "
            f"<b>{TH.classify(pd.Series([r[f'MSDI{T}']]), extreme, dry).iloc[0]}</b>.")
    ui.show_chart(msdi_tabs.history_chart(district, T, ui.selection()["years"]),
                  "The dashed lines are the paper's Table 4 cut-offs for this T.")


def step_thresholds():
    """Step 9: the class cut-offs from the paper's Table 4."""
    st.markdown("**What it does.** The MSDI is not standardised, so it cannot use the SPI classes. The paper gives two "
                "cut-offs for each T (Table 4). The same values are used for every district.")
    st.latex(r"\text{Extremely dry: } \text{MSDI}_T \le c_{E,T},\qquad \text{Dry: } c_{E,T} < \text{MSDI}_T \le c_{D,T},"
             r"\qquad \text{Normal: } \text{MSDI}_T > c_{D,T}")
    st.dataframe(pd.DataFrame([{"T": T, "Extremely dry at or below": e, "Dry at or below": d} for T, (e, d) in TH.TABLE4.items()]),
                 use_container_width=True, hide_index=True)
    st.markdown("A drought event is an unbroken run of months at or below the dry cut-off. Its severity is the sum of "
                "the amount below the cut-off.")


def step_compare(district: str, scale: int):
    """Step 10: how the three indices are compared and how to read the results."""
    comp = PX.comparison(scale)
    row = comp[comp["district"] == district].iloc[0]
    st.markdown("**What it does.** Measures how far the three indices tell the same story.")
    st.latex(r"r = \frac{\sum (a_t - \bar a)(b_t - \bar b)}{\sqrt{\sum (a_t-\bar a)^2 \sum (b_t - \bar b)^2}},\qquad "
             r"r(k) = \operatorname{corr}\big(a_t,\ b_{t+k}\big),\ k = 0..3")
    st.markdown("A month is **dry** at or below -1 for SPI and SSI, and at or below the Table 4 dry cut-off for MSDI. "
                "Spearman's rho is the same as Pearson's r on ranks. Dry-month overlap counts the months both indices "
                "flag, and the months only one of them flags.")
    lag = int(row["best lag SPI-SSI (months)"])
    ui.note(f"<b>Worked example, {district}, scale {scale}:</b> SPI-SSI r = {row['pearson SPI-SSI']:.2f} "
            f"(Spearman {row['spearman SPI-SSI']:.2f}), best lag {lag} month(s); SPI-MSDI r = "
            f"{row['pearson SPI-MSDI']:.2f}. Dry months: both SPI and SSI {int(row['dry both SPI-SSI'])}, only SPI "
            f"{int(row['dry only SPI in SPI-SSI'])}, only SSI {int(row['dry only SSI in SPI-SSI'])}.")
    st.markdown("**How to read it.** A high r with a lag of 1 month means soil moisture follows rain with a "
                "short delay. MSDI correlates weakly with SPI, partly because of the seasonal cycle it keeps, so a "
                "difference between them is not simply added heat information.")
    ui.show_chart(_lag_plot(row), "The highlighted bar is the delay at which soil moisture matches rainfall best.")


def step_limitations():
    """Step 11: limitations."""
    st.markdown(
        "- **Shared grid cells.** NASA POWER cells are about 50 km wide. Six pairs of small districts fall in "
        "the same cell and have identical series: Batticaloa/Polonnaruwa, Colombo/Gampaha, Jaffna/Kilinochchi, "
        "Kandy/Matale, Kegalle/Nuwara Eliya, Mullaitivu/Vavuniya.\n"
        "- **One set of MSDI cut-offs.** Table 4 was set on North Central Province and is used here for every "
        "district. msdi_outputs/common_threshold_shares.csv gives the share of months it flags.\n"
        "- **SPI and SSI classes.** The McKee cut-offs are used everywhere and have not been checked against crop "
        "losses.\n"
        "- **Copula tests often reject.** The copula is chosen by AIC and the test is reported only; it rejects the "
        "selected family in most districts.\n"
        "- **Model data, not gauges.** SPI and SSI both come from the NASA POWER system, so part of their "
        "agreement is built in. The data are retrospective (2-3 months behind), not a forecast.\n"
        "- **Short record.** Return periods beyond about 40 years go past the 1984-2025 record.")


GLOSSARY = [
    ("Accumulation window (scale) k", "Number of months summed (rain) or averaged (soil, temperature) before standardising."),
    ("AIC", "Akaike information criterion, 2k - 2 log-likelihood. Lower is better; it penalises extra parameters."),
    ("Copula", "A function that joins two marginal distributions into a joint one and holds only their dependence."),
    ("CDF", "Cumulative distribution function F(x) = P(X <= x)."),
    ("Drought event", "A continuous run of months with the index below 0 (run theory)."),
    ("Duration D", "Length of a drought event in months."),
    ("E(L)", "Mean inter-arrival time between drought events, in months."),
    ("GWETROOT", "NASA POWER root-zone soil wetness, 0 (dry) to 1 (saturated)."),
    ("Cut-off", "The MSDI value at or below which a month is Dry or Extremely dry: the paper's Table 4, one pair for each T."),
    ("Ground temperature", "NASA POWER earth skin temperature, TS, used by the MSDI."),
    ("Kendall's tau", "Rank correlation: (concordant pairs - discordant pairs) / all pairs."),
    ("Maha / Yala", "The two cultivation seasons: Maha September to March, Yala May to August."),
    ("Marginal", "The distribution of one variable on its own."),
    ("MLE", "Maximum likelihood estimation: the parameters that make the observed data most probable."),
    ("Phi^-1", "Inverse of the standard normal CDF; turns a probability into a z-score."),
    ("Prolonged period T", "Number of one-month MSDI values averaged: 3, 6, 9 or 12."),
    ("Return period T", "Average number of years between droughts at least this big."),
    ("Severity S", "Sum of |index| over the months of a drought event."),
    ("T2M", "NASA POWER air temperature at 2 m. The MSDI can be refitted with it, but uses TS."),
    ("Tail dependence", "How strongly the extremes of two variables occur together."),
]


def _how_to_use():
    """Short guide to the controls and pages."""
    st.markdown(
        "- **Sidebar**: pick the district, the index for the Overview map, the time window (1, 3 or 6 months) "
        "and the years shown in the time-series charts. Every page follows these choices.\n"
        "- **Overview**: national status first (cards and district tiles), then the map, then details.\n"
        "- **Rainfall / Soil moisture / Rainfall + heat**: one page per index, each with a national map, a "
        "district analysis and the drought events. Rainfall also shows return periods.\n"
        "- **Compare indices**: SPI, SSI and MSDI together.\n"
        "- **Colours**: brown = drier than normal, light = near normal, blue = wetter. SPI and SSI share the classes "
        "in the sidebar; MSDI has three of its own, from the paper's Table 4. Hover any chart for exact "
        "values.")


def page_learn():
    """The Learn page: how to use, the method steps and a glossary."""
    sel = ui.selection()
    district, scale = sel["district"], sel["scale"]
    ui.page_header("Learn and how to use",
                   f"Every step of the method with its equation, a worked example and a plot. Examples use the "
                   f"sidebar district and window: {district}, {scale} month(s).")
    with st.expander("How to use this dashboard", expanded=True):
        _how_to_use()
    steps = [
        ("1. Data: NASA POWER rainfall, soil moisture and temperature, 1984-2025", lambda: step_data(district, sel["years"])),
        ("2. SPI: the Standardized Precipitation Index", lambda: step_spi(district, scale)),
        ("3. Run theory: drought events, duration, severity, inter-arrival time", lambda: step_run_theory(district, scale)),
        ("4. Marginal distributions of duration and severity", lambda: step_marginals(district, scale)),
        ("5. Copulas: joining duration and severity", lambda: step_copulas(district, scale)),
        ("6. Joint return periods T_AND and T_OR", lambda: step_return_periods(district, scale)),
        ("7. SSI: the same standardisation for soil moisture", lambda: step_ssi(district, scale)),
        ("8. MSDI step by step: the published method", lambda: step_msdi(district, scale)),
        ("9. MSDI classes: the paper's Table 4 cut-offs", step_thresholds),
        ("10. Comparing the three indices, and how to read the results", lambda: step_compare(district, scale)),
        ("11. Limitations", step_limitations),
    ]
    for title, body in steps:
        with st.expander(title):
            body()
    with st.expander("Glossary"):
        st.dataframe(pd.DataFrame(GLOSSARY, columns=["Term", "Meaning"]), use_container_width=True, hide_index=True,
                     height=36 * (len(GLOSSARY) + 1) + 4)
    with st.expander("References"):
        st.markdown(
            "- McKee, T. B., Doesken, N. J. & Kleist, J. (1993). The relationship of drought frequency and duration "
            "to time scales. 8th Conference on Applied Climatology.\n"
            "- Shiau, J.-T. (2006). Fitting drought duration and severity with two-dimensional copulas. Water "
            "Resources Management 20, 795-815.\n"
            "- Hao, Z. & AghaKouchak, A. (2013). Multivariate Standardized Drought Index. Advances in Water "
            "Resources 57, 12-18.\n"
            "- Karunarathne, S., De Silva, K. & Perera, S. (2024). Modeling and indexing drought severity with "
            "multi-modal ground temperature data. Environmental and Ecological Statistics 31, 707-723.\n"
            "- Sanchez Alcalde, G. & Escorihuela, M. J. (2025). Remote sensing standardized soil moisture index for "
            "drought monitoring. Remote Sensing 17, 3916.\n"
            "- Genest, C., Remillard, B. & Beaudoin, D. (2009). Goodness-of-fit tests for copulas. Insurance: "
            "Mathematics and Economics 44, 199-213.\n"
            "- Nelsen, R. B. (2006). An Introduction to Copulas, 2nd ed. Springer.\n"
            "- Sklar, A. (1959). Fonctions de repartition a n dimensions et leurs marges. Publ. Inst. Statist. "
            "Univ. Paris 8, 229-231.")
