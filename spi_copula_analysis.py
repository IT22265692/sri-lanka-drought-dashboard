"""
spi_copula_analysis.py
======================
Standardized Precipitation Index (SPI), run-theory drought events, the
duration-severity copula and joint return periods.

Where it sits in the pipeline:

    common.read_rainfall  ->  compute_spi            (SPI-1, SPI-3, SPI-6)
                          ->  extract_drought_events (duration D, severity S)
                          ->  model_from_events      (D ~ Exponential, S ~ Gamma,
                                                      Gumbel / Clayton / Frank by AIC)
                          ->  T_and, T_or, return_period_table

The event, copula and return-period part is shared: SSI and MSDI events are
analysed with exactly the same functions (see index_data.events_and_model).

References: McKee, Doesken and Kleist (1993) for SPI; Shiau (2006) for the
duration-severity copula and joint return periods; Nelsen (2006) for the
copula formulas.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats


# =============================================================================
# 1. SPI COMPUTATION
# =============================================================================

def compute_spi(df: pd.DataFrame, scale: int = 1, precip_col: str = "prcp") -> pd.DataFrame:
    """SPI at one time scale from monthly rainfall (McKee et al. 1993).

    df: monthly table with Year, Month and precip_col (mm). scale: 1, 3 or 6.
    Returns df sorted by date with a new column SPI<scale>. Each calendar
    month gets its own Gamma fit, because a normal January and a normal July
    are very different amounts of rain.
    """
    df = df.sort_values(["Year", "Month"]).reset_index(drop=True).copy()
    p = df[precip_col].astype(float).values

    # Step 1: k-month total ending at each month (the first k - 1 months are NaN)
    if scale == 1:
        rolled = p.copy()
    else:
        rolled = pd.Series(p).rolling(window=scale, min_periods=scale).sum().values

    spi_col = f"SPI{scale}"
    df[spi_col] = np.nan

    for month in range(1, 13):
        mask = (df["Month"] == month) & pd.notna(rolled)
        x = rolled[mask].copy()
        if len(x) < 10:
            continue
        # Zero totals are set to 0.001 mm so the Gamma fit (support x > 0) works.
        # District-mean monthly totals are almost never exactly 0 in this data.
        x[x <= 0] = 0.001
        # Step 2: Gamma shape and scale by maximum likelihood, location fixed at 0
        shape, loc, scale_g = stats.gamma.fit(x, floc=0)
        # Step 3: SPI = Phi^-1(F_Gamma(x)); F is kept inside (1e-4, 1 - 1e-4),
        # so SPI lies between -3.72 and +3.72
        cdf = stats.gamma.cdf(np.clip(rolled[mask], 0.001, None), a=shape, scale=scale_g)
        cdf = np.clip(cdf, 1e-4, 1 - 1e-4)
        df.loc[mask, spi_col] = stats.norm.ppf(cdf)

    return df


def estimate_spi_from_input(precip_df: pd.DataFrame, scale: int, year: int, month: int,
                            rainfall_mm: float, precip_col: str = "prcp") -> float | None:
    """'What SPI would this reading give?' calculator.

    Puts rainfall_mm into the district's series at (year, month), replacing
    the value there or adding a new month, recomputes SPI at the requested
    scale and returns the value at that month (None if it cannot be computed).
    For SPI-3 and SPI-6 the other months of the window are the real data.
    Refitting the Gamma with one value changed has a negligible effect on the
    climatology, so this matches using the existing fit.
    """
    df = precip_df.copy()
    mask = (df["Year"] == year) & (df["Month"] == month)
    if mask.any():
        df.loc[mask, precip_col] = rainfall_mm
    else:
        new_row = {"Year": year, "Month": month, precip_col: rainfall_mm}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    spi_df = compute_spi(df, scale=scale, precip_col=precip_col)
    spi_col = f"SPI{scale}"
    match = spi_df[(spi_df["Year"] == year) & (spi_df["Month"] == month)]
    if len(match) == 0 or pd.isna(match[spi_col].iloc[0]):
        return None
    return float(match[spi_col].iloc[0])


# =============================================================================
# 2. DROUGHT EVENT EXTRACTION  (run theory, threshold 0)
# =============================================================================

def extract_drought_events(df: pd.DataFrame, spi_col: str) -> pd.DataFrame:
    """Run theory: a drought event is a continuous run of months with index < 0.

    df: table with Year, Month and spi_col (any standardised index).
    Returns one row per event: StartYear, StartMonth, EndYear, EndMonth,
    Duration D (months) and Severity S = sum of |index| over the run, rounded
    to 4 decimals. A missing value ends a run.
    """
    events = []
    in_drought = False
    duration = 0
    severity = 0.0
    start_idx = None
    df = df.reset_index(drop=True)

    for i, row in df.iterrows():
        spi = row[spi_col]
        if pd.notna(spi) and spi < 0:
            if not in_drought:
                in_drought = True
                duration, severity, start_idx = 1, abs(spi), i
            else:
                duration += 1
                severity += abs(spi)
        else:
            if in_drought:
                events.append({
                    "StartYear": int(df.loc[start_idx, "Year"]),
                    "StartMonth": int(df.loc[start_idx, "Month"]),
                    "EndYear": int(df.loc[i - 1, "Year"]),
                    "EndMonth": int(df.loc[i - 1, "Month"]),
                    "Duration": duration,
                    "Severity": round(severity, 4),
                })
                in_drought = False

    # A run still open at the end of the record is kept as an event
    if in_drought:
        last = df.index[-1]
        events.append({
            "StartYear": int(df.loc[start_idx, "Year"]),
            "StartMonth": int(df.loc[start_idx, "Month"]),
            "EndYear": int(df.loc[last, "Year"]),
            "EndMonth": int(df.loc[last, "Month"]),
            "Duration": duration,
            "Severity": round(severity, 4),
        })

    return pd.DataFrame(events)


# =============================================================================
# 3. MARGINAL FITS  (Exponential for duration, Gamma for severity)
# =============================================================================

@dataclass
class Marginals:
    """Fitted distributions of drought duration D and severity S.

    D ~ Exponential(rate lambda_d), S ~ Gamma(shape alpha_s, scale beta_s).
    F_* are CDFs, f_* densities and q_* quantile functions.
    """
    lambda_d: float
    alpha_s: float
    beta_s: float
    mean_D: float
    mean_S: float
    n_events: int

    def F_D(self, d):
        """P(D <= d), kept inside (1e-9, 1 - 1e-9) for the copula."""
        return np.clip(stats.expon.cdf(d, scale=1 / self.lambda_d), 1e-9, 1 - 1e-9)

    def F_S(self, s):
        """P(S <= s), kept inside (1e-9, 1 - 1e-9) for the copula."""
        return np.clip(stats.gamma.cdf(s, a=self.alpha_s, scale=self.beta_s), 1e-9, 1 - 1e-9)

    def f_D(self, d):
        """Exponential density of D."""
        return stats.expon.pdf(d, scale=1 / self.lambda_d)

    def f_S(self, s):
        """Gamma density of S."""
        return stats.gamma.pdf(s, a=self.alpha_s, scale=self.beta_s)

    def q_D(self, p):
        """Duration with non-exceedance probability p."""
        return stats.expon.ppf(p, scale=1 / self.lambda_d)

    def q_S(self, p):
        """Severity with non-exceedance probability p."""
        return stats.gamma.ppf(p, a=self.alpha_s, scale=self.beta_s)


def fit_marginals(events: pd.DataFrame) -> Marginals:
    """Fit D ~ Exponential and S ~ Gamma to an events table, both by maximum likelihood.

    The Exponential MLE is simply lambda = 1 / mean(D). Gamma uses location 0.
    """
    D = events["Duration"].astype(float).values
    S = events["Severity"].astype(float).values
    lam_d = 1.0 / D.mean()
    shape_s, _, scale_s = stats.gamma.fit(S, floc=0)
    return Marginals(
        lambda_d=float(lam_d), alpha_s=float(shape_s), beta_s=float(scale_s),
        mean_D=float(D.mean()), mean_S=float(S.mean()), n_events=len(D),
    )


# =============================================================================
# 4. COPULA FAMILIES  (Gumbel-Hougaard, Clayton, Frank)
# =============================================================================
#
#   Gumbel-Hougaard  upper-tail dependence (long and severe droughts together)
#   Clayton          lower-tail dependence
#   Frank            no tail dependence
#
# A Gaussian copula was tried during development and never won on AIC, so it
# was dropped. The MSDI module reuses these formulas.

def _clip01(x, eps=1e-6):
    """Keep probabilities inside (eps, 1 - eps) so the logs stay finite."""
    return np.clip(x, eps, 1 - eps)


def _gumbel_log_density(u, v, theta):
    """log c(u, v) of the Gumbel-Hougaard copula, theta >= 1."""
    if theta < 1.0:
        return -np.inf
    u, v = _clip01(u), _clip01(v)
    lu, lv = -np.log(u), -np.log(v)
    A = lu ** theta + lv ** theta
    A_pow = A ** (1.0 / theta)
    log_c = (
        -A_pow
        + (theta - 1) * (np.log(lu) + np.log(lv))
        + (1 - 2 * theta) / theta * np.log(A)
        + np.log(A_pow + theta - 1)
        + lu + lv
    )
    return log_c


def _gumbel_cdf(u, v, theta):
    """Gumbel copula, C = exp(-[(-ln u)^theta + (-ln v)^theta]^(1/theta))."""
    u, v = _clip01(u), _clip01(v)
    return np.exp(-(((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta)))


def _clayton_log_density(u, v, theta):
    """log c(u, v) of the Clayton copula, theta > 0."""
    if theta <= 0:
        return -np.inf
    u, v = _clip01(u), _clip01(v)
    return (
        np.log(1 + theta)
        + (-1 - theta) * (np.log(u) + np.log(v))
        + (-1 / theta - 2) * np.log(u ** -theta + v ** -theta - 1)
    )


def _clayton_cdf(u, v, theta):
    """Clayton copula, C = (u^-theta + v^-theta - 1)^(-1/theta)."""
    u, v = _clip01(u), _clip01(v)
    return (u ** -theta + v ** -theta - 1) ** (-1 / theta)


def _frank_log_density(u, v, theta):
    """log c(u, v) of the Frank copula, theta != 0."""
    if abs(theta) < 1e-6:
        return -np.inf
    u, v = _clip01(u), _clip01(v)
    eu = np.exp(-theta * u)
    ev = np.exp(-theta * v)
    e1 = np.exp(-theta)
    num = theta * (1 - e1) * eu * ev
    den = ((1 - e1) - (1 - eu) * (1 - ev)) ** 2
    return np.log(num) - np.log(den)


def _frank_cdf(u, v, theta):
    """Frank copula, C = -(1/theta) ln[1 + (e^(-theta u) - 1)(e^(-theta v) - 1) / (e^(-theta) - 1)]."""
    u, v = _clip01(u), _clip01(v)
    return -1 / theta * np.log(
        1 + (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1) / (np.exp(-theta) - 1)
    )


@dataclass
class CopulaFit:
    """Maximum likelihood fit of one copula family to the (u, v) pairs."""
    family: str
    parameter: float
    loglik: float
    aic: float
    tail_upper: bool
    tail_lower: bool


def _fit_family(u, v, family: str, tau_hat: float) -> CopulaFit:
    """MLE of one family's theta with L-BFGS-B, started from the tau-inversion value.

    AIC = -2 log L + 2 (one parameter).
    """
    if family == "Gumbel":
        # tau = 1 - 1/theta  ->  theta = 1 / (1 - tau)
        init = max(1.05, 1.0 / (1.0 - tau_hat)) if tau_hat < 0.99 else 5.0
        bounds = [(1.001, 20)]
        neg_ll = lambda p: -_gumbel_log_density(u, v, p[0]).sum()
        up, low = True, False
    elif family == "Clayton":
        # tau = theta / (theta + 2)  ->  theta = 2 tau / (1 - tau)
        init = max(0.05, 2 * tau_hat / (1 - tau_hat)) if tau_hat < 0.99 else 5.0
        bounds = [(0.001, 20)]
        neg_ll = lambda p: -_clayton_log_density(u, v, p[0]).sum()
        up, low = False, True
    elif family == "Frank":
        init = 5.0
        bounds = [(0.001, 50)]
        neg_ll = lambda p: -_frank_log_density(u, v, p[0]).sum()
        up, low = False, False
    else:
        raise ValueError(family)

    res = optimize.minimize(neg_ll, x0=[init], bounds=bounds, method="L-BFGS-B")
    theta_hat = float(res.x[0])
    ll = -float(res.fun)
    aic = -2 * ll + 2
    return CopulaFit(family=family, parameter=theta_hat, loglik=ll, aic=aic,
                     tail_upper=up, tail_lower=low)


def _copula_cdf(family: str, u, v, theta) -> float | np.ndarray:
    """C(u, v) for Gumbel, Clayton or Frank."""
    if family == "Gumbel":
        return _gumbel_cdf(u, v, theta)
    if family == "Clayton":
        return _clayton_cdf(u, v, theta)
    if family == "Frank":
        return _frank_cdf(u, v, theta)
    raise ValueError(family)


def _copula_log_density(family: str, u, v, theta):
    """log c(u, v) for Gumbel, Clayton or Frank."""
    if family == "Gumbel":
        return _gumbel_log_density(u, v, theta)
    if family == "Clayton":
        return _clayton_log_density(u, v, theta)
    if family == "Frank":
        return _frank_log_density(u, v, theta)
    raise ValueError(family)


COPULA_FAMILIES = ["Gumbel", "Clayton", "Frank"]


# =============================================================================
# 5. FULL PIPELINE:  events -> marginals -> best copula
# =============================================================================

@dataclass
class DistrictModel:
    """Duration-severity model for one district, index and scale.

    EL_months is the mean inter-arrival time E(L) between events, in months,
    estimated as (months in the record) / (number of events).
    """
    district: str
    spi_scale: int
    n_events: int
    kendall_tau: float
    spearman_rho: float
    marginals: Marginals
    all_fits: list[CopulaFit] = field(default_factory=list)
    best_family: str = ""
    best_theta: float = 0.0
    EL_months: float = 0.0

    def to_dict(self) -> dict:
        """Plain-dict copy, e.g. for saving to JSON."""
        return {
            "district": self.district, "spi_scale": self.spi_scale,
            "n_events": self.n_events, "kendall_tau": self.kendall_tau,
            "spearman_rho": self.spearman_rho, "EL_months": self.EL_months,
            "best_family": self.best_family, "best_theta": self.best_theta,
            "marginals": asdict(self.marginals),
            "all_fits": [asdict(f) for f in self.all_fits],
        }


def build_district_model(precip_df: pd.DataFrame, district: str, spi_scale: int):
    """SPI -> events -> duration-severity model for one district.

    Returns (DistrictModel, SPI table, events table).
    """
    spi_df = compute_spi(precip_df, scale=spi_scale)
    spi_col = f"SPI{spi_scale}"
    events = extract_drought_events(spi_df, spi_col=spi_col)
    model = model_from_events(events, district, spi_scale, n_months=len(precip_df))
    return model, spi_df, events


def model_from_events(events: pd.DataFrame, district: str, spi_scale: int,
                      n_months: int) -> DistrictModel:
    """Marginals, best copula (lowest AIC) and E(L) from a table of drought events.

    Shared by SPI, SSI and MSDI. Needs at least 10 events (raises ValueError).
    """
    if len(events) < 10:
        raise ValueError(f"Only {len(events)} events for {district} scale {spi_scale}; need >=10 for copula fit")

    marg = fit_marginals(events)
    tau = float(stats.kendalltau(events["Duration"], events["Severity"]).statistic)
    rho = float(stats.spearmanr(events["Duration"], events["Severity"]).statistic)

    # Sklar: put each variable on (0, 1) through its own fitted CDF
    u = marg.F_D(events["Duration"].values)
    v = marg.F_S(events["Severity"].values)

    fits = []
    for fam in COPULA_FAMILIES:
        try:
            fits.append(_fit_family(u, v, fam, tau_hat=tau))
        except Exception as e:
            print(f"  {fam} fit failed for {district} SPI-{spi_scale}: {e}")

    best = min(fits, key=lambda f: f.aic)
    EL = n_months / len(events)

    return DistrictModel(
        district=district, spi_scale=spi_scale, n_events=len(events),
        kendall_tau=tau, spearman_rho=rho, marginals=marg,
        all_fits=fits, best_family=best.family, best_theta=best.parameter,
        EL_months=EL,
    )


# =============================================================================
# 6. JOINT RETURN PERIODS  (Shiau 2006)
# =============================================================================
#
#   AND : P(D >= d AND S >= s) = 1 - F_D(d) - F_S(s) + C(F_D(d), F_S(s))
#   OR  : P(D >= d OR  S >= s) = 1 - C(F_D(d), F_S(s))
#   T   = E(L) / P / 12        (E(L) in months, T in years)

def joint_prob_and(model: DistrictModel, d, s) -> float:
    """P(D >= d AND S >= s), floored at 1e-9."""
    u = model.marginals.F_D(d)
    v = model.marginals.F_S(s)
    C = _copula_cdf(model.best_family, u, v, model.best_theta)
    return max(1 - u - v + float(C), 1e-9)


def joint_prob_or(model: DistrictModel, d, s) -> float:
    """P(D >= d OR S >= s), floored at 1e-9."""
    u = model.marginals.F_D(d)
    v = model.marginals.F_S(s)
    C = _copula_cdf(model.best_family, u, v, model.best_theta)
    return max(1 - float(C), 1e-9)


def T_and(model: DistrictModel, d, s) -> float:
    """Return period in years of a drought at least d months long AND at least s severe."""
    return model.EL_months / joint_prob_and(model, d, s) / 12.0


def T_or(model: DistrictModel, d, s) -> float:
    """Return period in years of a drought at least d months long OR at least s severe."""
    return model.EL_months / joint_prob_or(model, d, s) / 12.0


def return_period_table(model: DistrictModel, periods=(2, 5, 10, 20, 50, 100)) -> pd.DataFrame:
    """Design droughts for standard return periods, with their joint T_AND and T_OR.

    For each T, d and s are the duration and severity whose own (univariate)
    return period is T, i.e. exceedance probability E(L) / (12 T).
    """
    rows = []
    for T in periods:
        p_exc = model.EL_months / (T * 12)
        d = model.marginals.q_D(1 - p_exc)
        s = model.marginals.q_S(1 - p_exc)
        rows.append({
            "Return period (years)": T,
            "Drought length (months)": round(d, 1),
            "Drought severity score": round(s, 2),
            "T_AND(d,s) years": round(T_and(model, d, s), 1),
            "T_OR(d,s) years": round(T_or(model, d, s), 1),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 7. TAIL DEPENDENCE COEFFICIENTS
# =============================================================================

def tail_dependence_coefs(family: str, theta: float) -> tuple[float, float]:
    """(upper, lower) tail dependence: Gumbel 2 - 2^(1/theta), Clayton 2^(-1/theta), Frank 0."""
    if family == "Gumbel":
        return (2 - 2 ** (1 / theta), 0.0)
    if family == "Clayton":
        return (0.0, 2 ** (-1 / theta))
    return (0.0, 0.0)


# =============================================================================
# 8. DROUGHT CLASSES  (McKee et al. 1993, shared by SPI, SSI and MSDI)
# =============================================================================

# (lower bound, upper bound, label, colour). Colours: brown (dry) to blue (wet),
# a colour-blind safe diverging scheme with a light neutral middle.
SPI_CATEGORIES = [
    (2.0,  float("inf"), "Extremely wet",   "#08306B"),
    (1.5,  2.0,           "Very wet",        "#2171B5"),
    (1.0,  1.5,           "Moderately wet",  "#6BAED6"),
    (-1.0, 1.0,           "Near normal",     "#EDE8DF"),
    (-1.5, -1.0,          "Moderately dry",  "#F4A259"),
    (-2.0, -1.5,          "Severely dry",    "#C4561D"),
    (float("-inf"), -2.0, "Extremely dry",   "#7A2E0B"),
]


def spi_category(value: float) -> tuple[str, str, str]:
    """(label, unused, colour) for an index value, McKee classes."""
    if pd.isna(value):
        return ("No data", "", "#CCCCCC")
    for lo, hi, label, colour in SPI_CATEGORIES:
        if lo <= value < hi or (hi == float("inf") and value >= lo):
            return (label, "", colour)
    return ("No data", "", "#CCCCCC")


def spi_plain_english(value: float) -> str:
    """One-sentence explanation of what an index value means to a farmer."""
    if pd.isna(value):
        return "No rainfall data available for this month."
    if value >= 1.5:
        return "Much wetter than usual for this district. Good for water-sensitive crops; watch for flooding on low ground."
    if value >= 0.5:
        return "Wetter than usual. Rainfall is above average."
    if value > -0.5:
        return "Close to the normal level for this district and this time of year."
    if value >= -1.0:
        return "Slightly drier than usual. Nothing alarming yet, but keep monitoring if it continues."
    if value >= -1.5:
        return "Noticeably drier than usual. Consider conserving irrigation water and delaying non-essential planting."
    if value >= -2.0:
        return "Severely dry. Crop stress is likely; prioritise water for high-value crops."
    return "Extremely dry, among the driest 2-3% of months on record for this district. Serious risk to unirrigated crops."
