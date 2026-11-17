"""
model_checks.py
===============
Checks and comparisons that sit beside the main method: they do not change any
index, they show why the method's choices were made. Used by the notebooks in
notebooks/.

Where it sits in the pipeline:

    spi_copula_analysis / ssi_analysis / msdi_analysis   (the method)
        -> model_checks.py                               (this module: comparisons)
        -> notebooks/01, 02, 03 and the MSDI page

    fit_families            several candidate distributions by MLE, with AIC, BIC and KS
    ds_copula_table         duration-severity copulas: the SPI model's fits plus BIC,
                            tail dependence and a Cramer-von Mises bootstrap p-value
    msdi_copula_table       the MSDI model's six copula fits plus BIC, tau range, tails
    empirical_copula_grid   C_n on a grid, for contour plots
    rescore_msdi            MSDI recomputed with another copula family (sensitivity check)
    direction_check         does the index score dry and hot months lowest?
    seasonal_check          does the index follow the calendar?
    seasonal_share          how much of the index's variance is the calendar month?
    without_seasons         the index with each calendar month's mean removed
    cross_correlation       corr(a_t, b_{t+k}) over a range of lags
    dry_overlap             2 x 2 table of dry / not dry months for two indices
"""
from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import msdi_analysis as M
from spi_copula_analysis import tail_dependence_coefs

TAU_FIGURE = Path("validation") / "random_subset_tau.png"

# scipy name and a readable name of each candidate family
FAMILY_NAMES = {"gamma": "Gamma", "weibull_min": "Weibull", "lognorm": "Log-normal", "expon": "Exponential",
                "beta": "Beta", "genextreme": "GEV", "norm": "Normal"}


# =============================================================================
# 1. CANDIDATE DISTRIBUTIONS
# =============================================================================

def _fit_one(name: str, x: np.ndarray, beta_support: str):
    """MLE fit of one scipy family. Returns (frozen distribution, number of free parameters).

    Positive families (Gamma, Weibull, Log-normal, Exponential) have location
    fixed at 0. Beta uses support (0, 1) for soil wetness ("unit") or
    (0, max * (n + 1) / n) for rainfall ("zero_max", as in msdi_analysis).
    GEV and Normal are fitted with free location.
    """
    dist = getattr(stats, name)
    if name in ("gamma", "weibull_min", "lognorm"):
        params = dist.fit(x, floc=0)
        k = 2
    elif name == "expon":
        params = dist.fit(x, floc=0)
        k = 1
    elif name == "beta":
        scale = 1.0 if beta_support == "unit" else x.max() * (len(x) + 1) / len(x)
        params = dist.fit(np.clip(x, 1e-6, scale - 1e-6), floc=0, fscale=scale)
        k = 2 if beta_support == "unit" else 3
    else:                                   # genextreme, norm
        params = dist.fit(x)
        k = len(params)
    return dist(*params), k


def fit_families(x, families=("gamma", "weibull_min", "lognorm", "expon", "beta", "genextreme"),
                 beta_support: str = "zero_max") -> tuple[pd.DataFrame, dict]:
    """Fit each family to x by maximum likelihood and compare them.

    Returns (table sorted by AIC, {family: frozen scipy distribution}). The
    table has the parameters, log-likelihood, AIC = 2k - 2 log L,
    BIC = k ln n - 2 log L and the Kolmogorov-Smirnov statistic and p-value.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rows, frozen = [], {}
    for name in families:
        try:
            d, k = _fit_one(name, x, beta_support)
            ll = float(np.sum(d.logpdf(x)))
            if not np.isfinite(ll):
                continue
        except Exception:
            continue                         # a family that does not converge drops out
        ks = stats.kstest(x, d.cdf)
        frozen[name] = d
        rows.append({"family": FAMILY_NAMES.get(name, name), "scipy": name,
                     "parameters": ", ".join(f"{p:.4g}" for p in d.args) + (
                         f" (loc {d.kwds.get('loc', 0):.3g}, scale {d.kwds.get('scale', 1):.3g})" if d.kwds else ""),
                     "log L": ll, "k": k, "AIC": 2 * k - 2 * ll, "BIC": k * np.log(len(x)) - 2 * ll,
                     "KS D": ks.statistic, "KS p": ks.pvalue})
    table = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    return table, frozen


def per_month_winners(values: pd.Series, months: pd.Series, families, beta_support: str = "zero_max") -> pd.DataFrame:
    """Best and runner-up family by AIC for each calendar month, with the AIC gap to Gamma."""
    rows = []
    for m in range(1, 13):
        x = values[months == m].dropna().to_numpy()
        x = x[x > 0] if beta_support == "zero_max" else x
        t, _ = fit_families(x, families, beta_support)
        gamma_aic = t.loc[t["scipy"] == "gamma", "AIC"]
        rows.append({"month": m, "best": t.loc[0, "family"], "runner-up": t.loc[1, "family"],
                     "AIC best": round(t.loc[0, "AIC"], 1),
                     "AIC gap Gamma - best": round(float(gamma_aic.iloc[0]) - t.loc[0, "AIC"], 1) if len(gamma_aic) else np.nan,
                     "KS p best": round(t.loc[0, "KS p"], 3)})
    return pd.DataFrame(rows)


# =============================================================================
# 2. COPULA TABLES
# =============================================================================

def tail_dependence(family: str, theta: float) -> tuple[float, float]:
    """(lambda_U, lambda_L): Gumbel and Clayton from spi_copula_analysis; the others have none."""
    if family in ("Gumbel", "Clayton"):
        return tail_dependence_coefs(family, theta)
    return 0.0, 0.0


def ds_copula_table(model, events: pd.DataFrame, n_boot: int = 500) -> pd.DataFrame:
    """Duration-severity copulas: the fits the dashboard uses, plus a Gaussian check.

    Gumbel, Clayton and Frank come straight from model.all_fits (the same theta
    and AIC as the dashboard). Gaussian is fitted with msdi_analysis.fit_copula_mle
    on the same (u, v) as a check only. The Cramer-von Mises p-value is the
    parametric bootstrap of Genest et al. (2009), msdi_analysis.cvm_gof.
    """
    u = model.marginals.F_D(events["Duration"].values)
    v = model.marginals.F_S(events["Severity"].values)
    n = len(u)
    rows = []
    for f in model.all_fits:
        rows.append({"family": f.family, "theta": f.parameter, "log L": f.loglik, "AIC": f.aic})
    th, ll, _ = M.fit_copula_mle("Gaussian", u, v)
    rows.append({"family": "Gaussian (check)", "theta": th, "log L": ll, "AIC": 2 - 2 * ll})
    for r in rows:
        fam = r["family"].split(" ")[0]
        r["BIC"] = np.log(n) - 2 * r["log L"]
        r["lambda_U"], r["lambda_L"] = tail_dependence(fam, r["theta"])
        r["CvM p"] = M.cvm_gof(fam, u, v, n_boot)[1]
        r["selected"] = "yes" if fam == model.best_family else ""
    return pd.DataFrame(rows)


def msdi_copula_table(model) -> pd.DataFrame:
    """The six MSDI copula fits with BIC, the tau range of each family and tail dependence."""
    n = int(model.series["u"].notna().sum())
    rows = []
    for f in model.copula_fits:
        lo, hi = M.TAU_RANGE.get(f.family, (-1.0, 1.0))
        lam_u, lam_l = tail_dependence(f.family, f.theta) if f.usable else (np.nan, np.nan)
        rows.append({"family": f.family, "tau range": f"{lo:+.3f} to {hi:+.3f}", "theta": f.theta,
                     "log L": f.loglik, "AIC": f.aic, "BIC": np.log(n) - 2 * f.loglik if f.usable else np.nan,
                     "lambda_U": lam_u, "lambda_L": lam_l, "RMSE": f.rmse, "CvM p": f.p_value,
                     "usable": f.usable, "selected": "yes" if f.family == model.best_family else "",
                     "note": f.note})
    return pd.DataFrame(rows)


def empirical_copula_grid(u, v, grid) -> np.ndarray:
    """C_n(a, b) = share of pairs with U <= a and V <= b, on a square grid (rows = b)."""
    U, V = M.pseudo_obs(np.asarray(u)), M.pseudo_obs(np.asarray(v))
    return np.array([[np.mean((U <= a) & (V <= b)) for a in grid] for b in grid])


# =============================================================================
# 3. KNOWN ISSUES OF THE PUBLISHED METHOD
# =============================================================================

def rescore_msdi(model, family: str, theta: float) -> pd.Series:
    """One-month MSDI recomputed with another copula family on the same (u, v), for sensitivity."""
    s = model.series
    return pd.Series(M.msdi_from_probability(M.copula_cdf(family, s["u"].to_numpy(), s["v"].to_numpy(), theta)), index=s.index)


def direction_check(model) -> pd.DataFrame:
    """Mean one-month MSDI of dry and hot, dry and cool, and wet and cool months.

    Low and high mean below 0.2 and above 0.8 on the pooled u and v. A drought
    index for crops should score dry and hot lowest, and wet and cool near the top.
    """
    s = model.series
    dry, wet, cool, hot = s["u"] < 0.2, s["u"] > 0.8, s["v"] < 0.2, s["v"] > 0.8
    groups = {"dry and hot": dry & hot, "dry and cool": dry & cool, "wet and cool": wet & cool,
              "all months": pd.Series(True, index=s.index)}
    return pd.DataFrame([{"group": name, "months": int(mask.sum()), "mean MSDI-1": round(float(s.loc[mask, "MSDI1"].mean()), 2),
                          "mean rain (mm/day)": round(float(s.loc[mask, "rain"].mean()), 2),
                          "mean temperature (degC)": round(float(s.loc[mask, "temp"].mean()), 2)}
                         for name, mask in groups.items()])


def seasonal_check(model, T: int, dry: float) -> pd.DataFrame:
    """Mean one-month MSDI per calendar month, and the months flagged at T by the dry cut-off."""
    s = model.series
    flagged = s[s[f"MSDI{T}"] < dry].groupby("Month").size().reindex(range(1, 13), fill_value=0)
    mean = s.groupby("Month")["MSDI1"].mean()
    return pd.DataFrame({"month": [calendar.month_abbr[m] for m in range(1, 13)], "mean MSDI-1": mean.round(2).to_numpy(),
                         f"months flagged at T = {T}": flagged.to_numpy()})


def without_seasons(model, T: int) -> pd.Series:
    """MSDI-T minus the mean of its calendar month, indexed by (Year, Month)."""
    s = model.series.dropna(subset=[f"MSDI{T}"]).set_index(["Year", "Month"])[f"MSDI{T}"]
    return s - s.groupby(level="Month").transform("mean")


def seasonal_share(model, T: int) -> float:
    """Share of the variance of MSDI-T explained by the calendar month alone (0 to 1)."""
    return 1 - float(without_seasons(model, T).var() / model.series[f"MSDI{T}"].var())


# =============================================================================
# 4. INDEX AGREEMENT
# =============================================================================

def cross_correlation(a: pd.Series, b: pd.Series, lags=range(-6, 7)) -> pd.DataFrame:
    """Pearson r of a(t) with b(t + k) for each lag k (positive k: b later than a)."""
    rows = []
    for k in lags:
        pair = pd.concat([a, b.shift(-k)], axis=1).dropna()
        rows.append({"lag (months)": k, "r": stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic})
    return pd.DataFrame(rows)


def dry_overlap(a: pd.Series, b: pd.Series, name_a: str, name_b: str, threshold: float = -1.0) -> pd.DataFrame:
    """Confusion-style 2 x 2 table: months dry (<= threshold) or not in each index."""
    pair = pd.concat([a, b], axis=1).dropna()
    da, db = pair.iloc[:, 0] <= threshold, pair.iloc[:, 1] <= threshold
    return pd.DataFrame({f"{name_b} dry": [int((da & db).sum()), int((~da & db).sum())],
                         f"{name_b} not dry": [int((da & ~db).sum()), int((~da & ~db).sum())]},
                        index=[f"{name_a} dry", f"{name_a} not dry"])
