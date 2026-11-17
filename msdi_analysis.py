"""
msdi_analysis.py
================
Multivariate Standardized Drought Index (MSDI) from rainfall and ground
temperature for every district of Sri Lanka, by the published method:

    Karunarathne S, De Silva K, Perera S (2024). Modeling and indexing drought
    severity with multimodal ground temperature data. Environmental and
    Ecological Statistics 31:707-723. Following Hao and AghaKouchak (2013).

Where it sits in the pipeline:

    common.read_rainfall + common.read_temperature
        -> msdi_analysis.fit_msdi_model    (this module, one model per district)
        -> run_msdi_all_districts.py       (fits all districts, saves cache/msdi_fits.pkl)
        -> thresholds.py                   (class cut-offs: the paper's Table 4)
        -> index_data.py                   (serves the MSDI series to the dashboard)

The method:

    p = P(X <= x, Y <= y) = C(F_X(x), F_Y(y))     X = rainfall, Y = temperature   (paper eq. 6)
    MSDI = Phi^-1(p)                                                              (paper eq. 7)
    T-month MSDI = (1/T) * sum of the last T one-month MSDI values, T in 3, 6, 9, 12   (eq. 8)

    * X is the monthly average rainfall in mm/day, zero months set to 1e-10
      (Table 2 note). F_X is the lowest-AIC fit of Beta, Gamma and Half-Gaussian,
      pooled over all months (Table 2).
    * Y is ground temperature. F_Y is a two-component Gaussian mixture fitted by
      EM from a k-means start, pooled over all months (Sec. 2.1, eq. 2).
    * C is chosen by AIC from Frank, FGM and AMH (Table 3) plus Gaussian, Gumbel
      and Clayton as extra candidates.
    * Phi^-1(p) is capped at +/-5 (Sec. 3: "the infinity for the index is
      approximately taken as 5").

Rainfall and temperature are always paired month by month, as eq. 6 requires.
The MSDI is not standardised: its mean is about -1.1 and each T has its own
class cut-offs, so "below 0" and "<= -1" do not mean drought here. See
thresholds.py.

"""
from __future__ import annotations

import calendar
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, optimize, special, stats

import common
from spi_copula_analysis import (
    _clayton_cdf, _clayton_log_density, _frank_cdf, _frank_log_density,
    _gumbel_cdf, _gumbel_log_density,
)

T_SCALES = (3, 6, 9, 12)       # the paper's prolonged periods
ALL_SCALES = (1,) + T_SCALES   # 1 = the one-month MSDI that the T-month values average
TEMP_COL = "ts"                # NASA POWER earth skin temperature. The paper's "ground temperature":
                               # its range of 22.98 to 33.74 degC matches TS at the paper's point, not T2M
MSDI_CAP = 5.0                 # paper Sec. 3: infinity is approximately taken as 5
ZERO_RAIN = 1e-10              # paper Table 2 note: zero rainfall replaced with 1e-10
EPS = 1e-6                     # probabilities are kept inside (EPS, 1 - EPS)
GOF_ALPHA = 0.05               # reported only: a copula "passes" when its bootstrap p-value is at least this
PAPER_POINT_FILE = Path("validation") / "nc_point_1981_2021.csv"


def model_key(district: str, temp_col: str = TEMP_COL) -> str:
    """Cache key for one fitted model, e.g. "Anuradhapura__ts"."""
    return f"{district}__{temp_col}"


# =============================================================================
# 1. DATA
# =============================================================================

def load_inputs(district: str) -> pd.DataFrame:
    """Monthly rainfall and temperature for one district, each row one calendar month.

    Returns Year, Month, prcp (monthly total, mm), rain (monthly average, mm/day,
    the paper's unit, zero months set to 1e-10), t2m and ts (degC). The join on
    (Year, Month) is what keeps rainfall paired with the same month's temperature.
    """
    rain = common.read_rainfall(district)
    temp = common.read_temperature(district)
    df = rain[["Year", "Month", "prcp"]].merge(
        temp[["Year", "Month", "t2m", "ts"]], on=["Year", "Month"], how="inner", validate="1:1")
    df = df.sort_values(["Year", "Month"]).reset_index(drop=True)
    days = [calendar.monthrange(y, m)[1] for y, m in zip(df["Year"], df["Month"])]
    df.insert(3, "rain", (df["prcp"] / days).where(df["prcp"] > 0, ZERO_RAIN))
    return df


def load_paper_point() -> pd.DataFrame:
    """NASA POWER at the paper's own point (8.1996 N, 80.6327 E), 1981-2021, same columns as load_inputs."""
    df = pd.read_csv(PAPER_POINT_FILE).rename(columns={"prcp_mm_day": "rain"})
    df["rain"] = df["rain"].where(df["rain"] > 0, ZERO_RAIN)
    return df


# =============================================================================
# 2. MARGINAL DISTRIBUTIONS (paper Sec. 2.1, Table 2)
# =============================================================================
#
# k counts every parameter of the distribution, including location and bounds.
# This is what reproduces the paper's Table 2 AIC values (Beta k = 4, Gamma
# k = 3, Half-Gaussian k = 2).

RAIN_FAMILIES = ("beta", "gamma", "halfnorm")
BETA_UPPER_PAD = 0.01          # Beta upper bound = max + 1% of the data range


@dataclass
class RainFit:
    """One fitted rainfall distribution, with scipy's parameter order."""
    family: str
    params: tuple
    loglik: float
    k: int

    @property
    def aic(self) -> float:
        """Akaike information criterion, AIC = 2k - 2 log L."""
        return 2 * self.k - 2 * self.loglik

    @property
    def dist(self):
        """The scipy distribution of this family."""
        return {"beta": stats.beta, "gamma": stats.gamma, "halfnorm": stats.halfnorm}[self.family]

    def cdf(self, x) -> np.ndarray:
        """F_X(x)."""
        return self.dist.cdf(x, *self.params)

    def pdf(self, x) -> np.ndarray:
        """Density, used for plots."""
        return self.dist.pdf(x, *self.params)

    def describe(self) -> str:
        """Short text of the fitted parameters."""
        names = {"beta": ("alpha", "beta", "lower", "range"), "gamma": ("shape", "loc", "scale"),
                 "halfnorm": ("loc", "sigma")}[self.family]
        return ", ".join(f"{n}={p:.4g}" for n, p in zip(names, self.params))


def fit_rainfall(x: np.ndarray) -> list[RainFit]:
    """Beta, Gamma and Half-Gaussian by maximum likelihood, the families of the paper's Table 2.

    The four-parameter Beta is scaled to the data: lower bound 0 and upper
    bound just above the largest month, with alpha and beta by MLE. Returns the
    fits in RAIN_FAMILIES order.
    """
    x = np.asarray(x, dtype=float)
    upper = x.max() + BETA_UPPER_PAD * (x.max() - x.min())
    a, b, _, _ = stats.beta.fit(x, floc=0.0, fscale=upper)
    beta_p = (a, b, 0.0, upper)
    gamma_p = stats.gamma.fit(x)                  # shape, location and scale all by MLE
    half_p = stats.halfnorm.fit(x, floc=0)        # location fixed at 0, counted as in the paper
    return [RainFit("beta", beta_p, float(np.sum(stats.beta.logpdf(x, *beta_p))), 4),
            RainFit("gamma", gamma_p, float(np.sum(stats.gamma.logpdf(x, *gamma_p))), 3),
            RainFit("halfnorm", half_p, float(np.sum(stats.halfnorm.logpdf(x, *half_p))), 2)]


def best_rainfall(fits: list[RainFit]) -> RainFit:
    """The rainfall fit with the lowest AIC."""
    return min(fits, key=lambda f: f.aic)


@dataclass
class GMM:
    """Gaussian mixture f(y) = sum_i w_i N(y; mean_i, sd_i), components sorted by mean."""
    weights: np.ndarray
    means: np.ndarray
    sds: np.ndarray
    loglik: float
    n: int

    @property
    def k(self) -> int:
        """Number of free parameters: 3 per component, less one because the weights sum to 1."""
        return 3 * len(self.means) - 1

    @property
    def aic(self) -> float:
        """AIC = 2k - 2 log L."""
        return 2 * self.k - 2 * self.loglik

    @property
    def bic(self) -> float:
        """BIC = k ln n - 2 log L."""
        return self.k * np.log(self.n) - 2 * self.loglik

    def cdf(self, y) -> np.ndarray:
        """F_Y(y) = sum_i w_i Phi((y - mean_i) / sd_i)."""
        y = np.asarray(y, dtype=float)[..., None]
        return np.sum(self.weights * stats.norm.cdf(y, self.means, self.sds), axis=-1)

    def pdf(self, y) -> np.ndarray:
        """Mixture density, used for plots."""
        y = np.asarray(y, dtype=float)[..., None]
        return np.sum(self.weights * stats.norm.pdf(y, self.means, self.sds), axis=-1)

    def describe(self) -> str:
        """Short text such as "0.21*N(24.5, 0.65) + 0.79*N(28.0, 1.54)"."""
        return " + ".join(f"{w:.3f}*N({m:.3f}, {s:.3f})"
                          for w, m, s in zip(self.weights, self.means, self.sds))


def _kmeans_labels(y: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Two-cluster k-means on one variable: the paper's starting point for EM."""
    centres = np.percentile(y, [25, 75]).astype(float)
    labels = np.full(len(y), -1)
    for _ in range(max_iter):
        new = (np.abs(y - centres[1]) < np.abs(y - centres[0])).astype(int)
        if np.array_equal(new, labels):
            break
        labels = new
        centres = np.array([y[labels == 0].mean(), y[labels == 1].mean()])
    return labels


def fit_gmm(y: np.ndarray, tol: float = 1e-10, max_iter: int = 10_000) -> GMM:
    """Two-component Gaussian mixture by the EM algorithm, started from k-means (paper eq. 2)."""
    y = np.asarray(y, dtype=float)
    labels = _kmeans_labels(y)
    w = np.array([np.mean(labels == 0), np.mean(labels == 1)])
    m = np.array([y[labels == 0].mean(), y[labels == 1].mean()])
    s = np.array([y[labels == 0].std(), y[labels == 1].std()])
    previous, ll = -np.inf, -np.inf
    for _ in range(max_iter):
        # E-step: responsibility r_ij = w_j N(y_i; m_j, s_j) / sum_l w_l N(y_i; m_l, s_l)
        dens = w * stats.norm.pdf(y[:, None], m, s)
        r = dens / dens.sum(axis=1, keepdims=True)
        # M-step: weighted share, mean and standard deviation of each component
        nk = r.sum(axis=0)
        w = nk / len(y)
        m = (r * y[:, None]).sum(axis=0) / nk
        s = np.sqrt((r * (y[:, None] - m) ** 2).sum(axis=0) / nk)
        ll = float(np.sum(np.log(np.sum(w * stats.norm.pdf(y[:, None], m, s), axis=1))))
        if abs(ll - previous) < tol:
            break
        previous = ll
    order = np.argsort(m)
    return GMM(w[order], m[order], s[order], ll, len(y))


def fit_normal(y: np.ndarray) -> GMM:
    """One normal, as a one-component mixture: the comparison for the paper's multimodality claim."""
    y = np.asarray(y, dtype=float)
    m, s = y.mean(), y.std()
    return GMM(np.array([1.0]), np.array([m]), np.array([s]), float(np.sum(stats.norm.logpdf(y, m, s))), len(y))


# =============================================================================
# 3. COPULA FAMILIES
# =============================================================================
#
#   Frank      symmetric, any sign of dependence, no tail dependence
#   FGM        weak dependence only, |tau| <= 2/9
#   AMH        weak dependence only, -0.182 <= tau <= 1/3
#   Gaussian   any sign, no tail dependence
#   Gumbel     positive dependence, upper tail
#   Clayton    positive dependence, lower tail
#
# Frank, Gumbel and Clayton reuse the formulas in spi_copula_analysis.py.
# Formulas: Nelsen (2006), An Introduction to Copulas, chapter 4.

def _clip(u):
    """Keep probabilities inside (EPS, 1 - EPS) so logs and Phi^-1 stay finite."""
    return np.clip(u, EPS, 1 - EPS)


def _fgm_cdf(u, v, th):
    """Farlie-Gumbel-Morgenstern copula, C = uv[1 + theta(1-u)(1-v)]."""
    u, v = _clip(u), _clip(v)
    return u * v * (1 + th * (1 - u) * (1 - v))


def _fgm_log_density(u, v, th):
    """log c for FGM, c = 1 + theta(1-2u)(1-2v)."""
    u, v = _clip(u), _clip(v)
    return np.log(np.maximum(1 + th * (1 - 2 * u) * (1 - 2 * v), 1e-300))


def _amh_cdf(u, v, th):
    """Ali-Mikhail-Haq copula, C = uv / [1 - theta(1-u)(1-v)]."""
    u, v = _clip(u), _clip(v)
    return u * v / (1 - th * (1 - u) * (1 - v))


def _amh_log_density(u, v, th):
    """log c for AMH (second mixed derivative of the CDF above)."""
    u, v = _clip(u), _clip(v)
    d = 1 - th * (1 - u) * (1 - v)
    num = 1 + th * ((1 + u) * (1 + v) - 3) + th ** 2 * (1 - u) * (1 - v)
    return np.log(np.maximum(num, 1e-300)) - 3 * np.log(d)


def _gauss_cdf(u, v, rho):
    """Gaussian copula CDF: bivariate normal CDF at (Phi^-1(u), Phi^-1(v)).

    Computed with Owen's T function, which is fast and vectorised.
    """
    h = stats.norm.ppf(_clip(u))
    k = stats.norm.ppf(_clip(v))
    # Owen's T formula divides by h and k, so move exact zeros a hair away
    h = np.where(np.abs(h) < 1e-10, 1e-10, h)
    k = np.where(np.abs(k) < 1e-10, 1e-10, k)
    r = np.sqrt(1 - rho ** 2)
    a_h = (k - rho * h) / (h * r)
    a_k = (h - rho * k) / (k * r)
    corr = np.where(h * k < 0, 0.5, 0.0)
    out = 0.5 * (stats.norm.cdf(h) + stats.norm.cdf(k)) - special.owens_t(h, a_h) \
        - special.owens_t(k, a_k) - corr
    return np.clip(out, 0.0, 1.0)


def _gauss_log_density(u, v, rho):
    """log c for the Gaussian copula with correlation rho."""
    x = stats.norm.ppf(_clip(u))
    y = stats.norm.ppf(_clip(v))
    q = 1 - rho ** 2
    return -0.5 * np.log(q) - (rho ** 2 * (x ** 2 + y ** 2) - 2 * rho * x * y) / (2 * q)


def _debye1(th: float) -> float:
    """First Debye function D_1(theta) = (1/theta) * integral_0^theta t / (e^t - 1) dt."""
    if abs(th) < 1e-8:
        return 1.0
    val, _ = integrate.quad(lambda t: t / np.expm1(t) if t != 0 else 1.0, 0, th)
    return val / th


def _frank_tau(th):
    """Kendall's tau of the Frank copula, tau = 1 - 4/theta [1 - D_1(theta)]."""
    return 1 - 4 / th * (1 - _debye1(th))


def _amh_tau(th):
    """Kendall's tau of the AMH copula (Nelsen 2006, example 5.10)."""
    if abs(th) < 1e-8:
        return 0.0
    return 1 - 2 * (th + (1 - th) ** 2 * np.log1p(-th)) / (3 * th ** 2)


# name -> (cdf, log density, search intervals for the MLE, tau(theta))
COPULAS = {
    "Frank":    (_frank_cdf, _frank_log_density, [(-35.0, -1e-3), (1e-3, 35.0)], _frank_tau),
    "FGM":      (_fgm_cdf, _fgm_log_density, [(-1.0, 1.0)], lambda th: 2 * th / 9),
    "AMH":      (_amh_cdf, _amh_log_density, [(-1.0, 0.999)], _amh_tau),
    "Gaussian": (_gauss_cdf, _gauss_log_density, [(-0.98, 0.98)], lambda r: 2 / np.pi * np.arcsin(r)),
    "Gumbel":   (_gumbel_cdf, _gumbel_log_density, [(1.0001, 20.0)], lambda th: 1 - 1 / th),
    "Clayton":  (_clayton_cdf, _clayton_log_density, [(1e-3, 30.0)], lambda th: th / (th + 2)),
}
FAMILIES = ("Frank", "FGM", "AMH", "Gaussian", "Gumbel", "Clayton")
# Range of Kendall's tau each family can represent (others cover -1 to 1)
TAU_RANGE = {"FGM": (-2 / 9, 2 / 9), "AMH": (_amh_tau(-1.0), 1 / 3),
             "Gumbel": (0.0, 1.0), "Clayton": (0.0, 1.0)}


def copula_cdf(family: str, u, v, theta):
    """C(u, v) for any family in COPULAS."""
    return COPULAS[family][0](np.asarray(u, float), np.asarray(v, float), theta)


def copula_log_density(family: str, u, v, theta):
    """log c(u, v) for any family in COPULAS."""
    return COPULAS[family][1](np.asarray(u, float), np.asarray(v, float), theta)


def tau_of(family: str, theta: float) -> float:
    """Kendall's tau implied by a family and parameter."""
    return float(COPULAS[family][3](theta))


def fit_copula_mle(family: str, u, v) -> tuple[float, float, bool]:
    """Maximum likelihood theta for one family.

    Returns (theta, log-likelihood, at_bound). at_bound means the optimum sits
    on the edge of the parameter range, i.e. the data's dependence is stronger
    than this family can represent, so the family is marked unusable.
    """
    best = None
    for a, b in COPULAS[family][2]:
        # Maximise sum log c(u_i, v_i; theta) inside one search interval
        res = optimize.minimize_scalar(
            lambda th: -np.sum(copula_log_density(family, u, v, th)),
            bounds=(a, b), method="bounded", options={"xatol": 1e-6})
        if best is None or res.fun < best[1]:
            best = (float(res.x), float(res.fun), a, b)
    th, nll, a, b = best
    lo, hi = COPULAS[family][2][0][0], COPULAS[family][2][-1][1]
    at_bound = min(abs(th - lo), abs(th - hi)) < 1e-2 * (hi - lo)
    return th, -nll, at_bound


def sample_copula(family: str, theta: float, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Draw n pairs (u, v) from a fitted copula.

    Gaussian is sampled directly. The others by conditional inversion: draw u
    and q uniform, then solve h(v | u) = q by bisection, where h = dC/du is
    taken by central difference of the closed-form CDF. Used by the bootstrap
    test.
    """
    if family == "Gaussian":
        z1 = rng.standard_normal(n)
        z2 = theta * z1 + np.sqrt(1 - theta ** 2) * rng.standard_normal(n)
        return stats.norm.cdf(z1), stats.norm.cdf(z2)
    d = 1e-5
    u = rng.uniform(2 * d, 1 - 2 * d, n)
    q = rng.uniform(0, 1, n)
    lo, hi = np.zeros(n), np.ones(n)
    for _ in range(40):                    # 40 halvings: v accurate to about 1e-12
        mid = 0.5 * (lo + hi)
        h = (copula_cdf(family, u + d, mid, theta) - copula_cdf(family, u - d, mid, theta)) / (2 * d)
        below = h < q
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return u, 0.5 * (lo + hi)


def pseudo_obs(x: np.ndarray) -> np.ndarray:
    """Ranks scaled to (0, 1): rank / (n + 1). The standard input for copula tests."""
    return stats.rankdata(x) / (len(x) + 1)


def empirical_copula(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Empirical copula C_n at each observation: share of points below and left of it."""
    return ((u[None, :] <= u[:, None]) & (v[None, :] <= v[:, None])).mean(axis=1)


def _mle_or_none(family: str, u, v) -> float | None:
    """MLE theta, or None when it sits at the edge of the range (see fit_copula_mle)."""
    th, _, at_bound = fit_copula_mle(family, u, v)
    return None if at_bound else th


def cvm_gof(family: str, u, v, n_boot: int, seed: int = 0) -> tuple[float, float, float]:
    """Cramer-von Mises goodness-of-fit test with parametric bootstrap.

    Genest, Remillard and Beaudoin (2009):
        S_n = sum_i [C_n(U_i, V_i) - C_theta(U_i, V_i)]^2
    The p-value is the share of bootstrap samples, drawn from the fitted copula
    and refitted, whose S_n is at least the observed one.

    Returns (S_n, p-value, theta on the pseudo-observations). n_boot = 0 skips
    the bootstrap (p-value NaN).
    """
    U, V = pseudo_obs(u), pseudo_obs(v)
    th = _mle_or_none(family, U, V)
    if th is None:
        return np.nan, np.nan, np.nan
    sn = float(np.sum((empirical_copula(U, V) - copula_cdf(family, U, V, th)) ** 2))
    if n_boot <= 0:
        return sn, np.nan, th
    rng = np.random.default_rng(seed)
    n, exceed, done = len(U), 0, 0
    for _ in range(n_boot):
        bu, bv = sample_copula(family, th, n, rng)
        BU, BV = pseudo_obs(bu), pseudo_obs(bv)
        bth = _mle_or_none(family, BU, BV)
        if bth is None:
            continue
        bsn = np.sum((empirical_copula(BU, BV) - copula_cdf(family, BU, BV, bth)) ** 2)
        exceed += bsn >= sn
        done += 1
    # The +0.5 / +1 correction keeps the p-value away from exactly 0 or 1
    p = (exceed + 0.5) / (done + 1) if done else np.nan
    return sn, float(p), th


@dataclass
class CopulaResult:
    """Fit of one copula family to the (u, v) pairs."""
    family: str
    theta: float
    tau_implied: float
    loglik: float
    aic: float
    rmse: float           # sqrt(mean((C_n - C_theta)^2)) at the observations
    sn: float             # Cramer-von Mises statistic
    p_value: float        # bootstrap p-value of S_n
    usable: bool          # False if the dependence is outside the family's range
    note: str = ""


def fit_copulas(u: np.ndarray, v: np.ndarray, n_boot: int) -> list[CopulaResult]:
    """Fit all six families by MLE and test each one.

    u, v: month-paired probabilities. Returns one
    CopulaResult per family, in FAMILIES order. A family whose tau range does
    not include the sample tau is returned as unusable without fitting.
    """
    tau = float(stats.kendalltau(u, v).statistic)
    Cn = empirical_copula(pseudo_obs(u), pseudo_obs(v))
    out = []
    for fam in FAMILIES:
        lo, hi = TAU_RANGE.get(fam, (-1.0, 1.0))
        if not (lo < tau < hi):
            out.append(CopulaResult(fam, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                    False, f"tau = {tau:.3f} is outside this family's range "
                                           f"({lo:.3f} to {hi:.3f})"))
            continue
        th, ll, at_bound = fit_copula_mle(fam, u, v)
        rmse = float(np.sqrt(np.mean((Cn - copula_cdf(fam, pseudo_obs(u), pseudo_obs(v), th)) ** 2)))
        sn, p, _ = cvm_gof(fam, u, v, n_boot)
        out.append(CopulaResult(
            family=fam, theta=th, tau_implied=tau_of(fam, th), loglik=ll, aic=2 - 2 * ll,
            rmse=rmse, sn=sn, p_value=p, usable=not at_bound,
            note="parameter at the edge of its range" if at_bound else ""))
    return out


def select_copula(fits: list[CopulaResult]) -> CopulaResult:
    """The usable family with the lowest AIC. The test p-value is reported, not used."""
    return min((f for f in fits if f.usable), key=lambda f: f.aic)


# =============================================================================
# 4. MSDI (paper Sec. 3, eqs. 6-8)
# =============================================================================

@dataclass
class MSDIModel:
    """Everything fitted for one district.

    series columns: Year, Month, rain (mm/day), temp (degC), u = F_X(rain),
    v = F_Y(temp), p = C(u, v), MSDI1 and the T-month MSDI3, MSDI6, MSDI9, MSDI12.
    """
    district: str
    temp_col: str
    rain_fits: list[RainFit]                 # every candidate, for the AIC table
    rain: RainFit                            # lowest AIC, used for u
    gmm: GMM                                 # used for v
    normal: GMM                              # one normal, for comparison only
    kendall_tau: float
    spearman_rho: float
    copula_fits: list[CopulaResult]
    best_family: str
    best_theta: float
    series: pd.DataFrame
    n_boot: int = 0                          # bootstrap samples behind the GOF p-values


def msdi_from_probability(p) -> np.ndarray:
    """MSDI = Phi^-1(p), capped at +/-5 (paper eq. 7 and Sec. 3)."""
    return np.clip(stats.norm.ppf(p), -MSDI_CAP, MSDI_CAP)


def t_month_msdi(msdi1: pd.Series, T: int) -> pd.Series:
    """T-month MSDI = (1/T) * sum of the last T one-month values (paper eq. 8)."""
    return msdi1.rolling(T, min_periods=T).mean()


def fit_msdi_model(df: pd.DataFrame, district: str, temp_col: str = TEMP_COL,
                   n_boot: int = 500) -> MSDIModel:
    """Fit the published MSDI for one district or point.

    df: Year, Month, rain and temp_col, as from load_inputs(). n_boot: bootstrap
    samples for the copula test (the paper uses 2000). Returns an MSDIModel
    whose .series holds every monthly value.
    """
    # Marginals: one distribution for all months together
    x, y = df["rain"].to_numpy(float), df[temp_col].to_numpy(float)
    rain_fits = fit_rainfall(x)
    rain, gmm = best_rainfall(rain_fits), fit_gmm(y)
    u, v = _clip(rain.cdf(x)), _clip(gmm.cdf(y))

    # Copula on the month-paired (u, v), then p = C(u, v) (eq. 6)
    fits = fit_copulas(u, v, n_boot)
    best = select_copula(fits)
    series = df[["Year", "Month", "rain"]].reset_index(drop=True).assign(temp=y, u=u, v=v)
    series["p"] = copula_cdf(best.family, u, v, best.theta)

    # MSDI (eq. 7) and its T-month means (eq. 8)
    series["MSDI1"] = msdi_from_probability(series["p"])
    for T in T_SCALES:
        series[f"MSDI{T}"] = t_month_msdi(series["MSDI1"], T)

    return MSDIModel(district=district, temp_col=temp_col, rain_fits=rain_fits, rain=rain, gmm=gmm,
                     normal=fit_normal(y), kendall_tau=float(stats.kendalltau(u, v).statistic),
                     spearman_rho=float(stats.spearmanr(u, v).statistic), copula_fits=fits,
                     best_family=best.family, best_theta=best.theta, series=series, n_boot=n_boot)


def msdi_series(model: MSDIModel, scale: int) -> pd.DataFrame:
    """Year, Month and MSDI for T = scale (1 is the one-month MSDI)."""
    return model.series[["Year", "Month", f"MSDI{scale}"]].rename(columns={f"MSDI{scale}": "MSDI"})


# =============================================================================
# 5. MODEL CACHE
# =============================================================================

def load_model_cache(path: Path) -> dict:
    """The saved {key: MSDIModel} dict, or {} if the file does not exist yet."""
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def update_model_cache(path: Path, new: dict) -> dict:
    """Merge new models into the cache file and return the merged dict.

    Re-reads the file just before writing, so models another process (the
    batch script or a running dashboard) saved in the meantime are kept, and
    writes to a temporary file first so a reader never sees half a pickle.
    """
    merged = load_model_cache(path)
    merged.update(new)
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(merged, f)
    # On Windows the replace fails while another process (e.g. the running
    # dashboard) is reading the file; wait and retry instead of losing the fits
    for attempt in range(30):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(1)
    return merged
