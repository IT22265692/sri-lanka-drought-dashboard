"""
ssi_analysis.py
===============
Standardized Soil Moisture Index (SSI) from NASA POWER root-zone soil wetness
(GWETROOT, a fraction between 0 and 1).

Where it sits in the pipeline:

    common.read_soil  ->  compute_ssi (this module)  ->  index_data.ssi_series
                                                     ->  dashboard SSI page

Method (Sanchez Alcalde and Escorihuela 2025, Remote Sensing 17, 3916), the
same steps as SPI:
    1. k-month rolling MEAN of soil wetness (soil moisture is a state, so it
       is averaged, where rainfall is summed).
    2. For each calendar month, fit Beta, Gamma, Log-normal, Log-logistic,
       GEV and Normal by maximum likelihood and keep the lowest AIC.
    3. SSI = Phi^-1(F(x)), so SSI is on the same standard normal scale as SPI
       and uses the same McKee classes.
    4. Robustness check: an empirical SSI from Gringorten plotting positions,
       p = (rank - 0.44) / (n + 0.12), correlated with the parametric SSI.
    5. Run-theory drought events (index below 0), as for SPI.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------- constants

# McKee categories, applied unchanged to SSI so the legend is shared with SPI
DROUGHT_CATEGORIES = [
    (-np.inf, -2.0, "Extremely dry"),
    (-2.0, -1.5, "Severely dry"),
    (-1.5, -1.0, "Moderately dry"),
    (-1.0, 1.0, "Near normal"),
    (1.0, 1.5, "Moderately wet"),
    (1.5, 2.0, "Very wet"),
    (2.0, np.inf, "Extremely wet"),
]

# Candidate marginals for AIC selection. The paper found Beta best for about
# 60% of its grid and showed the choice barely moves the final SSI, so the
# selection is made per district and calendar month and the winner is recorded.
CANDIDATE_MARGINALS = ("beta", "gamma", "lognorm", "loglogistic", "gev", "norm")


# --------------------------------------------------------------- data classes

@dataclass
class MarginalFit:
    """Result of fitting the best marginal to one calendar-month group."""
    family: str
    params: tuple
    aic: float
    ks_stat: float
    ks_pvalue: float
    n: int

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """F(x) of the selected family."""
        return _cdf(self.family, self.params, x)


@dataclass
class SSIResult:
    """Everything computed for one (district, scale) combination."""
    district: str
    scale: int                              # 1, 3, or 6 months
    ssi: pd.Series                          # SSI values, indexed by month
    ssi_empirical: pd.Series                # Gringorten robustness check
    marginals: dict[int, MarginalFit]       # keyed by calendar month 1..12
    events: pd.DataFrame                    # run-theory drought events
    robustness_r: float                     # Pearson r, parametric vs empirical


# ------------------------------------------------------- distribution helpers

def _fit(family: str, x: np.ndarray) -> tuple:
    """MLE fit for one candidate family. Returns scipy `args`."""
    if family == "beta":
        # Bounded on (0, 1) with fixed loc=0, scale=1 for soil wetness fraction
        a, b, _, _ = stats.beta.fit(x, floc=0, fscale=1)
        return (a, b)
    if family == "gamma":
        a, _, sc = stats.gamma.fit(x, floc=0)
        return (a, 0, sc)
    if family == "lognorm":
        s, _, sc = stats.lognorm.fit(x, floc=0)
        return (s, 0, sc)
    if family == "loglogistic":
        # Fisk in scipy is the log-logistic
        c, _, sc = stats.fisk.fit(x, floc=0)
        return (c, 0, sc)
    if family == "gev":
        c, loc, sc = stats.genextreme.fit(x)
        return (c, loc, sc)
    if family == "norm":
        loc, sc = stats.norm.fit(x)
        return (loc, sc)
    raise ValueError(family)


def _cdf(family: str, params: tuple, x: np.ndarray) -> np.ndarray:
    """CDF of one candidate family with scipy-style parameters."""
    if family == "beta":
        return stats.beta.cdf(x, *params)
    if family == "gamma":
        return stats.gamma.cdf(x, *params)
    if family == "lognorm":
        return stats.lognorm.cdf(x, *params)
    if family == "loglogistic":
        return stats.fisk.cdf(x, *params)
    if family == "gev":
        return stats.genextreme.cdf(x, *params)
    if family == "norm":
        return stats.norm.cdf(x, *params)
    raise ValueError(family)


def _loglik(family: str, params: tuple, x: np.ndarray) -> float:
    """Log-likelihood sum log f(x_i) of one candidate family, used for AIC."""
    if family == "beta":
        return stats.beta.logpdf(x, *params).sum()
    if family == "gamma":
        return stats.gamma.logpdf(x, *params).sum()
    if family == "lognorm":
        return stats.lognorm.logpdf(x, *params).sum()
    if family == "loglogistic":
        return stats.fisk.logpdf(x, *params).sum()
    if family == "gev":
        return stats.genextreme.logpdf(x, *params).sum()
    if family == "norm":
        return stats.norm.logpdf(x, *params).sum()
    raise ValueError(family)


def _k_params(family: str) -> int:
    """Number of free parameters (location fixed at 0 counts as not free)."""
    return {"beta": 2, "gamma": 2, "lognorm": 2, "loglogistic": 2,
            "gev": 3, "norm": 2}[family]


def fit_best_marginal(x: np.ndarray,
                      candidates: tuple[str, ...] = CANDIDATE_MARGINALS
                      ) -> MarginalFit:
    """Fit each candidate, return the one with lowest AIC.

    Alcalde et al. (2025) picked beta on the strength of a KS test across their
    grid. We keep beta in the candidate set and let AIC decide per district and
    calendar month, because 42 years is a small enough sample that a single
    family may not win everywhere on the island.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    # Nudge away from the open-interval boundaries so beta doesn't blow up
    x = np.clip(x, 1e-6, 1 - 1e-6)

    best: MarginalFit | None = None
    for family in candidates:
        try:
            params = _fit(family, x)
            ll = _loglik(family, params, x)
            aic = 2 * _k_params(family) - 2 * ll
            if best is None or aic < best.aic:
                ks_stat, ks_p = stats.kstest(x, lambda v: _cdf(family, params, v))
                best = MarginalFit(family, params, aic, ks_stat, ks_p, len(x))
        except Exception:
            continue

    if best is None:
        raise RuntimeError("No candidate marginal converged for this group")
    return best


# --------------------------------------------------------------- main compute

def compute_ssi(df: pd.DataFrame,
                scale: int,
                sm_col: str = "gwetroot",
                date_col: str = "date") -> SSIResult:
    """
    Compute the SSI for one district at one accumulation scale.
    Returns an SSIResult (series, Gringorten check, marginals, events).

    Parameters
    ----------
    df : DataFrame with columns [date_col, sm_col] at monthly frequency.
         Must already be sorted by date and free of gaps.
    scale : accumulation window in months (1, 3, or 6).
    sm_col : name of the soil moisture column (values on (0, 1)).
    """
    if scale not in (1, 3, 6):
        raise ValueError("scale must be 1, 3, or 6")
    dfx = df[[date_col, sm_col]].copy()
    dfx[date_col] = pd.to_datetime(dfx[date_col])
    dfx = dfx.sort_values(date_col).reset_index(drop=True)

    # ---- 1. Aggregate: rolling MEAN, because soil moisture is a state.
    # SPI uses a rolling sum. For scale-family marginals sum and mean give the
    # same standardised value, and the mean keeps the 0-1 units of GWETROOT.
    dfx[f"sm_{scale}"] = dfx[sm_col].rolling(scale, min_periods=scale).mean()
    dfx["month"] = dfx[date_col].dt.month
    dfx["ssi"] = np.nan
    dfx["ssi_emp"] = np.nan

    marginals: dict[int, MarginalFit] = {}

    # ---- 2-4. Group by calendar month, fit marginal, standardize
    for m in range(1, 13):
        group = dfx[dfx["month"] == m][f"sm_{scale}"].dropna()
        if len(group) < 20:
            continue                                    # not enough history
        fit = fit_best_marginal(group.to_numpy())
        marginals[m] = fit

        # Parametric SSI = Phi^-1(F(x)), Alcalde eq. (11)-(12)
        idx = dfx.index[(dfx["month"] == m) & dfx[f"sm_{scale}"].notna()]
        F = fit.cdf(dfx.loc[idx, f"sm_{scale}"].to_numpy())
        dfx.loc[idx, "ssi"] = stats.norm.ppf(np.clip(F, 1e-6, 1 - 1e-6))

        # Empirical SSI via Gringorten position, Alcalde eq. (13)
        ranks = stats.rankdata(dfx.loc[idx, f"sm_{scale}"].to_numpy())
        p = (ranks - 0.44) / (len(idx) + 0.12)
        dfx.loc[idx, "ssi_emp"] = stats.norm.ppf(p)

    # ---- 5. Robustness r: parametric vs empirical SSI (should be close to 1)
    both = dfx[["ssi", "ssi_emp"]].dropna()
    if len(both) > 10:
        robustness_r = float(np.corrcoef(both["ssi"], both["ssi_emp"])[0, 1])
    else:
        robustness_r = float("nan")

    # ---- 6. Run-theory event extraction, same rule as SPI
    events = extract_drought_events(
        dfx.set_index(date_col)["ssi"].dropna(), index_name=f"SSI-{scale}")

    return SSIResult(
        district=str(df.get("district", ["unknown"])[0]) if "district" in df else "unknown",
        scale=scale,
        ssi=dfx.set_index(date_col)["ssi"].rename(f"SSI-{scale}"),
        ssi_empirical=dfx.set_index(date_col)["ssi_emp"].rename(f"SSI-{scale}_empirical"),
        marginals=marginals,
        events=events,
        robustness_r=robustness_r,
    )


# ---------------------------------------------------- run-theory events

def extract_drought_events(index_series: pd.Series,
                           threshold: float = 0.0,
                           index_name: str = "SSI") -> pd.DataFrame:
    """
    Find drought events as unbroken runs where the index < threshold.

    Returns a DataFrame with one row per event and these columns:
      start, end, duration_months, severity, peak_index, category
    """
    s = index_series.dropna()
    in_event = (s < threshold).to_numpy()
    if not in_event.any():
        return pd.DataFrame(columns=["start", "end", "duration_months",
                                     "severity", "peak_index", "category"])

    # Find run boundaries
    change = np.diff(in_event.astype(int), prepend=0, append=0)
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0] - 1

    rows = []
    dates = s.index
    vals = s.to_numpy()
    for i0, i1 in zip(starts, ends):
        run_vals = vals[i0:i1 + 1]
        peak = float(run_vals.min())
        rows.append({
            "start": dates[i0],
            "end": dates[i1],
            "duration_months": int(i1 - i0 + 1),
            "severity": float(-run_vals.sum()),   # positive number
            "peak_index": peak,
            "category": _category_of(peak),
        })
    return pd.DataFrame(rows)


def _category_of(z: float) -> str:
    """McKee class name of an index value."""
    for lo, hi, name in DROUGHT_CATEGORIES:
        if lo <= z < hi:
            return name
    return "Unknown"


# --------------------------------------------- SPI-SSI cross-validation

def cross_validate_ssi_spi(spi: pd.Series, ssi: pd.Series,
                           lags: tuple[int, ...] = (-1, 0, 1, 2)
                           ) -> pd.DataFrame:
    """
    Reproduce Alcalde et al. (2025) Table 1 for a single district.

    For each monthly lag delta in `lags`, compute the Pearson correlation
    between SPI(t) and SSI(t + delta). The paper found:

      - delta = 0 wins at short scales (1, 3 months)
      - delta = 1 slightly wins at 6 and 9 months
      - Across the board, comparing SSI-k against SPI-(k+1) gives a small but
        consistent improvement over SPI-k of the same window

    Used to check whether the Sri Lanka data follows the same pattern.
    """
    idx = spi.index.intersection(ssi.index)
    if len(idx) < 24:
        return pd.DataFrame({"lag": lags, "correlation": [np.nan] * len(lags)})
    spi_al = spi.loc[idx]
    ssi_al = ssi.loc[idx]

    out = []
    for d in lags:
        if d == 0:
            r = spi_al.corr(ssi_al)
        elif d > 0:
            r = spi_al.iloc[:-d].corr(ssi_al.iloc[d:].reset_index(drop=True)
                                       .set_axis(spi_al.iloc[:-d].index))
        else:
            r = spi_al.iloc[-d:].corr(ssi_al.iloc[:d].reset_index(drop=True)
                                       .set_axis(spi_al.iloc[-d:].index))
        out.append({"lag": d, "correlation": r})
    return pd.DataFrame(out)


def propagation_time(spi_by_scale: dict[int, pd.Series],
                     ssi_1: pd.Series) -> tuple[int, float]:
    """
    Propagation time: the SPI accumulation window whose correlation with
    1-month SSI is highest, i.e. how many months of rain the soil integrates.

    Returns (best_scale_months, best_correlation).
    """
    best_scale, best_r = 1, -np.inf
    for k, spi_k in spi_by_scale.items():
        idx = spi_k.index.intersection(ssi_1.index)
        if len(idx) < 24:
            continue
        r = spi_k.loc[idx].corr(ssi_1.loc[idx])
        if r > best_r:
            best_r = r
            best_scale = k
    return best_scale, float(best_r)


# ---------------------------------------------- CLI for quick district test

if __name__ == "__main__":
    """Run a smoke test on synthetic data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("1984-01-01", "2025-12-01", freq="MS")
    # Simulate soil wetness with a seasonal cycle and noise
    seasonal = 0.4 + 0.15 * np.sin(2 * np.pi * (dates.month - 3) / 12)
    noise = rng.normal(0, 0.06, size=len(dates))
    sm = np.clip(seasonal + noise, 0.05, 0.95)

    df = pd.DataFrame({"date": dates, "gwetroot": sm})
    print(f"Test data: {len(df)} months from {df.date.min()} to {df.date.max()}")

    for scale in (1, 3, 6):
        r = compute_ssi(df, scale)
        print(f"\nSSI-{scale}:")
        print(f"  Robustness r (parametric vs Gringorten): {r.robustness_r:.4f}")
        print(f"  Drought events: {len(r.events)}")
        if len(r.events):
            print(f"  Longest: {r.events['duration_months'].max()} months, "
                  f"severity {r.events['severity'].max():.2f}")
        # Which marginal won for each calendar month?
        families = [f"{m}:{f.family}" for m, f in sorted(r.marginals.items())]
        print(f"  Marginals selected: {' '.join(families)}")