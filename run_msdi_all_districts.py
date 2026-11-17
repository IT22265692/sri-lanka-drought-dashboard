"""
run_msdi_all_districts.py
=========================
Fits the published MSDI (msdi_analysis.py) for every district, sets the class
cut-offs (thresholds.py) and saves the models and tables. Run it after new
rainfall or temperature data.

Where it sits in the pipeline:

    msdi_analysis.fit_msdi_model  ->  this script  ->  cache/msdi_fits.pkl
                                                   ->  msdi_outputs/*.csv
    index_data.py then reads cache/msdi_fits.pkl for the dashboard.

    py -3.12 run_msdi_all_districts.py                 # TS, the paper's ground temperature
    py -3.12 run_msdi_all_districts.py --temp t2m      # air temperature instead (kept alongside)
    py -3.12 run_msdi_all_districts.py --boot 2000     # the paper's bootstrap size
    py -3.12 run_msdi_all_districts.py --workers 4     # fewer processes (less memory)

Writes:
    cache/msdi_fits.pkl                       {key: MSDIModel}, key = district__temp
    msdi_outputs/msdi_summary.csv             one row per district
    msdi_outputs/msdi_series.csv              every monthly value, long format
    msdi_outputs/threshold_*.csv              cut-offs, method per district, test-set skill

Existing cache entries are kept, so a re-run only fits what is missing.
Delete cache/msdi_fits.pkl to refit everything.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import common
import msdi_analysis as M
import thresholds

CACHE_FILE = common.CACHE_DIR / "msdi_fits.pkl"
OUT_DIR = Path("msdi_outputs")
PAPER_POINT = "Paper point"       # NASA POWER at the paper's own coordinates, for the replication


def _fit_job(district: str, temp_col: str, n_boot: int):
    """Fit one model in a worker process. Returns (cache key, MSDIModel)."""
    df = M.load_paper_point() if district == PAPER_POINT else M.load_inputs(district)
    return M.model_key(district, temp_col), M.fit_msdi_model(df, district, temp_col, n_boot)


def _summary_row(district: str, m: M.MSDIModel, shared: dict) -> dict:
    """One row of msdi_summary.csv for one fitted model."""
    best = next(f for f in m.copula_fits if f.family == m.best_family)
    aic = {f.family: f.aic for f in m.rain_fits}
    row = {
        "district": district, "province": thresholds.PROVINCE_OF[district], "temp": m.temp_col,
        "rainfall": m.rain.family, "rainfall parameters": m.rain.describe(),
        **{f"AIC {k}": round(v, 1) for k, v in aic.items()},
        "temperature mixture": m.gmm.describe(),
        "BIC mixture": round(m.gmm.bic, 1), "BIC one normal": round(m.normal.bic, 1),
        "kendall_tau": round(m.kendall_tau, 4),
        "copula": m.best_family, "theta": round(m.best_theta, 4), "copula_aic": round(best.aic, 2),
        "copula_rmse": round(best.rmse, 5), "gof_p": round(best.p_value, 4),
        "usable families": ", ".join(f.family for f in m.copula_fits if f.usable),
        "shares_grid_cell_with": shared.get(district, ""),
    }
    for T in M.ALL_SCALES:
        s = m.series[f"MSDI{T}"]
        row[f"MSDI{T} mean"] = round(s.mean(), 3)
        row[f"MSDI{T} sd"] = round(s.std(), 3)
    return row


def summarise(cache: dict, temp_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summary table and long monthly series for every district."""
    shared = common.shared_cells()
    rows, series = [], []
    for d in common.list_districts():
        m = cache[M.model_key(d, temp_col)]
        rows.append(_summary_row(d, m, shared))
        keep = ["Year", "Month", "rain", "temp", "u", "v", "p"] + [f"MSDI{T}" for T in M.ALL_SCALES]
        series.append(m.series[keep].assign(district=d, temp_var=temp_col))
    return pd.DataFrame(rows), pd.concat(series, ignore_index=True)


def fit_missing(cache: dict, temp_col: str, n_boot: int, workers: int) -> dict:
    """Fit every model that is missing or has fewer bootstrap samples than asked.

    Saves a checkpoint to the cache file every 10 models, so an interrupted
    run loses little work. Returns the merged cache.
    """
    def needed(key: str) -> bool:
        """True if this model must be (re)fitted."""
        return key not in cache or getattr(cache[key], "n_boot", 0) < n_boot

    jobs = [(d, temp_col, n_boot) for d in common.list_districts() + [PAPER_POINT]
            if needed(M.model_key(d, temp_col))]
    print(f"{len(jobs)} models to fit ({len(cache)} already cached), "
          f"{n_boot} bootstrap samples, {workers} workers")
    if not jobs:
        return cache

    t0 = time.time()
    pending = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fit_job, *j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            j = futures[fut]
            try:
                key, model = fut.result()
            except Exception as e:
                print(f"  !! {j[0]}: {e}")
                continue
            pending[key] = model
            print(f"  [{i}/{len(jobs)}] {j[0]}: {model.rain.family}, {model.best_family} "
                  f"(tau {model.kendall_tau:+.3f})  {time.time() - t0:.0f}s", flush=True)
            if len(pending) >= 10:
                M.update_model_cache(CACHE_FILE, pending)
                pending = {}
    return M.update_model_cache(CACHE_FILE, pending)


def main():
    """Parse arguments, fit what is missing, and write the tables."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", default=M.TEMP_COL, choices=["ts", "t2m"],
                    help="temperature variable; the paper's ground temperature matches ts")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap samples for the GOF test")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    t0 = time.time()
    cache = fit_missing(M.load_model_cache(CACHE_FILE), args.temp, args.boot, args.workers)

    OUT_DIR.mkdir(exist_ok=True)
    summary, series = summarise(cache, args.temp)
    summary.to_csv(OUT_DIR / "msdi_summary.csv", index=False)
    series.to_csv(OUT_DIR / "msdi_series.csv", index=False)
    if args.temp == M.TEMP_COL:
        thresholds.save_shares(thresholds.collect_series(cache))
    print(f"\nDone in {time.time() - t0:.0f}s. Wrote msdi_summary.csv, msdi_series.csv "
          f"and common_threshold_shares.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
