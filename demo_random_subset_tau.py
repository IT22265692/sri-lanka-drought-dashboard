"""
demo_random_subset_tau.py
=========================
Why the paper reports Kendall's tau = -0.00711 between rainfall and temperature.

The authors' script CopulaSelection.m (copy in validation/authors_code/) does
not pair each month's rainfall with that month's temperature. It builds the
temperature probabilities v from two random subsets of temperature values
(datasample without replacement), puts them one after the other, and then
correlates them with the rainfall probabilities u, which are in month order.
The months no longer line up, so tau is close to zero whatever the data.

This script repeats the calculation both ways, using the paper's own
parameters (lines 13-14 and 21-27 of CopulaSelection.m), on NASA POWER data
at the paper's point (8.1996 N, 80.6327 E), 1981-2021:

    month-matched : v_i = GMM CDF of the temperature of the SAME month as u_i
    random subset : v built exactly as lines 29-45 of CopulaSelection.m

    py -3.12 demo_random_subset_tau.py

Writes validation/random_subset_tau.png and validation/random_subset_tau.csv
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

AUTHORS_FILE = Path("validation/authors_code/CopulaSelection.m")
DATA_FILE = Path("validation/nc_point_1981_2021.csv")
FIGURE = Path("validation/random_subset_tau.png")
SUMMARY = Path("validation/random_subset_tau.csv")
N_DRAWS = 1000
PAPER_TAU = -0.00711

# Paper's parameters, as written in CopulaSelection.m
A1, B1 = 0.77897, 5.27087                        # lines 13-14: Beta for rainfall
MU1, MU2, SIGMA1, SIGMA2 = 24.8586, 28.65893, 0.80152, 1.31911   # lines 21-24
PCT1, PCT2 = 0.31, 0.69                          # lines 26-27

# Lines that break the pairing (see the printout)
KEY_LINES = {32: "random subset 1: 31% of months, drawn without regard to month order",
             33: "subset 1 goes through component 1's normal CDF only",
             40: "random subset 2: 69% of months, drawn independently of subset 1",
             41: "subset 2 goes through component 2's normal CDF only",
             43: "the two subsets are stacked: row i of v is no longer month i",
             45: "this unpaired vector becomes v",
             48: "tau is computed between month-ordered u and this unpaired v"}


def print_authors_lines():
    """Print lines 16-48 of CopulaSelection.m, marking the lines that break the pairing."""
    lines = AUTHORS_FILE.read_text(encoding="utf-8").splitlines()
    print(f"Source: {AUTHORS_FILE}  (unmodified copy, see SOURCE.txt)\n")
    for i in range(16, 49):
        mark = ">>" if i in KEY_LINES else "  "
        note = f"   <-- {KEY_LINES[i]}" if i in KEY_LINES else ""
        print(f"{mark} {i:3d}  {lines[i - 1]}{note}")


def random_subset_experiment(n_draws: int = N_DRAWS, seed: int = 2024) -> dict:
    """Kendall's tau with months paired, and with v built from random subsets as in CopulaSelection.m.

    Returns the data, u, the month-matched v and its tau, the n_draws
    random-subset taus and one example random-subset v. Used by main() and by
    notebooks/03_MSDI_Analysis_and_Validation.ipynb.
    """
    d = pd.read_csv(DATA_FILE)
    R, T = d["prcp_mm_day"].values, d["ts"].values

    # Lines 16-17: rainfall probabilities, in month order
    u = stats.beta.cdf((R - R.min()) / (R.max() - R.min()), A1, B1)

    # Month-matched: each month's temperature through the paper's two-component mixture
    v_matched = PCT1 * stats.norm.cdf(T, MU1, SIGMA1) + PCT2 * stats.norm.cdf(T, MU2, SIGMA2)
    tau_matched = stats.kendalltau(u, v_matched).statistic

    # Random subset: lines 29-45, repeated n_draws times
    rng = np.random.default_rng(seed)
    n1, n2 = round(PCT1 * len(T)), round(PCT2 * len(T))

    def v_random():
        """One draw of v built as in CopulaSelection.m: two random subsets, stacked."""
        s1 = rng.choice(T, n1, replace=False)                      # line 32
        s2 = rng.choice(T, n2, replace=False)                      # line 40
        return np.r_[stats.norm.cdf(s1, MU1, SIGMA1),              # lines 33, 41, 43
                     stats.norm.cdf(s2, MU2, SIGMA2)]

    taus = np.array([stats.kendalltau(u, v_random()).statistic for _ in range(n_draws)])
    return {"data": d, "u": u, "v_matched": v_matched, "tau_matched": tau_matched,
            "taus": taus, "v_example": v_random()}


def main():
    """Compute tau both ways, print the comparison and save the three-panel figure."""
    # File output only; switched here (not at import) so notebooks can import this module
    plt.switch_backend("Agg")
    print_authors_lines()
    r = random_subset_experiment()
    d, u, v_matched, tau_matched = r["data"], r["u"], r["v_matched"], r["tau_matched"]
    taus, v_example = r["taus"], r["v_example"]

    print(f"\nData: NASA POWER at the paper's point, {d.Year.min()}-{d.Year.max()}, n = {len(d)} months")
    print(f"\nKendall's tau, month-matched (each month's rainfall with the same month's temperature): "
          f"{tau_matched:+.4f}")
    print(f"Kendall's tau, random-subset version (CopulaSelection.m), {N_DRAWS} draws: "
          f"mean {taus.mean():+.4f}, SD {taus.std():.4f}, "
          f"95% of draws between {np.percentile(taus, 2.5):+.4f} and {np.percentile(taus, 97.5):+.4f}")
    print(f"Paper: {PAPER_TAU:+.5f}   -> inside the random-subset range: "
          f"{np.percentile(taus, 2.5) <= PAPER_TAU <= np.percentile(taus, 97.5)}")
    print(f"Random-subset draws as far from zero as the month-matched tau: "
          f"{np.sum(np.abs(taus) >= abs(tau_matched))} of {N_DRAWS}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    ax[0].scatter(u, v_matched, s=10, alpha=0.6, color="#C0392B")
    ax[0].set_title(f"Month-matched\nKendall's tau = {tau_matched:+.3f}")
    ax[1].scatter(u, v_example, s=10, alpha=0.6, color="#7F8C8D")
    ax[1].set_title(f"Random subsets (CopulaSelection.m lines 32-45), one draw\n"
                    f"Kendall's tau = {stats.kendalltau(u, v_example).statistic:+.3f}")
    for a in ax[:2]:
        a.set_xlabel("u  (rainfall CDF, month order)"); a.set_ylabel("v  (temperature CDF)")
        a.set_xlim(0, 1); a.set_ylim(0, 1)
    ax[2].hist(taus, bins=40, color="#AAB7B8", edgecolor="white",
               label=f"random-subset version, {N_DRAWS} draws")
    ax[2].axvline(PAPER_TAU, color="k", ls="--", lw=1.5, label=f"paper: {PAPER_TAU:+.4f}")
    ax[2].axvline(tau_matched, color="#C0392B", lw=2.5, label=f"month-matched: {tau_matched:+.3f}")
    ax[2].set_xlabel("Kendall's tau (u, v)"); ax[2].set_ylabel("number of draws")
    ax[2].set_title("The paper's tau is what the random subsets give")
    ax[2].legend(fontsize=8, loc="upper left")
    fig.suptitle("Rainfall-temperature dependence at the paper's point (8.1996 N, 80.6327 E), "
                 "1981-2021, with the paper's own marginal parameters", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURE, dpi=130)
    print(f"\nFigure: {FIGURE}")
    pd.DataFrame([{"tau months paired": tau_matched, "tau random subsets, mean": taus.mean(), "tau random subsets, SD": taus.std(),
                   "draws": N_DRAWS, "paper tau": PAPER_TAU}]).round(4).to_csv(SUMMARY, index=False)


if __name__ == "__main__":
    main()
