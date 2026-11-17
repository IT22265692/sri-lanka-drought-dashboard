# Sri Lanka Drought Dashboard

Drought monitoring and visualisation for paddy cultivation in Sri Lanka's 25 districts,
January 1984 to December 2025, from NASA POWER data.

| Index | Data | Method |
|---|---|---|
| **SPI** (rain) | PRECTOTCORR | Gamma per calendar month, SPI = Phi^-1(F_Gamma(x)), 1/3/6 months (McKee et al. 1993) |
| **SSI** (soil) | GWETROOT | best distribution per calendar month by AIC, same steps as SPI, 1/3/6 months (Sanchez Alcalde & Escorihuela 2025) |
| **MSDI** (rain + heat) | PRECTOTCORR + TS | published method: p = P(X <= x, Y <= y) = C(F_X(x), F_Y(y)), MSDI = Phi^-1(p), T-month MSDI = mean of the last T one-month values, T = 3, 6, 9, 12 (Karunarathne et al. 2024, Hao & AghaKouchak 2013). Marginals pooled over all months; copula by AIC from Frank, FGM, AMH, Gaussian, Gumbel, Clayton |

SPI and SSI are standardised and use the McKee classes. The MSDI is not standardised: it uses the
Dry and Extremely dry cut-offs of the paper's Table 4, one pair for each T, the same for every
district (`thresholds.py`).

| T | Extremely dry at or below | Dry at or below |
|---|---|---|
| 3 | -3.2 | -1.3 |
| 6 | -2.5 | -0.8 |
| 9 | -2.1 | -0.4 |
| 12 | -1.7 | -0.3 |

For SPI: run-theory drought events (duration D in months, severity S), D ~ Exponential and S ~ Gamma,
a duration-severity copula chosen by AIC (Gumbel, Clayton, Frank), and joint return periods T_AND and
T_OR (Shiau 2006). Events run below 0 for SPI and SSI. MSDI events are runs at or below the dry
cut-off, with severity as the sum of the amount below it. The paper does not define events, so this
is our own choice. SSI and MSDI have no return periods.

The SSI page shows run-theory drought events (runs below 0). SSI has no return period calculation
at this stage.

## Run the dashboard

```
py -3.12 -m pip install -r requirements.txt
py -3.12 -m streamlit run dashboard.py
```

Pages: **Overview**, **Rainfall (SPI)**, **Soil moisture (SSI)**, **Rainfall + heat (MSDI)**,
**Compare indices**, **Learn and how to use**.

## Rebuild the data and fits (only after new data)

```
py -3.12 "Nasapower district polygon pipeline.py"        # rainfall   -> district_rainfall/
py -3.12 "Nasapower district polygon pipeline SOIL.py"   # soil       -> district_soil/
py -3.12 "Nasapower district polygon pipeline TEMP.py"   # T2M and TS -> district_temperature/
py -3.12 run_msdi_all_districts.py --workers 4           # MSDI fits and shares -> cache/msdi_fits.pkl, msdi_outputs/
py -3.12 index_data.py                                   # index tables for the dashboard
py -3.12 compare_indices.py                              # comparison -> comparison_outputs/
```

## Files

| File | What it does |
|---|---|
| `common.py` | shared paths, data readers and constants (study period, documented droughts) |
| `spi_copula_analysis.py` | SPI, run-theory events, duration-severity copulas, return periods, drought classes |
| `ssi_analysis.py` | SSI and the SPI-SSI lag check |
| `msdi_analysis.py` | the published MSDI (marginals, six copulas, bootstrap test) |
| `thresholds.py` | the paper's Table 4 cut-offs, MSDI classes and events, and `common_threshold_shares.csv` |
| `run_msdi_all_districts.py` | fits the MSDI for all districts and writes the tables |
| `index_data.py` | serves SPI, SSI and MSDI to the dashboard in one table shape |
| `compare_indices.py` | correlation, lag, dry-month overlap and events |
| `model_checks.py` | checks beside the method: direction, seasonality, copula tables |
| `demo_random_subset_tau.py` | shows why the published tau is about 0 (random subsets in `CopulaSelection.m`) |
| `dashboard.py` | Streamlit app entry point and the SPI pages |
| `ui.py`, `pages_extra.py`, `events_panel.py`, `soil_tabs.py`, `msdi_tabs.py`, `learn_page.py` | dashboard pages and shared charts |

`validation/` holds checks of the published method:
`random_subset_tau.png`, `msdi_validation_report.md` and an unmodified copy of the authors'
`CopulaSelection.m`.

`_removed/` holds the files set aside during the clean-up, and `_removed/before_paper_switch/`
the state of the project before the MSDI moved to the published method. It is ignored by git and
can be deleted once nothing in it is needed. Each subfolder holds work set aside, with notes on how
to bring it back. `_removed/common_thresholds_change/` holds the earlier MSDI cut-off work.

## Open points

* Six pairs of small districts share one NASA POWER grid cell (identical series):
  Batticaloa/Polonnaruwa, Colombo/Gampaha, Jaffna/Kilinochchi, Kandy/Matale, Kegalle/Nuwara Eliya,
  Mullaitivu/Vavuniya.
* `msdi_outputs/common_threshold_shares.csv` gives the share of months Table 4 flags in each district
  and T.
* The SPI and SSI class cut-offs (McKee: -1, -1.5, -2) are the same for every district.
