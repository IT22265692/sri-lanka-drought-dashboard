# MSDI validation

> Written before the project moved to the published method, so its "MSDI corrected" rows are what is now called our adjustment (`_extension/`). Current results are in notebook 03 and `msdi_outputs/`.

## 1. Replication at the paper's point (8.1996 N, 80.6327 E, 1981-2021)

| item | paper | ours |
|---|---|---|
| Rainfall range (mm/day) | 0 to 18.41 | 0.00 to 20.71 |
| TS range (degC) | 22.98 to 33.74 | 22.98 to 33.75 |
| T2M range (degC) | 22.98 to 33.74 | 22.98 to 31.41 |
| Rainfall AIC by family | beta 2061.7, gamma 2104.2, HG 2265.9 | beta 2314.1, gamma 2243.3, halfnorm 2277.3 |
| Rainfall selected | beta (alpha 0.779, beta 5.271) | gamma (shape=0.9483, scale=3.78) |
| Temperature mixture | 0.31*N(24.86, 0.80) + 0.69*N(28.66, 1.32) | 0.18*N(24.36, 0.60) + 0.82*N(27.56, 1.63) |
| Kendall's tau, months paired | -0.0071 | -0.2922 |
| Kendall's tau, temperature shuffled (authors' code) | -0.0071 | 0.0016 |
| Copulas usable at this tau | Frank, FGM, AMH (FGM chosen) | Frank |
| Paper-formula MSDI-12, share of months below 0 | most months after 1992 (Fig. 7) | 100% (mean -1.13) |

## 2. Historical North Central droughts

A year counts as a drought year when at least 3 of its months are flagged dry. False alarm rate: share of the other 1984-2021 years flagged.

### Anuradhapura

| index | scale | hit rate | false alarm rate | skill (HR - FAR) | share of all months flagged |
|---|---|---|---|---|---|
| MSDI corrected | 1 | 0.38 | 0.2 | 0.18 | 0.15 |
| MSDI corrected | 3 | 0.62 | 0.28 | 0.34 | 0.16 |
| MSDI corrected | 6 | 0.5 | 0.2 | 0.3 | 0.16 |
| MSDI corrected | 9 | 0.5 | 0.16 | 0.34 | 0.14 |
| MSDI corrected | 12 | 0.38 | 0.16 | 0.22 | 0.13 |
| MSDI paper | 1 | 0.0 | 0.0 | 0.0 | 0.0 |
| MSDI paper | 3 | 0.88 | 1.0 | -0.12 | 0.36 |
| MSDI paper | 6 | 1.0 | 1.0 | 0.0 | 0.92 |
| MSDI paper | 9 | 1.0 | 1.0 | 0.0 | 0.98 |
| MSDI paper | 12 | 1.0 | 0.96 | 0.04 | 0.98 |
| SPI | 1 | 0.5 | 0.12 | 0.38 | 0.15 |
| SPI | 3 | 0.5 | 0.12 | 0.38 | 0.15 |
| SPI | 6 | 0.25 | 0.08 | 0.17 | 0.12 |
| SPI | 9 | 0.25 | 0.04 | 0.21 | 0.1 |
| SPI | 12 | 0.25 | 0.08 | 0.17 | 0.1 |

### Polonnaruwa

| index | scale | hit rate | false alarm rate | skill (HR - FAR) | share of all months flagged |
|---|---|---|---|---|---|
| MSDI corrected | 1 | 0.5 | 0.24 | 0.26 | 0.17 |
| MSDI corrected | 3 | 0.5 | 0.2 | 0.3 | 0.16 |
| MSDI corrected | 6 | 0.38 | 0.16 | 0.22 | 0.14 |
| MSDI corrected | 9 | 0.12 | 0.16 | -0.04 | 0.13 |
| MSDI corrected | 12 | 0.12 | 0.16 | -0.04 | 0.14 |
| MSDI paper | 1 | 0.0 | 0.0 | 0.0 | 0.0 |
| MSDI paper | 3 | 1.0 | 1.0 | 0.0 | 0.41 |
| MSDI paper | 6 | 1.0 | 1.0 | 0.0 | 0.96 |
| MSDI paper | 9 | 1.0 | 1.0 | 0.0 | 0.98 |
| MSDI paper | 12 | 1.0 | 0.96 | 0.04 | 0.98 |
| SPI | 1 | 0.38 | 0.16 | 0.22 | 0.16 |
| SPI | 3 | 0.38 | 0.12 | 0.26 | 0.14 |
| SPI | 6 | 0.38 | 0.12 | 0.26 | 0.13 |
| SPI | 9 | 0.25 | 0.08 | 0.17 | 0.11 |
| SPI | 12 | 0.25 | 0.08 | 0.17 | 0.1 |
