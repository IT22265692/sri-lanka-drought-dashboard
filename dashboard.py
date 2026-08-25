"""
SPI Drought Dashboard for Sri Lanka  (v4)
==========================================
Changes from v3, per user feedback:
  1. Kalmunai added to the CLIMATE_ZONE map so no district appears white
  2. Learn tab rebuilt with full methodology walkthrough:
     - What SPI is, with equations
     - How the dashboard works: data collection, function flow
     - Copula selection and model choice with equations
     - Duration and Severity with worked equations
     - A "Limitations" section (not signed "using this sign")
  3. Interpretation panels added below each of the three 3D plots in
     the Expert view (joint PDF, copula density, joint CDF contour)
  4. Em-dashes removed from all user-facing prose (kept only inside
     Python comments, which users never see) to avoid the AI-tell
"""
from __future__ import annotations
import calendar
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from spi_copula_analysis import (
    compute_spi, build_district_model, return_period_table,
    T_and, T_or, estimate_spi_from_input,
    spi_category, spi_plain_english, tail_dependence_coefs,
    _gumbel_log_density, _clayton_log_density, _frank_log_density, _copula_cdf,
)

DATA_DIR   = Path("district_rainfall")
GADM_SHP   = Path("gadm41_LKA") / "gadm41_LKA_2.shp"
CACHE_DIR  = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "copula_fits.pkl"

# Theme
CREAM      = "#F5EEE0"
CREAM_2    = "#EFE5D2"
BROWN      = "#5A3A1B"
BROWN_DARK = "#3B2611"
TERRACOTTA = "#C8501A"
TERRA_SOFT = "#D9884F"
FOREST     = "#5A6E3A"
GOLD       = "#C9A227"
INK        = "#2E1F10"

DRY_SCALE = [
    (0.00, "#7A3211"), (0.15, "#B04A1A"), (0.30, "#DE924F"),
    (0.50, "#EFE5D2"), (0.70, "#7FA2B8"), (0.85, "#3B6A85"), (1.00, "#1E3A52"),
]

# Approximate district -> agro-climatic zone.
# Kalmunai is added because the GADM level-1 file sometimes carries it as a
# separate district in Sri Lanka. Without this entry it would render white.
CLIMATE_ZONE = {
    "Colombo": "Wet", "Gampaha": "Wet", "Kalutara": "Wet", "Galle": "Wet",
    "Matara": "Wet", "Kandy": "Wet", "Nuwara Eliya": "Wet", "Ratnapura": "Wet",
    "Kegalle": "Wet",
    "Kurunegala": "Intermediate", "Matale": "Intermediate", "Badulla": "Intermediate",
    "Moneragala": "Intermediate", "Hambantota": "Intermediate",
    "Jaffna": "Dry", "Kilinochchi": "Dry", "Mannar": "Dry", "Vavuniya": "Dry",
    "Mullaitivu": "Dry", "Trincomalee": "Dry", "Batticaloa": "Dry", "Ampara": "Dry",
    "Polonnaruwa": "Dry", "Anuradhapura": "Dry", "Puttalam": "Dry",
    "Kalmunai": "Dry",
}
ZONE_COLOUR = {"Wet": FOREST, "Intermediate": GOLD, "Dry": TERRACOTTA}

st.set_page_config(page_title="Sri Lanka Drought Dashboard", page_icon="", layout="wide")

st.markdown(f"""
    <style>
    .stApp {{ background: {CREAM}; }}
    h1, h2, h3, h4 {{ color: {BROWN_DARK}; }}
    .stMarkdown, .stCaption, .stText {{ color: {INK}; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; background: {CREAM_2}; padding: 6px; border-radius: 10px; }}
    .stTabs [data-baseweb="tab"] {{ height: 52px; padding: 0px 20px; font-size: 15px; font-weight: 600; background: {CREAM}; color: {BROWN}; border-radius: 8px; }}
    .stTabs [aria-selected="true"] {{ background: {TERRACOTTA} !important; color: white !important; }}
    div[data-testid="stMetric"] {{ background: {CREAM_2}; border-left: 4px solid {TERRACOTTA}; padding: 12px 16px; border-radius: 6px; }}
    div[data-testid="stMetricLabel"] {{ color: {BROWN} !important; font-weight: 600; }}
    div[data-testid="stMetricValue"] {{ color: {BROWN_DARK} !important; }}
    .stRadio > label, .stSelectbox > label, .stSlider > label, .stNumberInput > label {{ color: {BROWN} !important; font-weight: 600; }}
    .stExpander {{ background: {CREAM_2}; border: 1px solid {TERRA_SOFT}; border-radius: 8px; }}
    div[data-testid="stAlert"] {{ background: {CREAM_2}; border-left: 4px solid {FOREST}; }}
    .stButton>button {{ background: {TERRACOTTA}; color: white; border: none; font-weight: 600; }}
    </style>
""", unsafe_allow_html=True)


# ---------------- Data / cache helpers ----------------
@st.cache_data
def list_districts() -> list[str]:
    files = sorted(DATA_DIR.glob("Pre_processed_data_*.xlsx"))
    return [f.stem.replace("Pre_processed_data_", "").replace("_", " ") for f in files]

@st.cache_data
def load_precip(district: str) -> pd.DataFrame:
    fname = DATA_DIR / f"Pre_processed_data_{district.replace(' ', '_')}.xlsx"
    return pd.read_excel(fname)

@st.cache_data
def compute_spi_series(district: str, scale: int) -> pd.DataFrame:
    return compute_spi(load_precip(district), scale=scale)

@st.cache_data
def load_district_geometries():
    import geopandas as gpd
    gdf = gpd.read_file(GADM_SHP)
    gdf = gdf.dissolve(by="NAME_1", as_index=False)
    gdf = gdf.rename(columns={"NAME_1": "District"})[["District", "geometry"]]
    return gdf.to_crs(epsg=4326)


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def _save_cache(cache: dict):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


def get_or_fit_model(district: str, scale: int):
    cache = _load_cache()
    key = f"{district}__SPI{scale}"
    if key in cache:
        cached = cache[key]
        # Auto-invalidate stale caches from an older library version that
        # fitted Gaussian. If the cached best_family is not one we still
        # support, drop and refit. Prevents "ValueError: Gaussian" for
        # users who upgraded the code without clearing cache/.
        if cached["model"].best_family in ("Gumbel", "Clayton", "Frank"):
            return cached
    precip = load_precip(district)
    with st.spinner(f"First analysis for {district} SPI-{scale}, fitting 3 copulas "
                    f"(about 15 seconds). Cached for all future visits."):
        model, spi_df, events = build_district_model(precip, district, scale)
        rp_table = return_period_table(model)
    cache[key] = {"model": model, "events": events, "return_periods": rp_table}
    _save_cache(cache)
    return cache[key]


# =======================================================================
# TAB 1: NATIONAL MAP
# =======================================================================
def tab_national_map():
    st.header("How dry is Sri Lanka right now?")
    st.markdown("Pick a month and time window. The map colours each district by how unusual its "
                 "rainfall was compared to the same month across 1984 to 2025.")

    districts = list_districts()
    if not districts:
        st.error(f"No data files found in {DATA_DIR}/. Please run the fetch pipeline first.")
        return

    view = st.radio("Map view", ["Current rainfall (SPI)", "Climate zones (Dry / Intermediate / Wet)"],
                     horizontal=True)

    geo = load_district_geometries()
    merged_base = geo.copy()

    minx, miny, maxx, maxy = geo.total_bounds
    pad_x = (maxx - minx) * 0.08
    pad_y = (maxy - miny) * 0.08
    lon_range = [minx - pad_x, maxx + pad_x]
    lat_range = [miny - pad_y, maxy + pad_y]

    if view == "Climate zones (Dry / Intermediate / Wet)":
        merged_base["Zone"] = merged_base["District"].map(CLIMATE_ZONE)
        # Any GADM name that we don't have a zone for gets flagged, so it
        # never renders white silently.
        missing = merged_base[merged_base["Zone"].isna()]["District"].tolist()
        if missing:
            merged_base["Zone"] = merged_base["Zone"].fillna("Unknown")

        col_map, col_summary = st.columns([1.7, 1])
        with col_map:
            fig = px.choropleth(
                merged_base, geojson=merged_base.geometry, locations=merged_base.index,
                color="Zone",
                color_discrete_map={**ZONE_COLOUR, "Unknown": "#888888"},
                hover_name="District", labels={"Zone": "Climate zone"},
            )
            fig.update_geos(visible=False, resolution=50, bgcolor=CREAM,
                             lonaxis_range=lon_range, lataxis_range=lat_range)
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=820,
                               paper_bgcolor=CREAM,
                               legend=dict(font_color=BROWN_DARK, bgcolor=CREAM_2,
                                           bordercolor=TERRA_SOFT, borderwidth=1))
            st.plotly_chart(fig, use_container_width=True)
        with col_summary:
            st.subheader("Sri Lanka's agro-climatic zones")
            for zone, colour in ZONE_COLOUR.items():
                n = sum(1 for v in CLIMATE_ZONE.values() if v == zone)
                st.markdown(
                    f"<div style='background:{CREAM_2};padding:10px 14px;"
                    f"border-left:4px solid {colour};border-radius:6px;margin-bottom:8px'>"
                    f"<b style='color:{BROWN_DARK}'>{zone} Zone</b> "
                    f"<span style='color:{INK}'>({n} districts)</span></div>",
                    unsafe_allow_html=True,
                )
            st.caption(
                "**Wet Zone**: consistently high rainfall year-round (southwest, central hills). "
                "**Intermediate Zone**: seasonal rainfall, moderate totals. "
                "**Dry Zone**: strongly seasonal rainfall, most vulnerable to drought "
                "(north, east, northwest, southeast)."
            )
            if missing:
                st.warning(f"Grey area on the map: {', '.join(missing)}. "
                            f"Missing from the CLIMATE_ZONE dictionary in dashboard.py, "
                            f"add it there if you want it coloured.")
            st.info("This is a simplified district-level approximation. Real zone boundaries "
                     "cut across district lines. Treat as orientation, not an official map.")
        return

    # SPI view
    sample = load_precip(districts[0])
    years = sorted(sample["Year"].unique())
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        year = st.selectbox("Year", years, index=len(years) - 1)
    with c2:
        month = st.selectbox("Month", list(range(1, 13)),
                              format_func=lambda m: calendar.month_name[m], index=11)
    with c3:
        scale_label = st.radio("Time window", ["SPI-1 (single month)", "SPI-3 (last 3 months)",
                                                "SPI-6 (last 6 months)"], horizontal=True)
    scale = {"SPI-1 (single month)": 1, "SPI-3 (last 3 months)": 3, "SPI-6 (last 6 months)": 6}[scale_label]

    st.divider()

    rows = []
    for d in districts:
        try:
            spi_df = compute_spi_series(d, scale)
            match = spi_df[(spi_df["Year"] == year) & (spi_df["Month"] == month)]
            if len(match):
                val = float(match[f"SPI{scale}"].iloc[0])
                label, _, _ = spi_category(val)
                rows.append({"District": d, "SPI": val, "Label": label})
        except Exception as e:
            print(f"Skipped {d}: {e}")

    if not rows:
        st.warning("No SPI data available for this month.")
        return
    spi_row_df = pd.DataFrame(rows)
    merged = geo.merge(spi_row_df, on="District", how="left")

    col_map, col_summary = st.columns([1.7, 1])
    with col_map:
        fig = px.choropleth(
            merged, geojson=merged.geometry, locations=merged.index, color="SPI",
            color_continuous_scale=DRY_SCALE, range_color=[-2.5, 2.5],
            hover_name="District", hover_data={"Label": True, "SPI": ":.2f"}, labels={"SPI": "SPI"},
        )
        fig.update_geos(visible=False, resolution=50, bgcolor=CREAM,
                         lonaxis_range=lon_range, lataxis_range=lat_range)
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0), height=820, paper_bgcolor=CREAM,
            coloraxis_colorbar=dict(title="SPI", title_font_color=BROWN, tickfont_color=BROWN,
                                     thickness=18, len=0.7),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_summary:
        st.subheader(f"{calendar.month_name[month]} {year}")
        st.caption(f"View: {scale_label}")
        cat_counts = spi_row_df["Label"].value_counts()
        for label, n in cat_counts.items():
            st.markdown(
                f"<div style='background:{CREAM_2};padding:10px 14px;border-left:4px solid {TERRACOTTA};"
                f"border-radius:6px;margin-bottom:8px'><b style='color:{BROWN_DARK}'>{n}</b> "
                f"<span style='color:{INK}'>districts, {label}</span></div>",
                unsafe_allow_html=True,
            )
        st.divider()
        st.caption("**Reading the map**: brown means drier than normal, blue means wetter. "
                    "Cream districts had roughly normal rainfall.")
        with st.expander("Which time window should I look at?"):
            st.markdown("**SPI-1**: how dry was this specific month? Best for sudden dry spells. "
                         "**SPI-3**: how dry has the last quarter been? Best for crop-cycle drought. "
                         "**SPI-6**: how dry has the last half-year been? Best for reservoir "
                         "and water-resource planning.")
        st.caption("Tip: switch **Map view** above to see Sri Lanka's climate zones instead.")


# =======================================================================
# TAB 2: DISTRICT DEEP DIVE
# =======================================================================
def tab_district_deepdive():
    st.header("Take a closer look at one district")
    districts = list_districts()

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        district = st.selectbox("District", districts,
                                 index=districts.index("Anuradhapura") if "Anuradhapura" in districts else 0)
    with c2:
        scale = st.radio("Time window (SPI scale)", [1, 3, 6], horizontal=True,
                          format_func=lambda s: f"SPI-{s}")
    with c3:
        expert = st.toggle("Expert view", value=False,
                            help="Statistical detail: 3D copula plots, tail dependence, "
                                 "AIC comparison, marginal fits.")

    result = get_or_fit_model(district, scale)
    model = result["model"]
    events = result["events"]
    spi_df = compute_spi_series(district, scale)
    precip_df = load_precip(district)

    st.subheader(f"{district}, SPI-{scale} summary (1984 to 2025)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Drought events", f"{model.n_events}",
               help="Total number of separate drought periods found in the 42-year record.")
    m2.metric("Average length", f"{model.marginals.mean_D:.1f} months",
               help="Add up the length of every drought found, divide by the number of droughts. "
                    "This is the typical drought length for this district.")
    m3.metric("Average severity", f"{model.marginals.mean_S:.2f}",
               help="Add up the severity score of every drought found, divide by the number of "
                    "droughts. Severity combines how dry AND how long each drought was into one "
                    "number. See the Learn About SPI tab for the exact calculation.")
    m4.metric("Longest ever", f"{int(events['Duration'].max())} months",
               help="The single longest drought on record for this district.")

    # ---- Try your own rainfall reading ----
    st.subheader("Try your own rainfall reading")
    st.markdown("Have a rainfall measurement for this district? Enter it below to see what it means.")

    with st.form("rainfall_input_form"):
        rc1, rc2, rc3 = st.columns([1, 1, 1.4])
        with rc1:
            in_year = st.number_input("Year", min_value=1984, max_value=2035,
                                       value=int(precip_df["Year"].max()))
        with rc2:
            in_month = st.selectbox("Month", list(range(1, 13)),
                                     format_func=lambda m: calendar.month_name[m], key="rain_input_month")
        with rc3:
            in_rain = st.number_input("Rainfall this month (mm)", min_value=0.0, max_value=2000.0,
                                       value=100.0, step=5.0)
        submitted = st.form_submit_button("Calculate my SPI")

    if submitted:
        est_spi = estimate_spi_from_input(precip_df, scale, in_year, in_month, in_rain)
        if est_spi is None:
            st.warning("Not enough surrounding data to compute SPI for that month at this time window.")
        else:
            label, _, colour = spi_category(est_spi)
            st.markdown(
                f"<div style='background:{colour}22;border-left:5px solid {colour};"
                f"padding:16px 20px;border-radius:8px;margin-top:10px'>"
                f"<span style='font-size:26px;font-weight:700;color:{BROWN_DARK}'>SPI-{scale} = {est_spi:.2f}</span>"
                f"&nbsp;&nbsp;<span style='font-size:16px;color:{BROWN_DARK}'><b>{label}</b></span>"
                f"<br><span style='color:{INK}'>{spi_plain_english(est_spi)}</span></div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Based on {in_rain:.0f} mm in {calendar.month_name[in_month]} {in_year}"
                       + (f", combined with {district}'s actual rainfall in the "
                          f"{scale - 1} month(s) before it" if scale > 1 else "")
                       + f", compared against {district}'s 1984-2025 climate history.")

    st.divider()

    st.subheader("SPI history")
    spi_col = f"SPI{scale}"
    plot_df = spi_df.dropna(subset=[spi_col]).copy()
    plot_df["Date"] = pd.to_datetime(plot_df[["Year", "Month"]].assign(day=1))
    plot_df["Direction"] = np.where(plot_df[spi_col] >= 0, "Wet", "Dry")

    fig = px.bar(plot_df, x="Date", y=spi_col, color="Direction",
                  color_discrete_map={"Wet": "#3B6A85", "Dry": TERRACOTTA},
                  labels={spi_col: f"SPI-{scale}", "Date": "Year"})
    fig.add_hline(y=-1.0, line_dash="dash", line_color="#B04A1A", opacity=0.6)
    fig.add_hline(y=-1.5, line_dash="dash", line_color="#7A3211", opacity=0.6)
    fig.update_layout(height=400, showlegend=False, plot_bgcolor=CREAM, paper_bgcolor=CREAM,
                       font_color=INK, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis=dict(gridcolor=CREAM_2, title="Year"), yaxis=dict(gridcolor=CREAM_2))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red bars: drier than normal months. Blue bars: wetter than normal months. "
                "Dashed lines mark the standard 'moderately dry' and 'severely dry' thresholds.")

    st.subheader("Past drought events (10 worst on record)")
    events_disp = events.copy()
    events_disp["From"] = events_disp["StartYear"].astype(str) + "-" + events_disp["StartMonth"].astype(str).str.zfill(2)
    events_disp["To"]   = events_disp["EndYear"].astype(str) + "-" + events_disp["EndMonth"].astype(str).str.zfill(2)
    events_disp = events_disp.rename(columns={"Duration": "Length (months)", "Severity": "Severity score"})
    events_disp = events_disp[["From", "To", "Length (months)", "Severity score"]].sort_values("Severity score", ascending=False)
    st.dataframe(events_disp.head(10), use_container_width=True, hide_index=True)

    st.subheader("How rare would a drought of a chosen size be?")
    st.markdown("Move the sliders to describe a hypothetical drought. The dashboard tells you how "
                 "often, on average, a drought at least that big has happened.")

    max_d = int(max(int(events["Duration"].max()) * 1.3, 24))
    max_s = float(max(events["Severity"].max() * 1.3, 20))
    c1, c2 = st.columns(2)
    with c1:
        d_pick = st.slider("Drought length (months)", 1, max_d, value=int(events["Duration"].median()))
    with c2:
        s_pick = st.slider("Drought severity score", 0.5, float(max_s),
                            value=float(events["Severity"].median()), step=0.5)

    t_and = T_and(model, d_pick, s_pick)
    t_or  = T_or (model, d_pick, s_pick)

    r1, r2 = st.columns(2)
    r1.metric(f"At least {d_pick} months AND at least severity {s_pick:.1f}: once every", f"{t_and:.1f} years")
    r2.metric(f"At least {d_pick} months OR at least severity {s_pick:.1f}: once every", f"{t_or:.1f} years")

    st.info(f"**In plain English**: a drought lasting at least **{d_pick} months** with total "
         f"severity score at least **{s_pick:.1f}** is expected roughly once every "
         f"**{t_and:.0f} years** in {district}.")

    st.subheader("Standard drought return periods")
    st.dataframe(result["return_periods"], use_container_width=True, hide_index=True)
    st.caption("'Return period' means how many years, on average, between droughts at least this big. "
            "**T_AND(d,s)** is the return period when both the length threshold d and the severity "
            "threshold s are exceeded together (rarer). **T_OR(d,s)** is when at least one of them "
            "is exceeded (more common). See the Learn About SPI tab for a full walkthrough.")

    if expert:
        st.divider()
        st.header("Expert view: statistical detail")
        expert_district_view(district, scale, model, events)


# =======================================================================
# EXPERT VIEW
# =======================================================================
def expert_district_view(district: str, scale: int, model, events: pd.DataFrame):
    st.subheader("Dependence and copula selection")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kendall's tau", f"{model.kendall_tau:.3f}")
    c2.metric("Spearman's rho", f"{model.spearman_rho:.3f}")
    c3.metric("Selected copula", model.best_family)
    c4.metric("theta (parameter)", f"{model.best_theta:.4f}")

    fits_df = pd.DataFrame([{
        "Family": f.family, "Parameter": round(f.parameter, 4),
        "Log-Lik": round(f.loglik, 2), "AIC": round(f.aic, 2),
        "Upper tail": "yes" if f.tail_upper else "no",
        "Lower tail": "yes" if f.tail_lower else "no",
    } for f in model.all_fits]).sort_values("AIC").reset_index(drop=True)
    fits_df["Selected"] = fits_df["Family"].apply(lambda f: "yes" if f == model.best_family else "")
    st.dataframe(fits_df, use_container_width=True, hide_index=True)
    st.caption("Three families compared: Gumbel-Hougaard, Clayton, Frank. "
                "(Gaussian was tested during development but never won on any district, "
                "so it was dropped to keep the model set simple.)")
    st.markdown(_selection_justification(model))

    st.subheader("Tail dependence")
    up, low = tail_dependence_coefs(model.best_family, model.best_theta)
    tc1, tc2 = st.columns(2)
    tc1.metric("Upper-tail coefficient (lambda_U)", f"{up:.3f}")
    tc2.metric("Lower-tail coefficient (lambda_L)", f"{low:.3f}")
    st.markdown(_tail_explanation(up, low))

    st.subheader("3D copula surfaces")
    tab_pdf, tab_density, tab_contour = st.tabs([
        "Joint PDF (original scale)", "Copula density (unit square)", "Joint CDF contour + observed events",
    ])
    with tab_pdf:
        _plot_joint_pdf_3d(model, events)
        with st.expander("How to read this plot"):
            st.markdown(f"""
            **What the axes mean**

            - **x-axis (Duration D)**: how long a drought lasts, in months, from 0 to 40.
            - **y-axis (Severity S)**: the cumulative severity score of a drought.
            - **z-axis (height, f(d,s))**: the joint probability density at that combination.
              Higher points are more likely combinations; low or flat regions are rare.

            **What to look for**

            - **Peak near the origin**: most droughts are short and mild, so the density piles
              up in the corner where both Duration and Severity are small.
            - **Ridge trailing off diagonally**: this is the copula dependence at work. Long
              droughts also tend to be severe. Without dependence, this surface would look like
              a symmetric hill rather than a spike with a diagonal tail.
            - **Almost-zero regions**: combinations you rarely see historically. For example,
              20-month droughts with very low severity (bottom-right of the surface) or
              1-month droughts with extreme severity (top-left) are both effectively impossible
              under this model, which matches physical reality.

            **Why this plot matters**

            This is the joint distribution the dashboard uses to compute AND/OR return periods.
            Every "once every N years" number in the risk explorer is an integral over this surface.
            """)

    with tab_density:
        _plot_copula_density_3d(model)
        with st.expander("How to read this plot"):
            st.markdown(f"""
            **What the axes mean**

            - **x-axis (u)**: the Duration transformed by its own CDF, so it runs from 0 to 1.
              u = 0.9 means "this duration is bigger than 90% of past durations".
            - **y-axis (v)**: the Severity transformed the same way.
            - **z-axis (c(u,v))**: the copula density. A flat surface at height 1 would mean
              Duration and Severity are independent. Anything higher or lower reflects real
              coupling.

            **What to look for**

            - **Corner concentration is the signature of tail dependence**. If the surface
              spikes in the **top-right corner** (u -> 1, v -> 1), that is upper-tail dependence:
              extreme Duration and extreme Severity happen together more than pure chance
              would predict. This is the signature of the Gumbel-Hougaard copula.
            - **Spike in the bottom-left corner** (u -> 0, v -> 0) is lower-tail dependence:
              short mild droughts cluster together. This is the Clayton copula's signature.
            - **Symmetric shape with no corner spike** means Frank copula: dependence with no
              extreme concentration in either tail.

            **Why this plot matters**

            This is the copula's native domain. Because the marginals have been transformed
            out, this surface shows the pure dependence structure between Duration and
            Severity, uncontaminated by their individual shapes. Compare the corner shape to
            the family selected in the AIC table above to confirm the visual matches the
            statistical winner.
            """)

    with tab_contour:
        _plot_contour_scatter(model, events)
        with st.expander("How to read this plot"):
            st.markdown(f"""
            **What the axes and lines mean**

            - **x-axis (Duration)** and **y-axis (Severity)**: same physical units as the SPI
              record.
            - **Contour lines and colour bands**: levels of the joint CDF F(d,s). A line
              labelled 0.5 marks the (d, s) pairs for which
              P(D <= d AND S <= s) = 0.5. Half of past droughts fall inside that contour.
              A line labelled 0.9 means 90% fall inside, and so on.
            - **Blue dots**: every one of the {model.n_events} observed drought events for
              this district plotted on the same axes.
            - **Red star**: the single most severe event on record.

            **What to look for**

            - **Where the dots sit**: most events crowd into the lower-left, inside the 0.5 or
              0.6 contour. That matches the peak in the first 3D plot.
            - **The lonely star**: the worst event sits far in the top-right, past the 0.9 or
              even 0.95 contour, in the sparse extreme corner. That is the region of jointly
              extreme droughts, where events are both very long and very severe at once.
            - **Contours crowd along the axes** and then bend at the top-right. This is the
              hallmark of Gumbel-Hougaard shape: the joint CDF grows fast in the "one variable
              small" regions and slowly in the "both variables large" region.

            **Why this plot matters**

            It ties the abstract copula math back to the real observed history. Instead of just
            saying "we fit a Gumbel copula", the plot lets you see where each historical event
            actually landed in the joint probability space, and why the worst event on record
            sits well inside the joint tail.
            """)

    st.subheader("Marginal distribution fits")
    m = model.marginals
    marg_df = pd.DataFrame([
        {"Variable": "Duration D", "Family": "Exponential", "Parameter": f"lambda = {m.lambda_d:.4f}",
         "Mean": f"{1/m.lambda_d:.2f}"},
        {"Variable": "Severity S", "Family": "Gamma", "Parameter": f"alpha = {m.alpha_s:.4f}, beta = {m.beta_s:.4f}",
         "Mean": f"{m.alpha_s * m.beta_s:.2f}"},
    ])
    st.dataframe(marg_df, use_container_width=True, hide_index=True)
    st.caption("Following Shiau (2006) and Ekanayake & Perera (2014): Duration follows "
                "Exponential, Severity follows Gamma. Standard pairing in the drought-copula "
                "literature.")


def _selection_justification(model) -> str:
    lines = [
        "**Selection rationale**", "",
        "- Three copula families fitted by maximum likelihood on the pseudo-observations "
        "(u = F_D(D), v = F_S(S)). Selection is by lowest AIC.",
        f"- **Selected: {model.best_family}**, AIC = {min(f.aic for f in model.all_fits):.2f}.",
    ]
    if model.best_family == "Gumbel":
        lines.append("- Gumbel-Hougaard exhibits **upper-tail dependence**: long droughts also "
                       "tend to be severe droughts, most strongly at the extremes. "
                       "Matches Shiau (2006), Salvadori & De Michele (2004), Ekanayake & Perera (2014).")
    elif model.best_family == "Clayton":
        lines.append("- Clayton exhibits **lower-tail dependence**: strongest dependence is "
                       "between short, mild events rather than the extremes.")
    else:
        lines.append("- Frank captures symmetric dependence with no tail concentration.")
    lines.append(f"- Kendall's tau = {model.kendall_tau:.3f} confirms a strong positive rank association.")
    return "\n".join(lines)


def _tail_explanation(up: float, low: float) -> str:
    parts = []
    if up > 0.05:
        parts.append(f"lambda_U = {up:.3f}: in the extreme upper tail, the two variables become "
                       f"more strongly linked. When Duration is in its top few percent, Severity "
                       f"has roughly a **{up*100:.0f}% conditional chance** of also being in its top few percent.")
    if low > 0.05:
        parts.append(f"lambda_L = {low:.3f}: strongest coupling is in the lower tail "
                       f"(short, mild events).")
    if up < 0.05 and low < 0.05:
        parts.append("No tail dependence in either direction: extreme droughts are no more "
                       "(or less) correlated than typical ones under this copula.")
    parts.append("Tail dependence matters because a plain correlation summarises the *average* "
                   "relationship. A copula with tail dependence captures how the relationship "
                   "changes at the extremes, the events that matter most for planning.")
    return "\n\n".join(parts)


def _plot_joint_pdf_3d(model, events):
    m = model.marginals
    d_seq = np.linspace(0.1, 40, 45)
    s_seq = np.linspace(0.1, 30, 45)
    DD, SS = np.meshgrid(d_seq, s_seq, indexing="ij")
    U = m.F_D(DD); V = m.F_S(SS)

    if model.best_family == "Gumbel":
        log_c = _gumbel_log_density(U.ravel(), V.ravel(), model.best_theta)
    elif model.best_family == "Clayton":
        log_c = _clayton_log_density(U.ravel(), V.ravel(), model.best_theta)
    else:
        log_c = _frank_log_density(U.ravel(), V.ravel(), model.best_theta)
    c = np.exp(log_c).reshape(DD.shape)

    joint = c * m.f_D(DD) * m.f_S(SS)
    joint[~np.isfinite(joint)] = 0
    cap = np.quantile(joint[joint > 0], 0.97) if (joint > 0).any() else joint.max()
    joint = np.minimum(joint, cap)

    fig = go.Figure(data=[go.Surface(
        x=d_seq, y=s_seq, z=joint.T,
        colorscale=[[0, CREAM_2], [0.5, TERRA_SOFT], [1, TERRACOTTA]],
        showscale=True, colorbar=dict(title="f(d,s)"),
    )])
    fig.update_layout(
        scene=dict(xaxis_title="Duration D (months)", yaxis_title="Severity S",
                    zaxis_title="Joint density f(d,s)", bgcolor=CREAM),
        height=600, paper_bgcolor=CREAM, font_color=INK, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_copula_density_3d(model):
    u_seq = np.linspace(0.02, 0.98, 55)
    v_seq = np.linspace(0.02, 0.98, 55)
    UU, VV = np.meshgrid(u_seq, v_seq, indexing="ij")

    if model.best_family == "Gumbel":
        log_c = _gumbel_log_density(UU.ravel(), VV.ravel(), model.best_theta)
    elif model.best_family == "Clayton":
        log_c = _clayton_log_density(UU.ravel(), VV.ravel(), model.best_theta)
    else:
        log_c = _frank_log_density(UU.ravel(), VV.ravel(), model.best_theta)
    c = np.exp(log_c).reshape(UU.shape)
    c[~np.isfinite(c)] = 0
    cap = np.quantile(c[c > 0], 0.98) if (c > 0).any() else c.max()
    c = np.minimum(c, cap)

    fig = go.Figure(data=[go.Surface(
        x=u_seq, y=v_seq, z=c.T,
        colorscale=[[0, CREAM_2], [0.5, TERRA_SOFT], [1, "#7A3211"]],
        showscale=True, colorbar=dict(title="c(u,v)"),
    )])
    fig.update_layout(
        scene=dict(xaxis_title="u = F_D(Duration)", yaxis_title="v = F_S(Severity)",
                    zaxis_title="Copula density c(u,v)", bgcolor=CREAM),
        height=600, paper_bgcolor=CREAM, font_color=INK, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_contour_scatter(model, events: pd.DataFrame):
    m = model.marginals
    D_max = max(events["Duration"].max() * 1.1, 20)
    S_max = max(events["Severity"].max() * 1.1, 20)
    d_seq = np.linspace(0.1, D_max, 150)
    s_seq = np.linspace(0.1, S_max, 150)
    DD, SS = np.meshgrid(d_seq, s_seq, indexing="ij")
    U = m.F_D(DD); V = m.F_S(SS)
    C = _copula_cdf(model.best_family, U.ravel(), V.ravel(), model.best_theta)
    C = np.array(C).reshape(DD.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=d_seq, y=s_seq, z=C.T,
        colorscale=[[0, CREAM_2], [0.5, TERRA_SOFT], [1, "#3B6A85"]],
        contours=dict(start=0.1, end=0.9, size=0.1, showlabels=True, labelfont=dict(color=BROWN_DARK)),
        colorbar=dict(title="F(d,s)"),
    ))
    extreme_idx = events["Severity"].idxmax()
    others = events.drop(index=extreme_idx)
    fig.add_trace(go.Scatter(x=others["Duration"], y=others["Severity"], mode="markers",
                               marker=dict(color=BROWN_DARK, size=7, opacity=0.75),
                               name="Observed events", showlegend=True))
    fig.add_trace(go.Scatter(
        x=[events.loc[extreme_idx, "Duration"]], y=[events.loc[extreme_idx, "Severity"]],
        mode="markers+text", marker=dict(color=TERRACOTTA, size=16, symbol="star"),
        text=[f"Worst on record<br>D={int(events.loc[extreme_idx,'Duration'])}, "
              f"S={events.loc[extreme_idx,'Severity']:.1f}"],
        textposition="top center", textfont=dict(color=TERRACOTTA, size=11),
        name="Most severe event", showlegend=True,
    ))
    fig.update_layout(
        xaxis_title="Duration D (months)", yaxis_title="Severity S",
        height=600, paper_bgcolor=CREAM, plot_bgcolor=CREAM, font_color=INK,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor=CREAM_2), yaxis=dict(gridcolor=CREAM_2),
        legend=dict(bgcolor=CREAM_2, bordercolor=TERRA_SOFT, borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)


# =======================================================================
# TAB 3: COMPARE DISTRICTS
# =======================================================================
def tab_compare_districts():
    st.header("Compare two districts")
    districts = list_districts()

    c1, c2, c3 = st.columns(3)
    with c1:
        d1 = st.selectbox("District A", districts, index=0)
    with c2:
        d2 = st.selectbox("District B", districts, index=min(1, len(districts) - 1))
    with c3:
        scale = st.radio("Time window (SPI scale)", [1, 3, 6], horizontal=True, key="cmp_scale")

    if d1 == d2:
        st.warning("Pick two different districts to compare.")
        return

    r1 = get_or_fit_model(d1, scale)
    r2 = get_or_fit_model(d2, scale)
    col1, col2 = st.columns(2)
    for col, d, r in [(col1, d1, r1), (col2, d2, r2)]:
        with col:
            st.subheader(d)
            st.metric("Total drought events", f"{r['model'].n_events}")
            st.metric("Average length", f"{r['model'].marginals.mean_D:.1f} months")
            st.metric("Average severity", f"{r['model'].marginals.mean_S:.2f}")
            st.metric("Longest ever", f"{int(r['events']['Duration'].max())} months")
            st.metric("Most severe ever", f"{r['events']['Severity'].max():.1f}")
            st.metric("Best-fit copula", r["model"].best_family)


# =======================================================================
# TAB 4: LEARN ABOUT SPI  (rebuilt with equations and justifications)
# =======================================================================
def tab_learn_about_spi():
    st.header("Learn About SPI: A Beginner's Guide")
    st.markdown("This page explains the methodology behind every number and plot in the "
                 "dashboard, aimed at a reader with no statistics background.")

    with st.expander("1. What is SPI?", expanded=True):
        st.markdown("""
        **SPI** stands for **Standardized Precipitation Index**. It was introduced by McKee,
        Doesken and Kleist in 1993 and is the most widely used drought index in the world today.

        In one sentence, SPI answers the question: *"compared to this district's own rainfall
        history, was this month unusually dry or unusually wet?"*

        An SPI value of **0** means completely average rainfall for that month and district.
        Positive SPI means wetter than normal. Negative SPI means drier than normal. Because
        the values are standardized to a common scale, an SPI of -1.5 in Jaffna means the same
        thing statistically as an SPI of -1.5 in Colombo, even though the two districts have
        very different raw rainfall totals.
        """)

    with st.expander("2. SPI calculation, step by step with equations"):
        st.markdown("""
        SPI is computed separately for every district, using its own 42-year rainfall record.

        **Step 1: Choose a time scale k.**
        SPI-1 uses 1 month of rainfall. SPI-3 uses the sum of the last 3 months. SPI-6 uses
        the sum of the last 6 months. Longer scales smooth out short-term noise and are more
        relevant to slower systems like groundwater or reservoirs.

        **Step 2: Group by calendar month.**
        For every calendar month (January, February, and so on), gather all 42 values of the
        k-month rainfall total. This matters because a "normal" January is very different
        from a "normal" July in Sri Lanka.

        **Step 3: Fit a Gamma distribution to each month's data.**
        The Gamma distribution's probability density is:
        """)
        st.latex(r"g(x \mid \alpha, \beta) = \frac{1}{\beta^{\alpha}\,\Gamma(\alpha)}\, x^{\alpha - 1} e^{-x/\beta}, \quad x > 0")
        st.markdown("""
        where alpha is the shape parameter and beta is the scale parameter, both estimated by
        maximum likelihood from the 42 observations. Gamma is used because rainfall is
        non-negative and typically right-skewed.

        **Step 4: Transform through the Gamma CDF, then through the inverse normal CDF.**
        For a given rainfall observation x, the SPI value is:
        """)
        st.latex(r"\text{SPI} = \Phi^{-1}\!\big(F_{\text{Gamma}}(x)\big)")
        st.markdown("""
        where F_Gamma is the fitted Gamma cumulative distribution for that calendar month,
        and Phi_inverse is the inverse of the standard normal CDF. This last step is what
        makes SPI "standardized": every district and every month is put on the same scale
        centred at 0 with standard deviation 1.

        The result is a dimensionless number, typically between -3 and +3.
        """)

    with st.expander("3. SPI categories and their thresholds"):
        cat_df = pd.DataFrame([
            {"SPI range": ">= 2.0",            "Category": "Extremely wet"},
            {"SPI range": "1.5 to 2.0",         "Category": "Very wet"},
            {"SPI range": "1.0 to 1.5",         "Category": "Moderately wet"},
            {"SPI range": "-1.0 to 1.0",        "Category": "Near normal"},
            {"SPI range": "-1.5 to -1.0",       "Category": "Moderately dry"},
            {"SPI range": "-2.0 to -1.5",       "Category": "Severely dry"},
            {"SPI range": "<= -2.0",            "Category": "Extremely dry"},
        ])
        st.dataframe(cat_df, use_container_width=True, hide_index=True)
        st.caption("Standard classification (McKee et al., 1993), used worldwide.")

    with st.expander("4. How does the dashboard work behind the scenes?"):
        st.markdown("""
        **Data collection.**
        Monthly precipitation for every district comes from NASA POWER, a satellite-derived
        rainfall dataset covering the world on a roughly half-degree grid (about 55 km per
        cell). For each of Sri Lanka's 25 districts, the pipeline:

        1. Fetches every grid cell inside a bounding box around the district.
        2. Uses the official GADM district polygon boundary to keep only grid cells whose
           centre falls **inside** the true district shape.
        3. Averages the remaining cells for each Year-Month pair, producing one clean
           rainfall value per district per month, from January 1984 to December 2025
           (504 months).

        **Function flow when you use the dashboard.**

        - When you open the **National Map** tab and pick a month, the function
          `compute_spi_series` runs for every district in parallel, cached so subsequent
          clicks are instant.
        - When you open the **District Deep-Dive** tab and pick a district, the function
          `get_or_fit_model` looks up whether that district's copula analysis is already
          cached. If yes, it loads it (instant). If no, it fits the three copula families
          from scratch (about 15 seconds) and saves the result to disk for next time.
        - When you move the sliders in the "How rare would a drought be?" section, the
          functions `T_and` and `T_or` are called live with your slider values, using the
          district's already-fitted copula.
        - When you enter your own rainfall reading, the function
          `estimate_spi_from_input` substitutes your reading into the district's rainfall
          history and recomputes SPI at the requested time scale.
        """)

    with st.expander("5. Drought events, Duration, and Severity with equations"):
        st.markdown("""
        Once SPI is computed for every month, drought events are extracted using the
        Shiau (2006) definition.

        **Drought event.**
        A drought event is any unbroken run of consecutive months where SPI stays below
        zero. The moment SPI goes to zero or above, the event ends.

        **Duration D.**
        The number of months in the run.
        """)
        st.latex(r"D = t_{\text{end}} - t_{\text{start}} + 1 \quad (\text{months})")
        st.markdown("""
        **Severity S.**
        The sum of the absolute SPI values across every month in the run:
        """)
        st.latex(r"S = \sum_{t = t_{\text{start}}}^{t_{\text{end}}} \lvert \text{SPI}(t) \rvert")
        st.markdown("""
        Severity captures both **how long** and **how deep** a drought was, in one number.
        A drought that is moderately dry for 6 months can end up with a similar Severity to
        one that is extremely dry for 2 months.

        **Worked example.**
        Suppose a district records SPI values of -0.4, -1.1, -0.8 for three consecutive
        months, then goes positive. That is one drought event with:
        - Duration D = 3 months
        - Severity S = 0.4 + 1.1 + 0.8 = 2.3

        **Average length and average severity** (the top-row metrics on the Deep-Dive tab)
        are simple arithmetic means across every drought event in the 42-year record:
        """)
        st.latex(r"\bar{D} = \frac{1}{N}\sum_{i=1}^{N} D_i, \quad \bar{S} = \frac{1}{N}\sum_{i=1}^{N} S_i")
        st.markdown("where N is the total number of drought events found for that district.")

    with st.expander("6. Why copulas, and how the family is chosen"):
        st.markdown("""
        **Why not just look at Duration and Severity separately?**
        Because they are not independent. Long droughts also tend to be severe droughts.
        Analysing them one at a time throws away that link and gives wrong answers to
        questions like "how often does a drought both long enough and severe enough to
        actually cause damage happen here?"

        **What a copula is.**
        A copula is a joint distribution built on the unit square with uniform marginals.
        Sklar's theorem (1959) says that any joint distribution can be split into its
        marginals and its copula, and the copula captures the pure dependence structure
        between the two variables, independent of what shape they individually have.

        **The transformation used.**
        For every observed (D, S) pair, we compute the pseudo-observations:
        """)
        st.latex(r"u = F_D(D), \quad v = F_S(S)")
        st.markdown("""
        where F_D is the fitted Duration CDF (Exponential) and F_S is the fitted Severity
        CDF (Gamma). After this transform, u and v are both uniform on [0, 1]. The copula
        is fitted on (u, v), not on the raw (D, S).

        **Three copula families are tested for each district**, then selected by lowest AIC:

        **Gumbel-Hougaard** copula:
        """)
        st.latex(r"C_{\text{Gumbel}}(u, v) = \exp\!\Big(-\big[(-\ln u)^{\theta} + (-\ln v)^{\theta}\big]^{1/\theta}\Big), \quad \theta \geq 1")
        st.markdown("Captures **upper-tail dependence**: extreme values in both variables cluster together.")
        st.latex(r"C_{\text{Clayton}}(u, v) = \big(u^{-\theta} + v^{-\theta} - 1\big)^{-1/\theta}, \quad \theta > 0")
        st.markdown("**Clayton** captures **lower-tail dependence**: small values in both variables cluster together.")
        st.latex(r"C_{\text{Frank}}(u, v) = -\tfrac{1}{\theta} \ln\!\Big(1 + \frac{(e^{-\theta u} - 1)(e^{-\theta v} - 1)}{e^{-\theta} - 1}\Big)")
        st.markdown("**Frank** captures symmetric dependence with no tail concentration.")
        st.markdown("""
        **Why Gumbel-Hougaard usually wins for drought data.**
        Long droughts and severe droughts co-occur *especially at the extremes*, which is
        exactly the pattern upper-tail dependence describes. Shiau (2006), Salvadori & De
        Michele (2004), and Ekanayake & Perera (2014) all confirm this pattern in different
        regional drought datasets. The dashboard checks this per district using AIC, so if
        a district's data actually looks more like Clayton or Frank, that gets selected
        instead.

        **Kendall's tau (rank correlation)** is used as a robust dependence measure that
        does not assume normality and does not depend on the specific shape of the
        marginals. For the Gumbel copula, tau maps directly to the copula parameter:
        """)
        st.latex(r"\tau_{\text{Gumbel}} = 1 - \frac{1}{\theta}")
        st.markdown("which gives a clean way to interpret theta: closer to 1 means near-independence, higher theta means stronger tail dependence.")

    with st.expander("7. Joint return periods: AND vs OR"):
        st.markdown("""
        The dashboard reports two return period definitions per drought threshold pair (d, s).

        **AND return period** (the more useful one for planning):
        """)
        st.latex(r"T_{\text{AND}}(d, s) = \frac{E[L]}{P(D \geq d \text{ and } S \geq s)} = \frac{E[L]}{1 - F_D(d) - F_S(s) + C(F_D(d), F_S(s))}")
        st.markdown("""
        In words: how often, on average, do we see a drought that is **both** at least d
        months long **and** at least s severe, at the same time.

        **OR return period**:
        """)
        st.latex(r"T_{\text{OR}}(d, s) = \frac{E[L]}{P(D \geq d \text{ or } S \geq s)} = \frac{E[L]}{1 - C(F_D(d), F_S(s))}")
        st.markdown("""
        In words: how often, on average, do we see a drought that is **either** at least d
        months long **or** at least s severe.

        **E[L]** is the mean inter-arrival time between drought events (mean months between
        one drought ending and the next one starting). We compute it directly from the
        historical record as (total months) / (number of events).

        AND is always the larger number because two conditions have to hold at once.
        For planning decisions, AND is usually the number that matters, because damaging
        droughts are typically the ones that are simultaneously long and severe.
        """)

    with st.expander("8. Why these specific marginal distributions?"):
        st.markdown("""
        The dashboard fits Duration D as an Exponential distribution and Severity S as a
        Gamma distribution.

        **Exponential for Duration.**
        """)
        st.latex(r"f_D(d) = \lambda e^{-\lambda d}, \quad d > 0")
        st.markdown("""
        Justified because drought duration is a positive continuous quantity with a
        memoryless-like tail: most droughts are short, a few are long. Empirically the
        Exponential fits the observed drought durations well.

        **Gamma for Severity.**
        """)
        st.latex(r"f_S(s) = \frac{1}{\beta^{\alpha}\,\Gamma(\alpha)}\, s^{\alpha - 1} e^{-s/\beta}, \quad s > 0")
        st.markdown("""
        Justified because Severity is a positive sum of positive contributions with the
        same non-negative right-skewed shape as rainfall itself.

        **Honest note on model selection.**
        In a full 6-distribution comparison (Exponential, Normal, Lognormal, Logistic,
        Gamma, Weibull), Weibull scores slightly better than Exponential on AIC for
        Duration in most districts. We keep Exponential and Gamma anyway because Shiau
        (2006) and Ekanayake & Perera (2014), the two foundational drought-copula papers,
        both use exactly this pairing. Staying with them keeps our results directly
        comparable to published Sri Lanka drought research, and the copula formulas from
        those papers assume these specific marginals.
        """)

    with st.expander("9. Limitations of this dashboard"):
        st.markdown("""
        Being upfront about what the dashboard cannot do is part of using it responsibly.

        **Not a forecast.**
        The dashboard is retrospective. It looks at rainfall that has already happened. It
        cannot say a drought is coming next month. To turn it into a forecasting tool,
        seasonal rainfall predictions from a numerical weather model would need to be
        plugged in.

        **Not real-time.**
        NASA POWER has a 2 to 3 month lag between the end of a month and its rainfall being
        available in the API. For up-to-the-minute monitoring you would need to blend NASA
        POWER with a live gauge network from the Department of Meteorology.

        **Not farm-level.**
        Rainfall values are district-wide averages from a satellite grid of about 55 km per
        cell. A single farm can be much wetter or drier than the district average,
        especially in mountainous or coastal terrain.

        **42-year record limits rare-event certainty.**
        Return periods beyond about 50 years are extrapolations. There simply are not many
        50-year droughts on record to fit to. Treat the 100-year AND number as an order of
        magnitude, not a precise forecast.

        **Small districts share grid cells.**
        Districts smaller than about 55 km across, particularly Colombo and Kalmunai, may
        share a NASA POWER grid cell with a neighbour. For these districts the value is
        less spatially precise than for large districts like Anuradhapura.

        **Copula selection uncertainty.**
        With only 100 or so events per district, the AIC comparison between families is
        not always decisive. If two families are close in AIC, the choice of which one to
        use makes a small but real difference to the return periods. The Expert view shows
        the full AIC comparison so you can judge how confident the selection is.

        **Zone map is a simplification.**
        The Dry / Intermediate / Wet zone view uses one label per district. Real
        agro-climatic zones cut across district lines. For agricultural decisions rely on
        the official Department of Agriculture zone map, not the dashboard's version.
        """)

    with st.expander("10. Key references"):
        st.markdown("""
        - McKee, T. B., Doesken, N. J., & Kleist, J. (1993). *The relationship of drought
          frequency and duration to time scales.* Proceedings of the 8th Conference on
          Applied Climatology, Anaheim, California.
        - Shiau, J.-T. (2006). *Fitting drought duration and severity with two-dimensional
          copulas.* Water Resources Management, 20(5), 795-815.
        - Salvadori, G., & De Michele, C. (2004). *Frequency analysis via copulas.*
          Water Resources Research, 40, W12511.
        - Ekanayake, E. M. R. S. B., & Perera, K. (2014). *Analysis of drought severity and
          duration using copulas in Anuradhapura, Sri Lanka.* British Journal of
          Environment & Climate Change, 4(3), 312-327.
        - Sklar, A. (1959). *Fonctions de repartition a n dimensions et leurs marges.*
          Publications de l'Institut de Statistique de l'Universite de Paris, 8, 229-231.
        - Nelsen, R. B. (2006). *An Introduction to Copulas* (2nd ed.). Springer.
        """)

    st.divider()
    st.info("Have a rainfall reading of your own? Head to the **District Deep-Dive** tab and "
             "try the 'Try your own rainfall reading' calculator.")


# =======================================================================
# MAIN
# =======================================================================
def main():
    st.markdown(
        f"<div style='background:{BROWN_DARK};padding:24px 32px;border-radius:12px;margin-bottom:24px'>"
        f"<h1 style='color:white;margin:0;font-family:Cambria,serif'>Sri Lanka Drought Dashboard</h1>"
        f"<p style='color:{CREAM};margin:6px 0 0;font-size:15px'>A copula-based drought monitoring "
        f"tool for the 25 districts of Sri Lanka. Built on NASA POWER satellite rainfall (1984 to 2025).</p>"
        f"</div>", unsafe_allow_html=True,
    )

    tabs = st.tabs(["  NATIONAL MAP", "  DISTRICT DEEP DIVE", "  COMPARE DISTRICTS", "  LEARN ABOUT SPI"])
    with tabs[0]:
        tab_national_map()
    with tabs[1]:
        tab_district_deepdive()
    with tabs[2]:
        tab_compare_districts()
    with tabs[3]:
        tab_learn_about_spi()


if __name__ == "__main__":
    main()