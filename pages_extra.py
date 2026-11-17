"""
pages_extra.py
==============
Dashboard pages built on the shared index tables (index_data.py) and the
shared look (ui.py).

Where it sits in the pipeline:

    index_data.load_tables  ->  this module  ->  dashboard.py (navigation)

    page_overview        headline status, the map, then national details
    national_map_tab     one GIS map tab, reused by the SPI, SSI and MSDI pages
    ssi_events_tab       SSI drought events only, with no copula or return periods
    page_ssi             the soil-moisture page
    page_compare         SPI vs SSI vs MSDI for a district and nationally

The sidebar (ui.sidebar_controls) supplies district, index, time window and
date range; every function here reads them with ui.selection().

SPI and SSI are dry at or below -1. The published MSDI is not standardised, so
its dry months come from its cut-offs: every table here carries a class and a
dry column, and the pages use those rather than a fixed line.
"""
from __future__ import annotations

import calendar

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import common
import compare_indices as CI
import events_panel
import index_data as D
import thresholds as TH
import ui

INDEX_COLOURS = {k: v["colour"] for k, v in D.INDEX_INFO.items()}
MAIN_INDICES = ("SPI", "SSI", "MSDI")

# Approximate district -> agro-climatic zone (orientation only)
CLIMATE_ZONE = {
    "Colombo": "Wet", "Gampaha": "Wet", "Kalutara": "Wet", "Galle": "Wet", "Matara": "Wet",
    "Kandy": "Wet", "Nuwara Eliya": "Wet", "Ratnapura": "Wet", "Kegalle": "Wet",
    "Kurunegala": "Intermediate", "Matale": "Intermediate", "Badulla": "Intermediate",
    "Moneragala": "Intermediate", "Hambantota": "Intermediate",
    "Jaffna": "Dry", "Kilinochchi": "Dry", "Mannar": "Dry", "Vavuniya": "Dry", "Mullaitivu": "Dry",
    "Trincomalee": "Dry", "Batticaloa": "Dry", "Ampara": "Dry", "Polonnaruwa": "Dry",
    "Anuradhapura": "Dry", "Puttalam": "Dry",
}
ZONE_COLOURS = {"Wet": "#2171B5", "Intermediate": "#9ECAE1", "Dry": "#C4561D"}


# ============================================================================= data access

@st.cache_resource(show_spinner="Preparing drought indices for all 25 districts (first run only)...")
def _bundle(version: float) -> dict:
    """Index tables for this version of the inputs (rebuilt when a source file changes)."""
    return D.load_tables()


def bundle() -> dict:
    """The current index tables bundle (see index_data.build_tables)."""
    return _bundle(D.sources_mtime())


@st.cache_resource(show_spinner=False)
def _classified(index: str, scale: int, version: float) -> pd.DataFrame:
    """One table with its classes, for this version of the inputs."""
    return D.classified_table(bundle()["tables"][(index, scale)], index, scale)


def table(index: str, scale: int) -> pd.DataFrame | None:
    """Long table of one index and scale for all districts, or None if missing."""
    if (index, scale) not in bundle()["tables"]:
        return None
    return _classified(index, scale, D.sources_mtime())


def class_order(index: str) -> list[str]:
    """The classes of one index, driest first."""
    return ui.MSDI_CLASS_ORDER if index == "MSDI" else ui.CLASS_ORDER


def latest_date(index: str = "SPI", scale: int = 1) -> pd.Timestamp:
    """Last month with data."""
    return table(index, scale).dropna(subset=["value"])["Date"].max()


def snapshot(index: str, scale: int, year: int, month: int) -> pd.DataFrame | None:
    """All districts for one month: District, value, class."""
    t = table(index, scale)
    if t is None:
        return None
    return t[(t["Year"] == year) & (t["Month"] == month)].dropna(subset=["value"])


def district_series(index: str, scale: int, district: str) -> pd.DataFrame:
    """Date, Year, Month and value of one index for one district."""
    t = table(index, scale)
    return t[t["District"] == district][["Date", "Year", "Month", "value"]]


def month_controls(key: str, show_colour: bool = True):
    """Year, month (and colour mode) in one row. Returns (year, month, colour)."""
    last = latest_date()
    years = list(range(common.START_YEAR, last.year + 1))
    cols = st.columns([1, 1, 1.4] if show_colour else [1, 1])
    year = cols[0].selectbox("Year", years, index=len(years) - 1, key=f"{key}_y")
    month = cols[1].selectbox("Month", list(range(1, 13)), index=last.month - 1, key=f"{key}_m",
                              format_func=lambda m: calendar.month_name[m])
    colour = "class"
    if show_colour:
        colour = {"Drought class": "class", "Index value": "value"}[
            cols[2].segmented_control("Colour by", ["Drought class", "Index value"],
                                      default="Drought class", key=f"{key}_c") or "Drought class"]
    return year, month, colour


# ============================================================================= map + ranking

def _class_counts_chart(s: pd.DataFrame, index: str):
    """Horizontal bars: number of districts in each drought class."""
    counts = s["class"].value_counts().reindex(class_order(index)).fillna(0).astype(int)
    fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h",
                           marker_color=[ui.CLASS_COLOURS[c] for c in counts.index],
                           marker_line=dict(color="white", width=2),
                           text=counts.values, textposition="outside",
                           hovertemplate="%{y}: %{x} districts<extra></extra>"))
    fig.update_layout(title="Districts per class", yaxis=dict(autorange="reversed"),
                      xaxis=dict(range=[0, max(counts.max() * 1.25, 5)]))
    return ui.style_fig(fig, 300, legend=False, x_title="number of districts", y_title="")


def _ranking_chart(s: pd.DataFrame, label: str, index: str):
    """Every district's value, driest at the top, coloured by class."""
    rank = s.sort_values("value").assign(**{"class": lambda t: t["class"].fillna("No data")})
    fig = go.Figure(go.Bar(x=rank["value"], y=rank["District"], orientation="h",
                           marker_color=[ui.CLASS_COLOURS.get(c, "#ccc") for c in rank["class"]],
                           marker_line=dict(color="white", width=1),
                           customdata=rank["class"],
                           hovertemplate="%{y}: %{x:.2f} (%{customdata})<extra></extra>"))
    if index != "MSDI":                         # MSDI cut-offs differ by district, so no single line
        fig.add_vline(x=-1, line_dash="dot", line_color=ui.DRY_LINE)
    fig.update_layout(title="All districts, driest first", yaxis=dict(autorange="reversed"))
    unit = "index value" if index == "MSDI" else "standard deviations"
    return ui.style_fig(fig, 580, legend=False, x_title=f"{label} ({unit})", y_title="")


def map_with_ranking(index: str, s: pd.DataFrame, scale: int, colour: str, when: str, key: str,
                     height: int = 700):
    """Big district map of one index and month, with class counts and a ranking beside it."""
    info = D.INDEX_INFO[index]
    label = f"{index}-{scale}"
    cmap, cside = st.columns([1.55, 1])
    with cmap:
        fig = ui.district_map(s, colour_by=colour, height=height, value_label=label, class_order=class_order(index),
                              title=f"{info['name']}, {when}, {scale}-month window")
        if colour == "class":
            fig.update_layout(showlegend=False)          # the class legend sits under the map
        ui.show_map(fig, key=f"{key}_map",
                    caption=("Classes use the paper's Table 4 cut-offs, the same for every district. "
                             if index == "MSDI" else "Brown districts are drier than normal for this time of "
                             "year, blue ones wetter. ") + "Hover a district for its value; scroll to zoom.")
        if colour == "class":
            ui.class_legend(class_order(index))
    with cside:
        ui.show_chart(_class_counts_chart(s, index), key=f"{key}_counts")
        ui.show_chart(_ranking_chart(s, label, index),
                      "Bar colours are each district's class." if index == "MSDI" else
                      "Bars left of the dotted line (-1) are districts in drought.", key=f"{key}_rank")


def national_map_tab(index: str, key: str, note_text: str = "", scale: int | None = None):
    """GIS map tab of one index for a chosen month, or the climate zones.

    scale: the time window, from the sidebar unless given (MSDI has its own T).
    """
    scale = scale or ui.selection()["scale"]
    year, month, colour = month_controls(key)
    view = st.segmented_control("Show", ["Drought index", "Climate zones"], default="Drought index",
                                key=f"{key}_view") or "Drought index"
    if view == "Climate zones":
        z = pd.DataFrame({"District": list(CLIMATE_ZONE), "zone": list(CLIMATE_ZONE.values())})
        fig = ui.district_map(z, class_col="zone", class_order=list(ZONE_COLOURS),
                              class_colours={**ZONE_COLOURS, "No data": "#D0D0D0"}, height=680,
                              discrete_title="Climate zone", title="Agro-climatic zones (simplified)")
        ui.show_map(fig, key=f"{key}_zones",
                    caption="One zone per district, for orientation only: real zone boundaries cut across "
                            "districts. Most drought-prone paddy land is in the Dry Zone.")
        return
    s = snapshot(index, scale, year, month)
    if s is None or s.empty:
        st.warning(f"No {index} values for this month.")
        return
    map_with_ranking(index, s, scale, colour, ui.month_label(year, month), key)
    if note_text:
        ui.note(note_text)


# ============================================================================= overview

def _headline_cards(snaps: dict[str, pd.DataFrame]) -> pd.Series:
    """KPI cards per index plus 'flagged by 2+ indices'. Returns the per-district vote count."""
    cards = []
    for i, s in snaps.items():
        worst = s.loc[s["value"].idxmin()]
        n = int(s["dry"].sum())
        rule = "by Table 4" if i == "MSDI" else "≤ -1"
        cards.append({"label": D.INDEX_INFO[i]["name"], "accent": INDEX_COLOURS[i],
                      "value": f"{n} <span style='font-size:16px;color:{ui.MUTED}'>of {len(s)} districts</span>",
                      "sub": f"in drought ({rule}). Lowest: <b>{worst['District']}</b> ({worst['value']:+.2f})"})
    flags = pd.concat([s[["District", "dry"]] for s in snaps.values()])
    votes = flags.groupby("District")["dry"].sum()
    multi = votes[votes >= 2].sort_values(ascending=False)
    cards.append({"label": "Flagged by 2+ indices", "accent": ui.DRY_LINE,
                  "value": f"{len(multi)} <span style='font-size:16px;color:{ui.MUTED}'>districts</span>",
                  "sub": ", ".join(multi.index[:4]) + ("..." if len(multi) > 4 else "") if len(multi)
                  else "no district has a drought signal from two indices"})
    ui.kpi_cards(cards)
    st.write("")
    return votes


def _small_maps(snaps: dict, scale: int, colour: str):
    """The three indices side by side for the same month."""
    labels = st.toggle("District names on maps", value=False, key="ov_labels")
    cols = st.columns(len(snaps))
    for col, (i, s) in zip(cols, snaps.items()):
        with col:
            fig = ui.district_map(s, colour_by=colour, height=500, labels=labels, zoom=5.95, class_order=class_order(i),
                                  title=D.INDEX_INFO[i]["name"], value_label=f"{i}-{scale}")
            fig.update_layout(showlegend=False)
            ui.show_map(fig, key=f"ov_map_{i}", caption=D.INDEX_INFO[i]["what"])


def _agreement(snaps: dict, votes: pd.Series):
    """Every district's values side by side, with the number of indices that show drought."""
    ui.note("Each index looks at drought from a different angle: <b>SPI</b> the rain that fell, "
            "<b>SSI</b> the water left in the root zone, <b>MSDI</b> rainfall together with temperature.")
    tab = pd.DataFrame({i: s.set_index("District")["value"] for i, s in snaps.items()}).round(2)
    tab["indices in drought"] = votes
    st.dataframe(tab.sort_values(list(snaps)[0]).style.background_gradient(
        cmap="BrBG", vmin=-2.5, vmax=2.5, subset=list(snaps))
        .format("{:+.2f}", subset=list(snaps)).format("{:.0f}", subset=["indices in drought"]),
        use_container_width=True, height=470)


def _national_history(snaps: dict, scale: int, years: tuple[int, int]):
    """Drought extent over time and the district x year drought calendar."""
    # The paper gives the one-month MSDI no classes, so it has no dry months to count
    full = {i: table(i, scale) for i in snaps if i != "MSDI" or TH.cutoff(scale)}
    smooth = st.toggle("Smooth (6-month average)", value=True, key="ov_smooth")
    ui.show_chart(ui.drought_extent_chart(full, INDEX_COLOURS, years, smooth=6 if smooth else 1),
                  "Peaks are national droughts. Shaded bands are the documented North Central droughts; most of them "
                  "line up with a peak.")
    tabs = st.tabs([D.INDEX_INFO[i]["name"] for i in full])
    for tb, i in zip(tabs, full):
        with tb:
            rule = "at or below the Table 4 dry cut-off" if i == "MSDI" else "≤ -1"
            ui.show_chart(ui.drought_calendar(full[i], f"Drought calendar: months per year with "
                                                       f"{i}-{scale} {rule}", years),
                          "Dark cells are dry years in that district. Vertical dark stripes are "
                          "island-wide droughts.", key=f"cal_{i}")


def _animation(scale: int, year: int):
    """Animated map of one year month by month for the sidebar index."""
    ai = ui.selection()["index"]
    ay = st.selectbox("Year to play", list(range(common.START_YEAR, latest_date().year + 1)),
                      index=year - common.START_YEAR, key="ov_anim_y")
    t = table(ai, scale)
    t = t[t["Year"] == ay].dropna(subset=["value"]).copy()
    t["Month name"] = t["Month"].map(lambda m: calendar.month_abbr[m])
    gj, _ = ui.district_geojson()
    fig = px.choropleth_map(t, geojson=gj, locations="District", color="value",
                            animation_frame="Month name", hover_name="District",
                            color_continuous_scale=ui.DIVERGING, range_color=ui.VALUE_RANGE,
                            map_style="carto-positron", center=ui.SRI_LANKA_CENTER, zoom=6.1,
                            opacity=0.88, labels={"value": f"{ai}-{scale}"})
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=0, b=0))
    ui.show_map(fig, key="ov_anim", caption=f"{D.INDEX_INFO[ai]['name']} through {ay} (index chosen in "
                                            "the sidebar). Press play to watch a drought spread or ease.")


def page_overview():
    """Overview: headline status first, then the map, then national details."""
    sel = ui.selection()
    scale = sel["scale"]
    ui.page_header("Sri Lanka drought overview",
                   "Rainfall, soil moisture and combined rainfall + heat drought for all 25 districts, "
                   "1984-2025, from NASA POWER data.")
    year, month, colour = month_controls("ov")
    when = ui.month_label(year, month)
    snaps = {i: snapshot(i, scale, year, month) for i in MAIN_INDICES}
    snaps = {i: s for i, s in snaps.items() if s is not None and len(s)}
    if not snaps:
        st.warning("No index values for this month.")
        return

    if scale == 1:
        st.caption("The paper gives the one-month MSDI no classes, so at this window it shows values only.")

    # 1. Headline numbers, then the map of every district and the panel of the one picked on it
    votes = _headline_cards(snaps)
    st.subheader(f"Drought status by district, {when}, {scale}-month window")
    _district_status(snaps, votes, scale, year, month)

    # 2. The map of the index chosen in the sidebar
    idx = sel["index"] if sel["index"] in snaps else list(snaps)[0]
    st.subheader(f"Map: {D.INDEX_INFO[idx]['name']} (change the index in the sidebar)")
    map_with_ranking(idx, snaps[idx], scale, colour, when, "ov_main", height=640)

    # 3. Details
    st.divider()
    st.subheader("Details")
    with st.expander("The three indices side by side", expanded=True):
        _small_maps(snaps, scale, colour)
    with st.expander("All districts in one table"):
        _agreement(snaps, votes)
    with st.expander("How much of Sri Lanka was in drought, and when?", expanded=True):
        _national_history(snaps, scale, sel["years"])
    with st.expander("Play a year month by month (animated map)"):
        _animation(scale, year)


# ============================================================================= district status (Overview)

# Plain words for the SPI and SSI classes
CLASS_WORDS = {"Extremely dry": "far below normal", "Severely dry": "well below normal", "Moderately dry": "below normal",
               "Near normal": "close to normal", "Moderately wet": "above normal", "Very wet": "well above normal",
               "Extremely wet": "far above normal"}
VOTE_NAMES = {0: "No index in drought", 1: "One index", 2: "Two indices", 3: "All three indices"}
VOTE_COLOURS = {"No index in drought": "#EDE8DF", "One index": "#F4A259", "Two indices": "#C4561D",
                "All three indices": "#7A2E0B", "No data": "#D0D0D0"}
INDEX_NAMES = {"SPI": "Rainfall", "SSI": "Soil moisture", "MSDI": "Rainfall + temperature"}


def _status_map(votes: pd.Series, picked: str):
    """Districts coloured by how many indices show drought, with the picked district outlined."""
    v = votes.reindex(sorted(votes.index)).rename("n").reset_index()
    v["status"] = v["n"].astype(int).map(VOTE_NAMES)
    fig = ui.district_map(v, class_col="status", class_order=list(VOTE_NAMES.values()), class_colours=VOTE_COLOURS,
                          height=600, labels=True, zoom=6.25, discrete_title="Indices in drought")
    gj, _ = ui.district_geojson()
    fig.add_trace(go.Choroplethmap(geojson=gj, locations=[picked], z=[1], colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                                   showscale=False, marker_line_width=3.5, marker_line_color=ui.INK, hoverinfo="skip"))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=0.01, x=0.01))
    # A click selects a district; keep the others at full colour, the outline marks the picked one
    fig.update_traces(unselected=dict(marker=dict(opacity=1)), selector=dict(type="choroplethmap"))
    return fig


def _picked_district(sidebar_district: str) -> str:
    """The district shown in the panel: the last one clicked on the map, or the sidebar district.

    A new click wins. A change of the sidebar district also wins, so the two never fight.
    """
    ss = st.session_state
    if ss.get("ov_sidebar_seen") != sidebar_district:
        ss["ov_sidebar_seen"] = sidebar_district
        ss["ov_pick"] = sidebar_district
    state = ss.get("ov_status")
    points = list(state["selection"]["points"]) if state and state.get("selection") else []
    names = list(ui.district_geojson()[1]["District"])
    # The district shapes and their name labels come first and second, in the same order
    clicked = {p.get("location") or names[int(p["point_index"])] for p in points
               if p.get("curve_number", 0) in (0, 1) and p.get("point_index") is not None}
    seen = set(ss.get("ov_click_seen", ()))
    new = sorted(clicked - seen)          # the selection can still hold the earlier click
    if new:
        ss["ov_pick"] = new[0]
    ss["ov_click_seen"] = tuple(sorted(clicked))
    return ss.get("ov_pick", sidebar_district)


def _plain_sentence(index: str, cls, scale: int) -> str:
    """One sentence that says what this index shows for the month."""
    span = "this month" if scale == 1 else f"the last {scale} months"
    if index == "SPI":
        return f"Rainfall over {span} was {CLASS_WORDS.get(cls, 'not available')}."
    if index == "SSI":
        return f"Root-zone soil moisture over {span} was {CLASS_WORDS.get(cls, 'not available')}."
    if not isinstance(cls, str):
        return "The paper gives the one-month MSDI no classes, so only its value is shown."
    return {"Extremely dry": "The combined rainfall and temperature index is in the Extremely dry class.",
            "Dry": "The combined rainfall and temperature index is in the Dry class.",
            "Normal": "The combined rainfall and temperature index shows no drought."}[cls]


def _scale_segments(index: str, scale: int) -> tuple[float, float, list]:
    """(low end, high end, [(from, to, colour)]) of the coloured scale bar of one index."""
    if index != "MSDI":
        cuts = [(-3.0, -2.0, "Extremely dry"), (-2.0, -1.5, "Severely dry"), (-1.5, -1.0, "Moderately dry"),
                (-1.0, 1.0, "Near normal"), (1.0, 1.5, "Moderately wet"), (1.5, 2.0, "Very wet"), (2.0, 3.0, "Extremely wet")]
        return -3.0, 3.0, [(a, b, ui.CLASS_COLOURS[c]) for a, b, c in cuts]
    cut = TH.cutoff(scale)
    if cut is None:
        return -4.0, 1.0, [(-4.0, 1.0, "#D9D2C5")]
    extreme, dry = cut
    low = extreme - 0.8
    return low, 1.0, [(low, extreme, ui.CLASS_COLOURS["Extremely dry"]), (extreme, dry, ui.CLASS_COLOURS["Dry"]),
                      (dry, 1.0, ui.CLASS_COLOURS["Normal"])]


def _scale_bar(index: str, scale: int, value: float) -> str:
    """HTML of a coloured bar with a marker at this value, drier to the left."""
    low, high, segments = _scale_segments(index, scale)
    pos = lambda v: 100 * (min(max(v, low), high) - low) / (high - low)
    stops = ", ".join(f"{c} {pos(a):.1f}%, {c} {pos(b):.1f}%" for a, b, c in segments)
    # A low MSDI can come from low rain or from low temperature, so its ends are not called drier and wetter
    ends = ("lower", "higher") if index == "MSDI" else ("drier", "wetter")
    return (f"<div class='dp-bar' style='background:linear-gradient(90deg, {stops})'>"
            f"<span class='dp-mark' style='left:{pos(value):.1f}%'></span></div>"
            f"<div class='dp-ends'><span>{ends[0]}</span><span>{ends[1]}</span></div>")


def _recent_cells(index: str, scale: int, district: str, year: int, month: int) -> str:
    """HTML of the last 12 months as small squares coloured by class, the chosen month last."""
    t = table(index, scale)
    t = t[(t["District"] == district) & (t["Date"] <= pd.Timestamp(year, month, 1))].dropna(subset=["value"]).tail(12)
    cells = []
    for _, r in t.iterrows():
        cls = r.get("class")
        colour = ui.CLASS_COLOURS.get(cls, "#D9D2C5") if isinstance(cls, str) else "#D9D2C5"
        label = f"{r['Date']:%b %Y}: {r['value']:+.2f}" + (f", {cls}" if isinstance(cls, str) else "")
        cells.append(f"<span title='{label}' style='background:{colour}'></span>")
    first = t["Date"].iloc[0] if len(t) else None
    span = f"{first:%b %Y} to {ui.month_label(year, month)}" if first is not None else ""
    return f"<div class='dp-cells'>{''.join(cells)}</div><div class='dp-ends'><span>{span}</span><span>last 12 months</span></div>"


def _district_panel(snaps: dict, votes: pd.Series, district: str, scale: int, year: int, month: int):
    """The panel for one district: how many indices show drought, then each index in plain words."""
    n = int(votes.get(district, 0))
    status = VOTE_NAMES[n]
    count = {0: "No index shows drought", 1: "One index shows drought", 2: "Two indices show drought",
             3: "All three indices show drought"}[n]
    rows = []
    for i, s in snaps.items():
        r = s[s["District"] == district]
        if r.empty:
            continue
        value, cls = float(r["value"].iloc[0]), r["class"].iloc[0]
        pill = cls if isinstance(cls, str) else "No class"
        colour = ui.CLASS_COLOURS.get(pill, "#D9D2C5")
        rows.append(
            f"<div class='dp-row' style='--accent:{INDEX_COLOURS[i]}'>"
            f"<div class='dp-top'><span class='dp-lab'>{INDEX_NAMES[i]} <small>{i}-{scale}</small></span>"
            f"<span class='dp-pill' style='background:{colour};color:{ui.text_colour_on(colour)}'>{pill}</span>"
            f"<span class='dp-val'>{value:+.2f}</span></div>"
            f"<div class='dp-text'>{_plain_sentence(i, cls, scale)}</div>"
            f"{_scale_bar(i, scale, value)}{_recent_cells(i, scale, district, year, month)}</div>")
    head = (f"<div class='dp-head'><div><div class='dp-name'>{district}</div>"
            f"<div class='dp-sub'>{TH.PROVINCE_OF.get(district, '')} Province, {ui.month_label(year, month)}, "
            f"{scale}-month window</div></div>"
            f"<span class='dp-status' style='background:{VOTE_COLOURS[status]};color:{ui.text_colour_on(VOTE_COLOURS[status])}'>"
            f"{count}</span></div>")
    st.markdown(f"<div class='dp'>{head}{''.join(rows)}</div>", unsafe_allow_html=True)


def _district_status(snaps: dict, votes: pd.Series, scale: int, year: int, month: int):
    """Map of how many indices show drought, and the panel of the district picked on it."""
    picked = _picked_district(ui.selection()["district"])
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.plotly_chart(_status_map(votes, picked), use_container_width=True, key="ov_status", on_select="rerun",
                        selection_mode="points", config={"scrollZoom": True, "displaylogo": False})
        st.caption("Click a district to see its values. The sidebar district is shown first.")
    with c2:
        _district_panel(snaps, votes, picked, scale, year, month)


# ============================================================================= SSI events

@st.cache_data(show_spinner=False)
def _ssi_events(district: str, scale: int, version: float) -> pd.DataFrame:
    """SSI drought events for one district and scale, cached per input version."""
    return D.index_events(district_series("SSI", scale, district)[["Year", "Month", "value"]])


def ssi_events_tab(district: str, scale: int, key: str):
    """SSI drought events for one district and scale: runs below 0, with no copula or return periods."""
    ev = _ssi_events(district, scale, D.sources_mtime())
    st.markdown(f"**{district}, SSI-{scale}.** Each continuous run of months with the index below 0 is one drought "
                "event: its **duration** is the number of months and its **severity** the sum of the distance below 0.")
    events_panel.render_events(ev, f"SSI-{scale}", key)


# ============================================================================= SSI page

def _spi_by_scale(district: str, compute_spi_series) -> dict:
    """{1: SPI-1, 3: SPI-3, 6: SPI-6} as date-indexed Series, for the SSI lag analysis."""
    out = {}
    for k in common.SCALES:
        spi_df = compute_spi_series(district, k)
        out[k] = spi_df.set_index(pd.to_datetime(spi_df[["Year", "Month"]].assign(day=1)))[f"SPI{k}"].dropna()
    return out


def page_ssi(render_soil_moisture_tab, render_validation_tab, compute_spi_series):
    """Soil-moisture page: national map, district analysis, events, SPI vs SSI check."""
    sel = ui.selection()
    district, scale = sel["district"], sel["scale"]
    ui.page_header("Soil moisture drought (SSI)",
                   "Standardized Soil Moisture Index from NASA POWER root-zone soil wetness (GWETROOT). "
                   f"District and time window come from the sidebar ({district}, {scale} months).")
    t_map, t_district, t_events, t_val = st.tabs(
        ["National map", "District analysis", "Drought events", "SPI vs SSI validation"])
    with t_map:
        national_map_tab("SSI", "ssi_nat", "SSI reacts more slowly than SPI: soil keeps the memory of "
                                           "past rain, so soil droughts are fewer but last longer.")
    with t_district:
        render_soil_moisture_tab(district, _spi_by_scale(district, compute_spi_series), scale, sel["years"])
    with t_events:
        ssi_events_tab(district, scale, f"ssi_ev_{district}_{scale}")
    with t_val:
        render_validation_tab(district, _spi_by_scale(district, compute_spi_series))


# ============================================================================= compare indices

def wide_table(district: str, scale: int) -> pd.DataFrame:
    """One column per index for one district, indexed by (Year, Month)."""
    cols = {}
    for i in MAIN_INDICES:
        t = table(i, scale)
        if t is not None and district in set(t["District"]):
            cols[i] = t[t["District"] == district].set_index(["Year", "Month"])["value"]
    return pd.concat(cols, axis=1).sort_index()


@st.cache_data(show_spinner="Comparing indices for all districts...")
def _comparison(scale: int, version: float) -> pd.DataFrame:
    """compare_indices statistics for every district at one scale (same functions as the CSV)."""
    rows = []
    for d in D.list_districts():
        df = wide_table(d, scale)
        names = list(df.columns)
        cut = CI.dry_cutoff(scale)
        row = {"district": d, "scale": scale}
        for a, b in [(x, y) for n, x in enumerate(names) for y in names[n + 1:]]:
            row.update(CI.pair_stats(df, a, b, cut))
        for n in names:
            row.update(CI.index_stats(df, n, cut))
        rows.append(row)
    return pd.DataFrame(rows)


def comparison(scale: int) -> pd.DataFrame:
    """The comparison table for the current inputs."""
    return _comparison(scale, D.sources_mtime())


def _compare_maps(comp: pd.DataFrame):
    """National maps of SPI-SSI correlation, SPI-SSI best lag and SPI-MSDI correlation."""
    gj, _ = ui.district_geojson()
    m1, m2, m3 = st.columns(3)
    for col, metric, title, rng, scale_c, cap in [
            (m1, "pearson SPI-SSI", "Correlation, SPI vs SSI", (0.4, 1.0), "Blues",
             "Darker = soil moisture follows rainfall more closely."),
            (m2, "best lag SPI-SSI (months)", "Months for soil to respond to rain", (0, 3), "YlOrBr",
             "The shift (0-3 months) at which SSI matches SPI best."),
            (m3, "pearson SPI-MSDI", "Correlation, SPI vs MSDI", (0.4, 1.0), "Purples",
             "Lighter = heat adds more information beyond rainfall.")]:
        if metric not in comp:
            continue
        with col:
            fig = px.choropleth_map(comp, geojson=gj, locations="district", color=metric,
                                    color_continuous_scale=scale_c, range_color=rng,
                                    map_style="carto-positron", center=ui.SRI_LANKA_CENTER, zoom=5.9,
                                    opacity=0.9, hover_name="district")
            fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=14, color=ui.INK)),
                              height=460, margin=dict(l=0, r=0, t=36, b=0),
                              coloraxis_colorbar=dict(thickness=10, len=0.6, title=None))
            ui.show_map(fig, key=f"cmp_{metric}", caption=cap)


def _agreement_bars(row: pd.Series, pairs):
    """Stacked bars per pair: months dry in both, only the first, only the second."""
    fig = go.Figure()
    for part, colr, name in [("both", "#7A2E0B", "both dry"), ("only a", "#F4A259", "only the first"),
                             ("only b", "#9ECAE1", "only the second")]:
        ys = [row[f"dry both {a}-{b}"] if part == "both" else
              row[f"dry only {a} in {a}-{b}"] if part == "only a" else row[f"dry only {b} in {a}-{b}"] for a, b in pairs]
        fig.add_trace(go.Bar(x=[f"{a} vs {b}" for a, b in pairs], y=ys, marker_color=colr, name=name,
                             text=ys, textposition="inside", marker_line=dict(color="white", width=2),
                             hovertemplate="%{x}, " + name + ": %{y} months<extra></extra>"))
    fig.update_layout(barmode="stack", title="Dry months: agreement between indices")
    return ui.style_fig(fig, 380, x_title="pair of indices", y_title="months")


def _whole(v) -> int | None:
    """A count for a table cell, or None when it is not defined."""
    return None if pd.isna(v) else int(v)


def _rounded(v, digits: int) -> float | None:
    """A rounded value for a table cell, or None when it is not defined."""
    return None if pd.isna(v) else round(v, digits)


def _district_comparison(district: str, scale: int, years, row: pd.Series):
    """Three-index panel, line overlay, agreement bars and statistics for one district."""
    df = wide_table(district, scale)
    series = {i: table(i, scale).query("District == @district").reset_index(drop=True) for i in df.columns}
    ui.show_chart(ui.three_index_panel(series, years, f"-{scale}"),
                  "Read down a column: a drought that shows in all three rows is a rain, soil and heat "
                  "drought at once. SSI usually starts later and lasts longer than SPI.")
    ui.show_chart(ui.index_timeseries(series, INDEX_COLOURS, 380, f"{district}: the three indices on one axis",
                                      years),
                  "Shaded bands are the documented North Central droughts. The dotted line is the SPI and SSI dry "
                  "line (-1). MSDI is not standardised and sits around -1, so read it against its own cut-off in the "
                  "panel above.")
    pairs = [(a, b) for a, b in CI.PAIRS if pd.notna(row.get(f"dry both {a}-{b}"))]
    c1, c2 = st.columns(2)
    with c1:
        ui.show_chart(_agreement_bars(row, pairs),
                      "A tall brown part means the two indices agree on the dry months.")
    with c2:
        stats_rows = [{"Index": i, "Dry months": _whole(row[f"dry months {i}"]),
                       "% dry": _rounded(row[f"% dry months {i}"], 1), "Events": _whole(row[f"events {i}"]),
                       "Mean length (mo)": _rounded(row[f"mean duration {i}"], 1),
                       "Mean severity": _rounded(row[f"mean severity {i}"], 2),
                       "Longest (mo)": _whole(row[f"longest duration {i}"])} for i in df.columns]
        st.markdown("**Drought statistics, 1984-2025**")
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)
        r = {f"{a}-{b}": row.get(f"pearson {a}-{b}") for a, b in pairs}
        lag = row.get("best lag SPI-SSI (months)")
        ui.note("Correlation: " + ", ".join(f"<b>{k}</b> {v:.2f}" for k, v in r.items() if pd.notna(v))
                + (f". Soil moisture follows rainfall best after <b>{int(lag)} month(s)</b>." if pd.notna(lag) else ""))


def page_compare():
    """SPI vs SSI vs MSDI: national maps and the sidebar district in detail."""
    sel = ui.selection()
    district, scale = sel["district"], sel["scale"]
    ui.page_header("Compare drought indices",
                   f"How SPI (rain), SSI (soil) and MSDI (rain + heat) agree and differ. "
                   f"District and time window from the sidebar: {district}, {scale} months.")
    comp = comparison(scale)
    ui.note("MSDI dry months and events use the paper's Table 4 cut-offs, the same for every district."
            + (" The paper sets no cut-off for the 1-month MSDI, so its dry-month and event figures are empty "
               "at this window." if scale == 1 else ""), method=True)

    st.subheader("Across Sri Lanka")
    _compare_maps(comp)

    st.subheader(f"{district}: the three indices together")
    _district_comparison(district, scale, sel["years"], comp[comp["district"] == district].iloc[0])

    with st.expander("Full comparison table (all districts)"):
        st.dataframe(comp.round(3), use_container_width=True, hide_index=True)
        st.download_button("Download as CSV", comp.to_csv(index=False).encode(),
                           f"index_comparison_{scale}month.csv", "text/csv")
