"""
theme.py
========
Colour and style tokens shared by the dashboard (ui.py) and the notebooks
(notebooks/nbtools.py), so a chart looks the same in both places.

Where it sits in the pipeline: no calculations; only colours. It has no
Streamlit import, so it can be used from plain Python and Jupyter.
"""
from __future__ import annotations

from spi_copula_analysis import SPI_CATEGORIES

BG = "#F7F4EE"            # page background
CARD = "#FFFFFF"          # cards and chart surfaces
BORDER = "#E6DFD3"
INK = "#2B2118"           # main text
MUTED = "#6E6255"         # secondary text
ACCENT = "#8C4A1F"        # accents, active tab
GRID = "#EEE9E0"          # light gridlines

# Drought classes (McKee 1993): dry = browns, near normal = light neutral, wet = blues
CLASS_ORDER = [c[2] for c in reversed(SPI_CATEGORIES)]          # Extremely dry ... Extremely wet
CLASS_COLOURS = {c[2]: c[3] for c in SPI_CATEGORIES}
CLASS_COLOURS["No data"] = "#D0D0D0"

# The published MSDI has three classes of its own (paper Table 4), in the same browns
MSDI_CLASS_ORDER = ["Extremely dry", "Dry", "Normal"]
CLASS_COLOURS["Dry"] = CLASS_COLOURS["Moderately dry"]
CLASS_COLOURS["Normal"] = CLASS_COLOURS["Near normal"]

# Index line colours (blue, vermilion, purple): colour-blind separated, 3:1 contrast on white
INDEX_COLOURS = {"SPI": "#0072B2", "SSI": "#D55E00", "MSDI": "#7B4FA0"}

DRY_LINE = "#7A2E0B"                         # the -1 threshold line
DRY_SHADE = (196 / 255, 86 / 255, 29 / 255, 0.13)     # dry spells (index <= -1), RGBA 0-1
EVENT_SHADE = (150 / 255, 120 / 255, 60 / 255, 0.16)  # documented drought years, RGBA 0-1
SEQUENTIAL = "YlOrBr"                        # counts and return periods


def rgba_css(c: tuple) -> str:
    """RGBA tuple with 0-1 channels -> CSS 'rgba(r, g, b, a)' string for Plotly."""
    return f"rgba({round(c[0] * 255)}, {round(c[1] * 255)}, {round(c[2] * 255)}, {c[3]})"
