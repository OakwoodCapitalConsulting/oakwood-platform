"""
SMI/BTC Strategy Backtester — Oakwood Capital
=============================================
Integrated daily simulation with:
  - Initial allocation (default 85% SMI / 15% BTC)
  - Dividend harvesting → 12-month DCA into BTC
  - Threshold-based rebalancing: when BTC > upper threshold,
    sell down to target and reallocate to SMI by weight
  - Quarterly SMI rebalancing to target weights
"""

import base64
from pathlib import Path
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------------------------
# Oakwood Capital CI
# ---------------------------------------------------------------------------
OAK_GREEN     = "#293624"
OAK_GREEN_2   = "#1F2A1B"
OAK_GREEN_3   = "#3A4A33"
OAK_SAGE      = "#99A796"
OAK_SAGE_DIM  = "#A9B5A4"   # lightened for legibility on dark green (was #6B7868)
OAK_AXIS      = "#6B7868"   # dark muted tone retained for chart axes/gridlines only
OAK_CREAM     = "#F5F5F1"
OAK_CREAM_DIM = "#D4D4CE"
OAK_GOLD      = "#C9A961"
OAK_BORDER    = "#3D4A36"
OAK_BTC       = "#F7931A"
OAK_RED       = "#B85042"

CHART_GRID = "#3A4A33"

# ---------------------------------------------------------------------------
# Swiss withholding tax (Verrechnungssteuer) on dividends.
# In an AMC (Actively Managed Certificate) wrapper, the 35% Swiss withholding
# tax on dividends is NOT reclaimable. So only the net (1 - 35%) = 65% of each
# gross dividend is actually available for reinvestment. Applied consistently
# to both the strategy's dividend-funded DCA and the SMI Total Return benchmark
# so the comparison stays on the same after-tax basis.
# ---------------------------------------------------------------------------
WITHHOLDING_TAX = 0.35
DIVIDEND_NET_FACTOR = 1.0 - WITHHOLDING_TAX  # 0.65

CHART_BAR_COLORS = [
    OAK_SAGE, OAK_GOLD, OAK_CREAM, OAK_BTC,
    "#7A8975", "#B59A4D", "#D4D4CE", "#E08F2A",
    "#5C6B57", "#A08945", "#BCBCB6", "#C77F1F",
    "#4A584F", "#8C7639", "#9E9E97", "#A66B16",
    "#3A4A33", "#6E5A2D", "#82827C", "#7D4F0B",
]

# ---------------------------------------------------------------------------
# SMI constituents
# ---------------------------------------------------------------------------
SMI_CONSTITUENTS = {
    "NESN.SW": ("Nestlé", 16.5, "Consumer Staples"),
    "NOVN.SW": ("Novartis", 14.5, "Healthcare"),
    "RO.SW":   ("Roche", 13.0, "Healthcare"),
    "UBSG.SW": ("UBS Group", 7.0, "Financials"),
    "ZURN.SW": ("Zurich Insurance", 6.0, "Financials"),
    "ABBN.SW": ("ABB", 6.5, "Industrials"),
    "CFR.SW":  ("Richemont", 5.5, "Consumer Discretionary"),
    "SIKA.SW": ("Sika", 3.5, "Materials"),
    "LONN.SW": ("Lonza", 3.0, "Healthcare"),
    "HOLN.SW": ("Holcim", 3.0, "Materials"),
    "GIVN.SW": ("Givaudan", 3.0, "Materials"),
    "ALC.SW":  ("Alcon", 3.5, "Healthcare"),
    "PGHN.SW": ("Partners Group", 2.5, "Financials"),
    "SREN.SW": ("Swiss Re", 2.5, "Financials"),
    "LOGN.SW": ("Logitech", 2.0, "Technology"),
    "GEBN.SW": ("Geberit", 1.5, "Industrials"),
    "SCMN.SW": ("Swisscom", 2.0, "Telecom"),
    "SLHN.SW": ("Swiss Life", 2.0, "Financials"),
    "KNIN.SW": ("Kühne+Nagel", 1.5, "Industrials"),
    "SOON.SW": ("Sonova", 1.0, "Healthcare"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean_index(obj):
    if obj is None:
        return obj
    if hasattr(obj, "empty") and obj.empty:
        # IMPORTANT: an empty Series/DataFrame defaults to a RangeIndex. Returning
        # it unchanged makes every later `index <= timestamp` comparison raise a
        # TypeError (int index vs Timestamp) under pandas 3.x. Give it an empty
        # DatetimeIndex so downstream comparisons stay type-safe and simply
        # yield empty results.
        if not isinstance(obj.index, pd.DatetimeIndex):
            obj = obj.copy()
            obj.index = pd.DatetimeIndex([])
        return obj
    if hasattr(obj.index, "tz") and obj.index.tz is not None:
        obj.index = obj.index.tz_localize(None)
    obj.index = pd.to_datetime(obj.index).normalize()
    obj = obj[~obj.index.duplicated(keep="last")]
    obj = obj.sort_index()
    return obj


def _norm_ts(x):
    """Coerce any date/datetime/Timestamp (tz-aware or naive) to a tz-naive,
    midnight-normalized pd.Timestamp. pandas 3.x raises on datetime64-vs-date
    comparisons and won't match tz-aware keys against tz-naive ones, so every
    scalar used in a cross-series comparison or dict-key lookup goes through
    this first."""
    t = pd.Timestamp(x)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t.normalize()


def _to_series(x):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] >= 1:
            return x.iloc[:, 0]
    return x


def load_logo_base64():
    here = Path(__file__).parent.parent / "assets"
    for name in ("oakwood_logo.png", "logo.png", "OAKWOOD-CAPITAL-LOGO-DARK.png"):
        path = here / name
        if path.exists():
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
    return None


def style_plotly(fig, height=500):
    fig.update_layout(
        plot_bgcolor=OAK_GREEN_2, paper_bgcolor=OAK_GREEN,
        font=dict(family="'Inter', sans-serif", size=12, color=OAK_CREAM),
        height=height, margin=dict(l=60, r=30, t=40, b=50),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=OAK_GREEN_2, font_color=OAK_CREAM, bordercolor=OAK_SAGE,
                        font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(31,42,27,0.0)", borderwidth=0,
                    font=dict(size=11, color=OAK_CREAM_DIM)),
    )
    # Softer, more transparent gridlines; brighter axis lines for legibility
    fig.update_xaxes(showgrid=True, gridcolor="rgba(169,181,164,0.10)", gridwidth=1,
                     showline=True, linewidth=1, linecolor="rgba(169,181,164,0.35)", zeroline=False,
                     ticks="outside", tickcolor="rgba(169,181,164,0.35)",
                     tickfont=dict(color=OAK_CREAM_DIM, size=11),
                     title_font=dict(color=OAK_CREAM, size=12))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(169,181,164,0.10)", gridwidth=1,
                     showline=True, linewidth=1, linecolor="rgba(169,181,164,0.35)", zeroline=False,
                     ticks="outside", tickcolor="rgba(169,181,164,0.35)",
                     tickfont=dict(color=OAK_CREAM_DIM, size=11),
                     title_font=dict(color=OAK_CREAM, size=12))
    return fig


# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Oakwood Capital — Swiss Blue Chip / Bitcoin",
                   page_icon="🌳", layout="wide", initial_sidebar_state="expanded")

logo_b64 = load_logo_base64()

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', sans-serif !important;
}}
[data-testid="stAppViewContainer"] {{ background-color: {OAK_GREEN}; }}
[data-testid="stAppViewContainer"] > .main {{ background-color: {OAK_GREEN}; color: {OAK_CREAM}; }}
.main .block-container {{ padding-top: 1rem; padding-bottom: 3rem; max-width: 1400px; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
#MainMenu, footer {{ visibility: hidden; }}

.oak-bar {{
    background: linear-gradient(180deg, {OAK_GREEN_2} 0%, #1A2317 100%);
    border-bottom: 1px solid {OAK_BORDER};
    padding: 28px 36px; margin: -1rem -1rem 36px -1rem;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}}
.oak-bar .oak-logo img {{ height: 56px; width: auto; }}
.oak-bar .oak-tagline {{
    text-align: right; color: {OAK_SAGE};
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 16px; font-style: italic; letter-spacing: 0.02em;
}}
.oak-bar .oak-tagline .stamp {{
    display: block; font-family: 'Inter', sans-serif; font-style: normal;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.2em;
    color: {OAK_SAGE_DIM}; margin-top: 6px;
}}

.main h1, [data-testid="stMarkdownContainer"] h1, [data-testid="stHeading"] h1 {{
    color: {OAK_CREAM} !important;
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-weight: 500 !important; font-size: 44px !important; letter-spacing: -0.01em;
    margin: 8px 0 4px 0; line-height: 1.1;
}}
.main h1 a, [data-testid="stMarkdownContainer"] h1 a,
.main h1 span, [data-testid="stMarkdownContainer"] h1 span {{
    color: {OAK_CREAM} !important;
}}
.main h2, [data-testid="stMarkdownContainer"] h2 {{
    color: {OAK_CREAM} !important;
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-weight: 500 !important; font-size: 30px !important; letter-spacing: -0.01em;
    margin-top: 44px; margin-bottom: 16px; padding-bottom: 10px;
    border-bottom: 1px solid {OAK_BORDER};
}}
.main h3, [data-testid="stMarkdownContainer"] h3 {{
    color: {OAK_CREAM} !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; font-size: 13px !important; letter-spacing: 0.12em;
    text-transform: uppercase; margin-top: 24px; margin-bottom: 12px;
    padding-bottom: 6px; border-bottom: 1px solid {OAK_GREEN_3};
}}
.main h4, [data-testid="stMarkdownContainer"] h4 {{
    color: {OAK_CREAM} !important; font-weight: 600 !important;
    font-size: 14px !important; margin-top: 16px;
}}
.main p, .main li, .main span, .main label, .main div {{ color: {OAK_CREAM_DIM}; }}
.main strong, .main b, [data-testid="stMarkdownContainer"] strong {{ color: {OAK_CREAM} !important; }}

[data-testid="stSidebar"] {{ background-color: {OAK_GREEN_2}; border-right: 1px solid {OAK_BORDER}; }}
[data-testid="stSidebar"] * {{ color: {OAK_CREAM} !important; }}

/* Sidebar page navigation links (multipage nav) */
[data-testid="stSidebarNav"] a {{ color: {OAK_CREAM} !important; }}
[data-testid="stSidebarNav"] a span {{ color: {OAK_CREAM} !important; }}
[data-testid="stSidebarNav"] a:hover {{ background-color: {OAK_GREEN_3} !important; }}
[data-testid="stSidebarNav"] li div a span {{ color: {OAK_CREAM} !important; }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {{
    color: {OAK_CREAM} !important;
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-weight: 500 !important; font-size: 22px !important;
    padding-bottom: 8px; border-bottom: 1px solid {OAK_SAGE_DIM};
    margin-bottom: 16px; margin-top: 8px; letter-spacing: 0; text-transform: none;
}}
[data-testid="stSidebar"] h3 {{
    color: {OAK_CREAM} !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 0.18em; font-weight: 700 !important;
    margin-top: 24px; margin-bottom: 10px;
    padding-bottom: 6px; border-bottom: 1px solid {OAK_GREEN_3};
}}
[data-testid="stSidebar"] label {{
    color: {OAK_SAGE} !important; font-size: 11px !important;
    font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.12em;
}}
[data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stSelectbox label > div {{
    text-transform: none; letter-spacing: 0; font-size: 13px !important;
    color: {OAK_CREAM} !important;
}}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {{
    background-color: {OAK_GREEN} !important; color: {OAK_CREAM} !important;
    border: 1px solid {OAK_BORDER} !important; border-radius: 9px !important;
}}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div > div {{
    background-color: {OAK_SAGE} !important;
}}
[data-testid="stSidebar"] .stSlider [role="slider"] {{
    background-color: {OAK_CREAM} !important; border-color: {OAK_SAGE} !important;
}}

.stButton > button {{
    border-radius: 9px !important; font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.1em;
    font-size: 12px !important; padding: 14px 24px !important;
    transition: all 0.2s ease;
}}
.stButton > button[kind="primary"] {{
    background-color: {OAK_SAGE} !important; color: {OAK_GREEN_2} !important;
    border: 1px solid {OAK_SAGE} !important;
}}
.stButton > button[kind="primary"]:hover {{
    background-color: {OAK_CREAM} !important; border-color: {OAK_CREAM} !important;
}}
/* Secondary buttons (e.g. stress-test scenario tiles) */
.stButton > button[kind="secondary"] {{
    background-color: {OAK_GREEN_3} !important; color: {OAK_CREAM} !important;
    border: 1px solid {OAK_BORDER} !important;
    text-transform: none !important; letter-spacing: 0.02em !important;
    font-size: 11px !important; padding: 8px 10px !important;
    min-height: 56px !important; white-space: normal !important;
    line-height: 1.25 !important;
}}
.stButton > button[kind="secondary"]:hover {{
    border-color: {OAK_GOLD} !important; color: {OAK_CREAM} !important;
    background-color: {OAK_GREEN} !important;
}}
.stButton > button[kind="secondary"] p {{
    color: {OAK_CREAM} !important; font-size: 11px !important;
}}

[data-testid="stMetric"] {{
    background: {OAK_GREEN_2};
    padding: 22px 26px;
    border: 1px solid {OAK_BORDER};
    border-left: 3px solid {OAK_SAGE};
    border-radius: 10px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.02);
    transition: border-color 0.2s ease, transform 0.15s ease;
}}
[data-testid="stMetric"]:hover {{
    border-left-color: {OAK_GOLD};
}}
[data-testid="stMetricLabel"] {{
    color: {OAK_SAGE} !important; font-size: 10px !important;
    font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.14em;
}}
[data-testid="stMetricValue"] {{
    color: {OAK_CREAM} !important;
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-size: 32px !important; font-weight: 500 !important;
    letter-spacing: -0.01em; margin-top: 6px; line-height: 1.1;
}}
[data-testid="stMetricDelta"] {{
    color: {OAK_CREAM_DIM} !important; font-size: 11px !important; font-weight: 500 !important;
    margin-top: 6px;
}}
[data-testid="stMetricDelta"] svg {{ fill: {OAK_SAGE} !important; }}

[data-testid="stExpander"] {{
    background-color: {OAK_GREEN_2}; border: 1px solid {OAK_BORDER} !important;
    border-radius: 9px !important; margin-bottom: 12px;
}}
[data-testid="stExpander"] summary, [data-testid="stExpander"] details > summary {{
    background-color: transparent !important; color: {OAK_CREAM} !important;
    font-weight: 600 !important; padding: 14px 18px !important;
    font-size: 13px !important; letter-spacing: 0.05em; text-transform: uppercase;
}}
[data-testid="stExpander"] summary:hover {{ background-color: {OAK_GREEN_3} !important; }}

[data-testid="stAlert"] {{
    background-color: {OAK_GREEN_2} !important; border-radius: 9px !important;
    border-left: 3px solid {OAK_SAGE} !important; color: {OAK_CREAM} !important;
}}
[data-testid="stAlert"] * {{ color: {OAK_CREAM} !important; }}

[data-testid="stDataFrame"] {{ border: 1px solid {OAK_BORDER}; border-radius: 9px; }}
hr {{ border-color: {OAK_BORDER} !important; margin: 32px 0 !important; }}
.stSpinner > div {{ border-top-color: {OAK_SAGE} !important; }}
.modebar {{ background-color: transparent !important; }}
.modebar-btn path {{ fill: {OAK_SAGE_DIM} !important; }}
.modebar-btn:hover path {{ fill: {OAK_CREAM} !important; }}

.oak-footer {{
    margin-top: 56px; padding: 24px 0 8px 0;
    border-top: 1px solid {OAK_BORDER}; color: {OAK_SAGE_DIM};
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em; text-align: center;
}}
.oak-footer .oak-mark {{
    font-family: 'Cormorant Garamond', Georgia, serif; text-transform: none;
    letter-spacing: 0; font-style: italic; font-size: 13px;
    color: {OAK_SAGE}; margin-top: 8px; display: block;
}}

/* Risk metrics table */
.oak-metrics-table {{
    width: 100%; border-collapse: collapse;
    background: {OAK_GREEN_2}; border: 1px solid {OAK_BORDER};
    font-family: 'Inter', sans-serif; font-size: 13px;
    margin-bottom: 16px;
}}
.oak-metrics-table thead th {{
    background: {OAK_GREEN_3}; color: {OAK_CREAM};
    font-weight: 600; font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.12em; padding: 12px 16px; text-align: right;
    border-bottom: 1px solid {OAK_BORDER};
}}
.oak-metrics-table thead th:first-child {{ text-align: left; }}
.oak-metrics-table tbody td {{
    padding: 10px 16px; color: {OAK_CREAM_DIM}; text-align: right;
    border-bottom: 1px solid {OAK_GREEN_3}; font-variant-numeric: tabular-nums;
}}
.oak-metrics-table tbody td.metric-label {{
    text-align: left; color: {OAK_CREAM}; font-weight: 500;
}}
.oak-metrics-table tbody td.metric-label .hint {{
    display: block; color: {OAK_SAGE_DIM}; font-size: 10px; font-weight: 400;
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px;
}}
.oak-metrics-table tr.oak-section td {{
    background: {OAK_GREEN}; color: {OAK_SAGE};
    font-weight: 600; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.15em; padding: 14px 16px 6px 16px;
    border-bottom: 1px solid {OAK_BORDER}; text-align: left;
}}
.oak-metrics-table tr:last-child td {{ border-bottom: none; }}
.oak-metrics-table td.strategy-col {{ color: {OAK_GOLD}; font-weight: 600; }}

/* ---- Defensive legibility: ensure no dark-on-dark text slips through ---- */
/* Dataframe / table cells */
[data-testid="stDataFrame"] *, [data-testid="stTable"] * {{
    color: {OAK_CREAM_DIM} !important;
}}
[data-testid="stDataFrame"] [role="columnheader"] {{
    color: {OAK_CREAM} !important; background-color: {OAK_GREEN_3} !important;
}}
/* Slider min/max + current value labels */
[data-testid="stSlider"] [data-testid="stTickBar"],
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] div[data-baseweb] div {{
    color: {OAK_CREAM_DIM} !important;
}}
[data-testid="stSlider"] [role="slider"] + div, .stSlider [data-testid="stThumbValue"] {{
    color: {OAK_CREAM} !important;
}}
/* Selectbox / dropdown popover options (rendered in a portal) */
[data-baseweb="popover"] li, [data-baseweb="menu"] li,
ul[role="listbox"] li, [data-baseweb="select"] span {{
    color: {OAK_CREAM} !important;
}}
[data-baseweb="popover"] ul, [data-baseweb="menu"] ul, ul[role="listbox"] {{
    background-color: {OAK_GREEN_2} !important;
}}
[data-baseweb="popover"] li:hover, ul[role="listbox"] li:hover {{
    background-color: {OAK_GREEN_3} !important;
}}
/* Number input text + radio/checkbox labels */
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {{
    color: {OAK_CREAM} !important;
}}
.stRadio label, .stCheckbox label, [data-testid="stWidgetLabel"] {{
    color: {OAK_CREAM} !important;
}}
/* Tooltips (the small "?" help bubbles) */
[data-baseweb="tooltip"], [role="tooltip"] {{
    background-color: {OAK_GREEN_2} !important; color: {OAK_CREAM} !important;
    border: 1px solid {OAK_BORDER} !important;
}}
[data-baseweb="tooltip"] * {{ color: {OAK_CREAM} !important; }}
/* Date input */
[data-testid="stDateInput"] input {{ color: {OAK_CREAM} !important; }}
/* General caption text */
[data-testid="stCaptionContainer"], .stCaption {{ color: {OAK_SAGE_DIM} !important; }}

/* Softer card shadows for depth */
[data-testid="stMetric"] {{
    box-shadow: 0 2px 8px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.03) !important;
}}

/* ---- Visibility fixes for default Streamlit chrome ---- */
/* Sidebar collapse arrow + scrollbar are primarily handled by the dark base
   theme + gold primaryColor in .streamlit/config.toml. The rules below are a
   defensive fallback for the scrollbar in case the theme doesn't fully cover it. */
::-webkit-scrollbar {{ width: 11px; height: 11px; }}
::-webkit-scrollbar-track {{ background: {OAK_GREEN_2}; }}
::-webkit-scrollbar-thumb {{
    background: {OAK_SAGE_DIM}; border-radius: 8px;
    border: 2px solid {OAK_GREEN_2};
}}
::-webkit-scrollbar-thumb:hover {{ background: {OAK_GOLD}; }}
/* Firefox */
html, body, [data-testid="stSidebar"], section[data-testid="stSidebar"] > div {{
    scrollbar-color: {OAK_SAGE_DIM} {OAK_GREEN_2}; scrollbar-width: thin;
}}

/* 3. Number input +/- stepper buttons (initial capital etc.) */
[data-testid="stNumberInput"] button {{
    background-color: {OAK_GREEN_3} !important;
    border: 1px solid {OAK_BORDER} !important;
    color: {OAK_CREAM} !important;
}}
[data-testid="stNumberInput"] button svg,
[data-testid="stNumberInput"] button path,
[data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"] svg,
[data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"] svg {{
    fill: {OAK_CREAM} !important; color: {OAK_CREAM} !important;
}}
[data-testid="stNumberInput"] button:hover {{
    background-color: {OAK_SAGE} !important;
}}
[data-testid="stNumberInput"] button:hover svg,
[data-testid="stNumberInput"] button:hover path {{
    fill: {OAK_GREEN_2} !important;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if logo_b64:
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="Oakwood Capital"/>'
else:
    logo_html = '<span style="color:#F5F5F1; font-family:Cormorant Garamond, serif; font-size:28px;">Oakwood Capital</span>'

st.markdown(f"""
<div class="oak-bar">
    <div class="oak-logo">{logo_html}</div>
    <div class="oak-tagline">
        Quantitative Strategy Research
        <span class="stamp">Internal Tool · Confidential</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f"<h1 style='color:{OAK_CREAM}; font-family:\"Cormorant Garamond\", Georgia, serif; "
    f"font-weight:500; font-size:44px; letter-spacing:-0.01em; margin:8px 0 4px 0; "
    f"line-height:1.1;'>OAK Swiss Blue Chip / Bitcoin</h1>",
    unsafe_allow_html=True
)
st.markdown(
    f"<p style='color:{OAK_CREAM_DIM}; font-size:15px; margin-top:0; max-width: 820px;'>"
    "Disciplined SMI replication with structural BTC allocation, dividend-funded DCA "
    "and threshold-based risk management. Backtest on historical market data."
    "</p>",
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Parameter")

    st.markdown("### Stress-Test-Szenarien")
    st.markdown(
        f"<p style='color:{OAK_SAGE_DIM}; font-size:11px; margin-top:-6px;'>"
        "One-click historical crisis windows. Sets the backtest period below.</p>",
        unsafe_allow_html=True)
    _scenarios = {
        "COVID Crash (2020)": (date(2020, 1, 1), date(2020, 12, 31)),
        "BTC Bear Market (2022)": (date(2022, 1, 1), date(2022, 12, 31)),
        "Banking Crisis / CS (2023)": (date(2023, 1, 1), date(2023, 12, 31)),
        "Full History (2018–today)": (date(2018, 1, 1), date.today()),
    }
    _sc_cols = st.columns(2)
    for _i, (_label, (_s, _e)) in enumerate(_scenarios.items()):
        if _sc_cols[_i % 2].button(_label, use_container_width=True, key=f"sc_{_i}"):
            st.session_state["scenario_start"] = _s
            st.session_state["scenario_end"] = _e
            st.session_state["smi_has_run"] = True  # auto-show results

    st.markdown("### Backtest-Zeitraum")
    _default_start = st.session_state.get("scenario_start", date(2018, 1, 1))
    _default_end = st.session_state.get("scenario_end", date.today())
    start_date = st.date_input("Startdatum", value=_default_start,
                               min_value=date(2010, 1, 1),
                               max_value=date.today() - relativedelta(months=6))
    end_date = st.date_input("Enddatum", value=_default_end,
                             min_value=start_date + relativedelta(months=6),
                             max_value=date.today())
    initial_capital = st.number_input("Anfangskapital (CHF)", min_value=10_000,
                                      max_value=10_000_000_000, value=1_000_000, step=10_000)

    st.markdown("### Allokation")
    initial_btc_pct = st.slider("Initial BTC Allokation (%)",
                                min_value=0, max_value=50, value=15, step=1) / 100.0
    upper_threshold = st.slider("Upper Threshold — Sell-Down Trigger (%)",
                                min_value=15, max_value=75, value=25, step=1) / 100.0
    target_btc_pct = st.slider("Target nach Sell-Down (%)",
                               min_value=0, max_value=50, value=15, step=1) / 100.0

    if target_btc_pct >= upper_threshold:
        st.error("Target muss kleiner als Upper Threshold sein.")
        st.stop()

    threshold_check_freq = st.selectbox(
        "Schwellenprüfung-Frequenz (Bitcoin-Band)",
        ["Monatlich (Standard)", "Quartalsweise", "Halbjährlich"], index=0,
        help="Wie oft wird geprüft, ob Bitcoin die obere Schwelle überschritten "
             "hat? Unabhängig von der DCA-Käufe (die laufen immer monatlich) "
             "und unabhängig vom Aktien-Rebalancing unten. Seltener prüfen "
             "erlaubt mehr Drift über der Schwelle zwischen den Terminen, "
             "dafür weniger Transaktionen. Standard = jeden Monatsultimo, "
             "das historisch verifizierte Design.")

    st.markdown("### Aktien-Sleeve")
    weighting_method = st.radio("SMI Gewichtung",
        ["Marktkapitalisierung (Approx. + 18% Cap)", "Equal Weight (5 % je Titel)"])
    rebalance_freq = st.selectbox("SMI Rebalancing-Frequenz",
        ["Quartalsweise", "Halbjährlich", "Jährlich", "Keine"], index=0)

    st.markdown("### Dividenden-Wiederanlage")
    dca_months = st.slider("DCA-Zeitraum (Monate)", 1, 24, 12)
    btc_source = st.radio("BTC-Quelle",
        ["BTC-USD (gesamte Historie)", "IBIT ETF (ab Jan 2024)", "BTC-USD bis 2024, dann IBIT"],
        index=2)

    st.markdown("### Risikoanalyse")
    risk_free_rate = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=5.0,
                               value=1.0, step=0.25,
                               help="Annualisiert. Default ~1% entspricht historischem CHF/SARON-Durchschnitt.") / 100.0

    st.markdown("### Kosten & Gebühren")
    tx_cost_bps = st.slider("Transaction Cost (bps per trade)", min_value=0.0, max_value=50.0,
                            value=10.0, step=1.0,
                            help="Cost in basis points applied to traded notional at each "
                                 "trade (initial allocation, DCA buys, threshold sells, "
                                 "rebalancing turnover). 10 bps = 0.10%.")
    mgmt_fee_pct = st.slider("Management Fee (% p.a.)", min_value=0.0, max_value=3.0,
                             value=1.5, step=0.05,
                             help="Daily accrual, deducted from NAV (1/252 per trading day).") / 100.0
    perf_fee_pct = st.slider("Performance Fee (%)", min_value=0.0, max_value=30.0,
                             value=15.0, step=1.0,
                             help="Charged on gains above the High Water Mark.") / 100.0
    hurdle_type = st.selectbox("Hurdle Type",
                               ["Hard Hurdle", "Soft Hurdle", "No Hurdle (HWM only)"], index=0,
                               help="Hard: performance fee only on returns ABOVE the hurdle rate. "
                                    "Soft: once the hurdle is cleared, the fee applies to the ENTIRE "
                                    "gain above HWM (catch-up). No Hurdle: fee on all gains above HWM.")
    hwm_hurdle_pct = st.slider("Hurdle Rate Year 1 (%)", min_value=0.0, max_value=15.0,
                               value=5.0, step=0.5,
                               help="Annual hurdle return the strategy must beat before performance "
                                    "fees apply in Year 1. After Year 1 the HWM governs.") / 100.0
    crystallization_freq = st.selectbox("Performance Fee Crystallization",
                                         ["Quarterly", "Semi-Annual", "Annual"], index=0,
                                         help="How often the performance fee is crystallized against the HWM.")

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Backtest starten", type="primary", use_container_width=True,
                        disabled=(target_btc_pct >= upper_threshold))

    # Make the backtest "sticky": once run, keep showing results across reruns
    # (e.g. when the user clicks the PDF button) instead of clearing the page.
    if run_btn:
        st.session_state["smi_has_run"] = True
    _show_results = run_btn or st.session_state.get("smi_has_run", False)

    st.markdown(
        f"<div style='font-size:10px; color:{OAK_SAGE_DIM}; text-transform:uppercase; "
        f"letter-spacing:0.12em; padding-top:24px; margin-top:24px; "
        f"border-top:1px solid {OAK_BORDER};'>"
        "Data Source: Yahoo Finance · Adj. Close<br>"
        "FX: USDCHF Spot · Threshold checks: monthly"
        "</div>", unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
import time as _time


def _download_with_retry(tickers, start, end, attempts=3):
    """Download with retry/backoff to survive Yahoo Finance rate limiting."""
    last_exc = None
    for i in range(attempts):
        try:
            data = yf.download(tickers, start=start, end=end, progress=False,
                               auto_adjust=False, actions=False,
                               group_by="ticker", threads=False)
            if data is not None and not data.empty:
                return data
        except Exception as e:
            last_exc = e
        # Exponential backoff: 2s, 4s, 8s — gives Yahoo time to lift the limit
        if i < attempts - 1:
            _time.sleep(2 * (2 ** i))
    return None


@st.cache_data(ttl=21600, show_spinner=False)
def _get_split_series(ticker_symbol):
    """Fetch stock-split events (ratio per split date), independent of
    dividends. Mirrors _get_dividend_series' fallback chain for robustness
    across yfinance versions."""
    tk = yf.Ticker(ticker_symbol)
    try:
        splits = tk.splits
        if splits is not None and not splits.empty:
            return splits
    except Exception:
        pass
    try:
        hist = tk.history(period="max", actions=True, auto_adjust=False)
        if hist is not None and not hist.empty and "Stock Splits" in hist.columns:
            s = hist["Stock Splits"]
            s = s[s > 0]
            if not s.empty:
                return s
    except Exception:
        pass
    return pd.Series(dtype=float)


def _apply_split_adjustment(raw_close, splits):
    """Adjust a RAW (unadjusted) Close series for stock splits ONLY.

    CRITICAL: dividends are deliberately NOT adjusted for here. The strategy
    extracts real per-share dividend cash separately (fetch_dividends /
    div_lookup) to fund the Bitcoin DCA — that cash must be the ONLY place
    the dividend shows up. Yahoo's "Adj Close" bakes dividends into the price
    itself (as if reinvested into the same stock), which would credit every
    dividend TWICE: once as phantom price appreciation, once as harvested
    cash. Splits are cosmetic (no economic value change) and must still be
    adjusted, or a real split creates a fake overnight NAV collapse.

    Convention matches Yahoo's own split methodology: prices strictly BEFORE
    a split date are divided by the cumulative product of all split ratios
    that occur after them, so the series is continuous across the split.
    """
    if splits is None or splits.empty:
        return raw_close.copy()
    s = splits.copy()
    try:
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    s = s[s > 0]
    if s.empty:
        return raw_close.copy()
    # VALIDIERUNG PER KONTINUITÄT, nicht per Ratio-Grössenordnung. Eine frühere
    # Fassung verwarf Ratios ausserhalb 0.05–20 als "unplausibel" — das war
    # eine ungeprüfte Annahme und FALSCH: Sika führte am 13./14. Juni 2018
    # einen realen, gut dokumentierten 60:1-Split durch (Bareaktien-Split im
    # Zuge der Saint-Gobain/Burkard-Übernahmeschlacht, bestätigt u.a. durch
    # die Eurex-Corporate-Action-Meldung und Sikas eigene Investor-Relations-
    # Daten). Ein Grössen-Schwellenwert hätte diesen echten Split verworfen.
    # Stattdessen: für jede gemeldete Split-Ratio prüfen, ob sie tatsächlich
    # den beobachteten Kurssprung im ROHDATENSATZ erklärt (implizite Ratio =
    # Kurs davor / Kurs danach, verglichen mit der gemeldeten Ratio). Erklärt
    # sie ihn (auch bei sehr hoher Ratio wie 60), wird sie angewendet —
    # unabhängig von ihrer Grösse. Erklärt sie ihn NICHT, ist sie vermutlich
    # ein Datenfehler und wird verworfen.
    valid = {}
    for split_date, ratio in s.items():
        before = raw_close[raw_close.index < split_date]
        after = raw_close[raw_close.index >= split_date]
        if before.empty or after.empty:
            continue
        p_before, p_after = before.iloc[-1], after.iloc[0]
        if p_before <= 0 or p_after <= 0:
            continue
        implied_ratio = p_before / p_after
        # Grosszügige Toleranz (±40%) für normale Kursbewegung rund um das
        # Split-Datum — die Ratio muss die Grössenordnung des Sprungs
        # erklären, nicht exakt zu ihm passen.
        if 0.6 <= (implied_ratio / float(ratio)) <= 1.6:
            valid[split_date] = float(ratio)
    if not valid:
        return raw_close.copy()
    factor = pd.Series(1.0, index=raw_close.index)
    for split_date, ratio in valid.items():
        factor.loc[factor.index < split_date] *= ratio
    return raw_close / factor


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_prices(tickers, start, end):
    # IMPORTANT: use RAW "Close", never "Adj Close". Adj Close is adjusted for
    # BOTH dividends and splits — using it here would double-count every
    # dividend (see _apply_split_adjustment docstring). We adjust for splits
    # only, explicitly, below, and leave dividends to fetch_dividends().
    data = _download_with_retry(tickers, start, end)
    cols = {}
    if data is not None and not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
            level0 = data.columns.get_level_values(0).unique().tolist()
            for t in tickers:
                if t in level0:
                    try:
                        sub = data[t]
                        if "Close" in sub.columns:
                            cols[t] = sub["Close"]
                    except Exception:
                        pass
        else:
            if "Close" in data.columns:
                cols[tickers[0]] = data["Close"]

    # Per-ticker fallback for any ticker the batch download missed
    # (rate-limited tickers often succeed on an individual retry)
    missing = [t for t in tickers if t not in cols]
    for t in missing:
        try:
            _time.sleep(0.5)
            single = yf.download(t, start=start, end=end, progress=False,
                                 auto_adjust=False, actions=False, threads=False)
            if single is not None and not single.empty:
                if isinstance(single.columns, pd.MultiIndex):
                    single.columns = single.columns.get_level_values(0)
                if "Close" in single.columns:
                    cols[t] = single["Close"]
        except Exception:
            pass

    if not cols:
        return pd.DataFrame()

    # Split-adjust each raw Close series independently (dividends untouched).
    for t in list(cols.keys()):
        try:
            raw = _clean_index(cols[t].dropna())
            splits = _get_split_series(t)
            cols[t] = _apply_split_adjustment(raw, splits)
        except Exception:
            pass  # fall back to the raw (unsplit-adjusted) series rather than drop the ticker

    out = pd.DataFrame(cols)
    out = _clean_index(out)
    out = out.dropna(axis=1, how="all")
    return out.dropna(how="all")




@st.cache_data(ttl=21600, show_spinner=False)
def _get_dividend_series(ticker_symbol):
    """Robustly fetch a dividend Series across yfinance versions.
    Newer yfinance versions changed internals ('PriceHistory' object has no
    attribute '_dividends'), so we try several access paths and fall back
    to parsing the 'Dividends' column from full history."""
    tk = yf.Ticker(ticker_symbol)
    # Method 1: the .dividends property (works on most versions)
    try:
        divs = tk.dividends
        if divs is not None and not divs.empty:
            return divs
    except Exception:
        pass
    # Method 2: extract from .history(actions=True) Dividends column
    try:
        hist = tk.history(period="max", actions=True, auto_adjust=False)
        if hist is not None and not hist.empty and "Dividends" in hist.columns:
            divs = hist["Dividends"]
            divs = divs[divs > 0]
            if not divs.empty:
                return divs
    except Exception:
        pass
    # Method 3: get_dividends() method (older/alternative API)
    try:
        divs = tk.get_dividends()
        if divs is not None and not divs.empty:
            return divs
    except Exception:
        pass
    return pd.Series(dtype=float)


def fetch_dividends(tickers, start, end):
    rows = []
    failed = []
    for t in tickers:
        try:
            divs = _get_dividend_series(t)
            if divs is None or len(divs) == 0:
                continue
            divs = _clean_index(divs)
            # Normalize timezone-aware index to naive for comparison
            try:
                if divs.index.tz is not None:
                    divs.index = divs.index.tz_localize(None)
            except (AttributeError, TypeError):
                pass
            divs = divs[(divs.index >= pd.Timestamp(start)) & (divs.index <= pd.Timestamp(end))]
            for d, v in divs.items():
                if float(v) > 0:
                    rows.append({"date": d, "ticker": t, "dividend_per_share": float(v)})
        except Exception:
            failed.append(t)
    if failed:
        st.info(f"Dividend data unavailable for: {', '.join(failed)}. "
                f"These titles contribute price returns only (no dividend DCA into BTC).")
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "dividend_per_share"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_series(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False,
                     auto_adjust=False, threads=False)
    if df is None or df.empty:
        # Empty but DATE-indexed, so downstream `index <= d` comparisons are safe.
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    if isinstance(df.columns, pd.MultiIndex):
        col = ("Adj Close", ticker) if ("Adj Close", ticker) in df.columns else df.columns[0]
        s = df[col]
    else:
        s = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    s = _to_series(s)
    s = _clean_index(s)
    return s.dropna()


# ---------------------------------------------------------------------------
# Integrated Strategy Simulation
# ---------------------------------------------------------------------------
def get_rebalance_dates(idx, freq):
    """Rebalance-Kalender für den SMI-Aktienkern.

    KORREKTUR: SIX überprüft die SMI-ZUSAMMENSETZUNG nur einmal jährlich, am
    dritten Freitag im SEPTEMBER (nicht Dezember, wie zuvor hier gesetzt) —
    das ist der reale Termin, an dem Indexmitglieder wechseln. Die 18%-
    Gewichtskappung läuft laut SIX-Methodik separat quartalsweise (März/
    Juni/September/Dezember), ist aber eine reine Kappungs-Korrektur für
    Titel über 18%, kein volles Zurücksetzen auf Zielgewichte — unser
    "Quartalsweise" resettet dagegen ALLE Gewichte, was näher an einem
    vereinfachten Constant-Mix-Rebalancing liegt als an der echten SIX-
    Mechanik. Für Genauigkeit daher "Jährlich" (September) empfohlen; die
    quartalsweise Cap-only-Korrektur ist als offener Verfeinerungspunkt in
    Abschnitt 9 des Reglements vermerkt, nicht hier implementiert.
    """
    if freq == "Keine":
        return set()
    if freq == "Quartalsweise":
        months = {3, 6, 9, 12}
    elif freq == "Halbjährlich":
        months = {3, 9}
    else:
        months = {9}   # Jährlich = September, wie bei SIX (war: Dezember)
    out = set()
    df = pd.DataFrame(index=idx)
    df["ym"] = df.index.to_period("M")
    for (ym), sub in df.groupby("ym"):
        m = sub.index[-1].month
        if m in months:
            out.add(sub.index[-1])
    return out


def run_strategy(prices, dividends_df, btc_prices_usd, fx_chf_usd,
                 initial_capital, weights,
                 initial_btc_pct, upper_threshold, target_btc_pct,
                 rebalance_dates_set, dca_months, tx_cost_bps=0.0,
                 threshold_check_dates_set=None):
    """Integrated daily simulation.
    Returns: timeseries_df, transactions_df, threshold_events_df

    threshold_check_dates_set: dates on which the Bitcoin band (upper_threshold
    -> target_btc_pct) is evaluated. Defaults to EVERY month-end (None ->
    falls back to month_ends below), matching the original verified design
    (documented decision: "Rebalancing: Quarterly (SMI) · Monthly (BTC
    check)"). Pass a coarser date set (e.g. via get_rebalance_dates) to check
    less often — this only changes HOW OFTEN the band is evaluated, never
    whether DCA purchases happen (DCA always executes at every month-end,
    independent of this parameter).

    tx_cost_bps: round-trip transaction cost in basis points applied to the
    traded notional at each trade (initial buy, DCA buys, threshold sells, and
    quarterly equity rebalancing turnover). 10 bps = 0.10%.
    """
    tx_cost = tx_cost_bps / 10000.0  # bps -> fraction
    total_tx_costs = 0.0  # accumulator in CHF
    total_wht = 0.0       # gross dividend withheld at source (35%, non-reclaimable)
    # Normalize the index to tz-naive, midnight Timestamps so that comparisons
    # against the (cleaned) BTC/FX series and the dividend/rebalance keys stay
    # consistent under pandas 3.x.
    prices = _clean_index(prices.copy())
    rebalance_dates_set = {_norm_ts(x) for x in rebalance_dates_set}
    available = [t for t in weights if t in prices.columns]
    if not available:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    w = pd.Series({t: weights[t] for t in available})
    w = w / w.sum()
    # IMPORTANT: do NOT do a cross-column dropna() here — that would throw out
    # every trading day where any one ticker is missing (e.g. Alcon listed only
    # from April 2019, which would cut all 2018 data). Keep all dates where at
    # least one ticker has a price; per-day we work with the active universe.
    prices_clean = prices[available].dropna(how="all")
    if prices_clean.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # Forward-fill isolated missing quotes per ticker so a single NaN day does
    # not value the holding at zero (fake NAV dips / spurious threshold sells).
    # Leading NaNs are NOT filled — late listings (e.g. Alcon) stay intact.
    prices_clean = prices_clean.ffill()

    # First trading date for each ticker (the day it starts having a price).
    # Tickers added later (e.g. Alcon spin-off Apr 2019) enter the portfolio
    # on or after this date at the next rebalance.
    ticker_first_date = {t: prices_clean[t].first_valid_index() for t in available}

    btc_prices_usd = _clean_index(btc_prices_usd.copy())
    fx_chf_usd = _clean_index(fx_chf_usd.copy())

    def get_btc_price(d):
        sub = btc_prices_usd[btc_prices_usd.index <= d]
        return float(sub.iloc[-1]) if not sub.empty else None

    def get_fx(d):
        sub = fx_chf_usd[fx_chf_usd.index <= d]
        return float(sub.iloc[-1]) if not sub.empty else None

    div_lookup = {}
    if not dividends_df.empty:
        for _, r in dividends_df.iterrows():
            # Net dividend after non-reclaimable 35% Swiss withholding tax (AMC wrapper)
            div_lookup[(_norm_ts(r["date"]), r["ticker"])] = \
                r["dividend_per_share"] * DIVIDEND_NET_FACTOR

    # Month-end dates within our index
    month_ends = set()
    df_idx = pd.DataFrame(index=prices_clean.index)
    df_idx["ym"] = df_idx.index.to_period("M")
    for ym, sub in df_idx.groupby("ym"):
        month_ends.add(sub.index[-1])

    first_day = prices_clean.index[0]
    btc_price_0 = get_btc_price(first_day)
    fx_0 = get_fx(first_day)

    # Subset of tickers that already have a price on the first day
    active_t0 = [t for t in available if pd.notna(prices_clean.loc[first_day, t])]
    if not active_t0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # Renormalize weights across the day-0 active universe
    w_t0 = w[active_t0] / w[active_t0].sum()

    initial_smi_chf = initial_capital * (1 - initial_btc_pct)
    initial_btc_chf = initial_capital * initial_btc_pct

    smi_shares = {t: 0.0 for t in available}
    transactions = []
    # --- Attribution ------------------------------------------------------
    att_btc_init_invested = initial_capital * initial_btc_pct   # brutto CHF, Tag 1
    att_btc_dca_invested = 0.0        # brutto CHF via Dividenden-DCA
    att_sold_gross_init = 0.0         # Brutto-Verkaufswert aus dem Start-Lot
    att_sold_gross_dca = 0.0          # Brutto-Verkaufswert aus dem DCA-Lot
    att_div_income = 0.0              # vereinnahmte Netto-Dividenden
    att_equity_invested = initial_capital * (1 - initial_btc_pct)  # brutto in Aktien
    # Per-event log of the actual net dividend cash harvested on the *live*
    # (evolving) share counts — this is the real cash that funds the BTC DCA,
    # as opposed to a frozen initial-share approximation.
    dividend_cashflows = []

    if btc_price_0 is None or fx_0 is None or fx_0 == 0 or btc_price_0 == 0 or initial_btc_pct == 0:
        # No initial BTC possible — full to SMI
        cost = initial_capital * tx_cost
        total_tx_costs += cost
        investable = initial_capital - cost
        for t in active_t0:
            smi_shares[t] = (investable * w_t0[t]) / prices_clean.loc[first_day, t]
        btc_held = 0.0
        btc_u_init = 0.0
        btc_u_dca = 0.0
    else:
        # Cost charged on both equity and BTC legs of the initial allocation
        cost = initial_capital * tx_cost
        total_tx_costs += cost
        smi_invest = initial_smi_chf - initial_smi_chf * tx_cost
        for t in active_t0:
            smi_shares[t] = (smi_invest * w_t0[t]) / prices_clean.loc[first_day, t]
        btc_invest = initial_btc_chf - initial_btc_chf * tx_cost
        usd_0 = btc_invest / fx_0
        btc_held = usd_0 / btc_price_0
        # ATTRIBUTION: zwei getrennte Lots — Startallokation (Tag 1) vs.
        # dividendenfinanzierter DCA. Verkäufe reduzieren sie PRO RATA.
        btc_u_init = btc_held
        btc_u_dca = 0.0
        transactions.append({
            "date": first_day, "type": "BUY", "reason": "INITIAL",
            "btc_amount": btc_held, "chf_amount": initial_btc_chf,
            "usd_amount": usd_0, "btc_price_usd": btc_price_0, "usdchf": fx_0,
        })

    # DCA queue
    pending_dca = []  # each: {"remaining": int, "monthly_chf": float}
    dividend_cash = 0.0  # harvested net dividends awaiting DCA deployment —
                         # part of the NAV (was previously omitted: NAV dipped
                         # at every ex-date and the undeployed queue vanished)

    records = []
    threshold_events = []

    def _active_on(d):
        """Tickers with a valid price on day d."""
        return [t for t in available if pd.notna(prices_clean.loc[d, t])]

    def _smi_value_on(d, row):
        """Sum portfolio value across tickers that have a price on day d."""
        return sum(smi_shares[t] * row[t] for t in available if pd.notna(row[t]))

    for d in prices_clean.index:
        row = prices_clean.loc[d]
        btc_price_d = get_btc_price(d)
        fx_d = get_fx(d)
        active_today = _active_on(d)

        # 1. Dividend ex-date — collect cash, queue DCA tranches
        for t in active_today:
            key = (d, t)
            if key in div_lookup:
                cash = smi_shares[t] * div_lookup[key]
                if cash > 0:
                    dividend_cashflows.append(
                        {"date": d, "ticker": t, "cash_chf": cash})
                    # cash is already net (×0.65); back out the 35% withheld
                    total_wht += cash * (WITHHOLDING_TAX / DIVIDEND_NET_FACTOR)
                    dividend_cash += cash
                    att_div_income += cash
                    pending_dca.append({"remaining": dca_months,
                                        "monthly_chf": cash / dca_months,
                                        "ticker": t})

        # 2. Month-end: execute DCA buys
        is_month_end = d in month_ends
        # Threshold-check cadence is INDEPENDENT of the DCA cadence. Defaults
        # to every month-end (the original verified design) unless a coarser
        # date set is supplied.
        is_threshold_check_day = (
            is_month_end if threshold_check_dates_set is None
            else d in threshold_check_dates_set
        )
        if is_month_end and pending_dca:
            total_dca_chf = sum(e["monthly_chf"] for e in pending_dca
                                if e["remaining"] > 0)

            if total_dca_chf > 0 and btc_price_d and fx_d and fx_d > 0:
                # Consume one tranche per entry ONLY now that the buy executes
                # (previously tranches were consumed even when BTC/FX quotes
                # were missing — that money silently vanished).
                for entry in pending_dca:
                    if entry["remaining"] > 0:
                        entry["remaining"] -= 1
                pending_dca = [e for e in pending_dca if e["remaining"] > 0]
                cost = total_dca_chf * tx_cost
                total_tx_costs += cost
                net_dca_chf = total_dca_chf - cost
                dividend_cash -= total_dca_chf   # deployed (incl. tx cost)
                usd = net_dca_chf / fx_d
                btc_bought = usd / btc_price_d
                btc_held += btc_bought
                btc_u_dca += btc_bought              # DCA-Lot
                att_btc_dca_invested += total_dca_chf  # brutto (inkl. tx)
                transactions.append({
                    "date": d, "type": "BUY", "reason": "DCA",
                    "btc_amount": btc_bought, "chf_amount": total_dca_chf,
                    "usd_amount": usd, "btc_price_usd": btc_price_d, "usdchf": fx_d,
                })

        # 3. Threshold check (independent cadence, default = month-end)
        if is_threshold_check_day and btc_price_d and fx_d and fx_d > 0:
            smi_value = _smi_value_on(d, row)
            btc_value_chf = btc_held * btc_price_d * fx_d
            total = smi_value + btc_value_chf + dividend_cash
            if total > 0:
                btc_pct = btc_value_chf / total
                if btc_pct > upper_threshold:
                    target_btc_chf = total * target_btc_pct
                    sell_chf = btc_value_chf - target_btc_chf
                    sell_usd = sell_chf / fx_d
                    sell_btc = sell_usd / btc_price_d
                    btc_held -= sell_btc
                    # PRO RATA (nie FIFO — sonst verzerrt die Verkaufsreihenfolge
                    # die Attribution)
                    _tot_u = btc_u_init + btc_u_dca
                    _f_init = (btc_u_init / _tot_u) if _tot_u > 0 else 0.0
                    btc_u_init -= sell_btc * _f_init
                    btc_u_dca -= sell_btc * (1 - _f_init)
                    att_sold_gross_init += sell_chf * _f_init
                    att_sold_gross_dca += sell_chf * (1 - _f_init)
                    att_equity_invested += sell_chf   # Erlös geht in Aktien

                    # Transaction cost on the BTC sale and the equity re-purchase
                    # (two legs: selling BTC, buying equity with the proceeds)
                    cost = sell_chf * tx_cost * 2
                    total_tx_costs += cost
                    net_to_smi = sell_chf - cost

                    # Reallocate net proceeds to active tickers by renormalized weights
                    if active_today:
                        w_active = w[active_today] / w[active_today].sum()
                        for t in active_today:
                            extra_chf = net_to_smi * w_active[t]
                            smi_shares[t] += extra_chf / row[t]

                    transactions.append({
                        "date": d, "type": "SELL", "reason": "THRESHOLD",
                        "btc_amount": -sell_btc, "chf_amount": -sell_chf,
                        "usd_amount": -sell_usd, "btc_price_usd": btc_price_d, "usdchf": fx_d,
                    })

                    smi_value_after = _smi_value_on(d, row)
                    btc_value_after = btc_held * btc_price_d * fx_d
                    total_after = smi_value_after + btc_value_after + dividend_cash
                    threshold_events.append({
                        "date": d, "btc_pct_before": btc_pct,
                        "btc_pct_after": btc_value_after / total_after if total_after > 0 else 0,
                        "btc_sold": sell_btc, "chf_to_smi": sell_chf,
                    })

        # 4. Quarterly SMI rebalance (back to target weights)
        # This is also where new index members (e.g. Alcon from Apr 2019) enter
        # the portfolio: the active-today set grows, weights re-renormalize.
        if d in rebalance_dates_set and d != first_day and active_today:
            smi_value = _smi_value_on(d, row)
            if smi_value > 0:
                w_active = w[active_today] / w[active_today].sum()
                # Turnover = sum of absolute value changes / 2 (one-way turnover)
                turnover_chf = 0.0
                for t in active_today:
                    current_val = smi_shares[t] * row[t]
                    target_value = smi_value * w_active[t]
                    turnover_chf += abs(target_value - current_val)
                turnover_chf /= 2.0
                cost = turnover_chf * tx_cost
                total_tx_costs += cost
                # Apply rebalance to active tickers, then scale to absorb cost
                for t in active_today:
                    target_value = smi_value * w_active[t]
                    smi_shares[t] = target_value / row[t]
                if smi_value > 0:
                    shrink = (smi_value - cost) / smi_value
                    for t in active_today:
                        smi_shares[t] *= shrink

        # 5. Record state of day
        smi_value = _smi_value_on(d, row)
        btc_value_chf = btc_held * btc_price_d * fx_d if (btc_price_d and fx_d) else 0
        total = smi_value + btc_value_chf + dividend_cash
        records.append({
            "date": d, "smi_value": smi_value, "btc_value_chf": btc_value_chf,
            "btc_held": btc_held, "dividend_cash": dividend_cash,
            "total_value": total,
            "btc_pct": btc_value_chf / total if total > 0 else 0,
        })

    ts = pd.DataFrame(records).set_index("date")
    txs = pd.DataFrame(transactions) if transactions else pd.DataFrame()
    evts = pd.DataFrame(threshold_events) if threshold_events else pd.DataFrame()
    ts.attrs["total_tx_costs"] = total_tx_costs
    ts.attrs["total_wht"] = total_wht

    # ================= RENDITEZERLEGUNG (ATTRIBUTION) =====================
    # Zerlegt die BRUTTO-P&L (vor Management-/Performance-Gebühren, die
    # nachgelagert in apply_fees anfallen). Herleitung: die Verkaufserlöse
    # fliessen in die Aktien, deshalb heben sich die Brutto-Verkaufswerte
    # zwischen Aktien-Basis und BTC-Lots exakt auf:
    #
    #   NAV_end − Startkapital = Aktien + Dividenden + BTC(Start) + BTC(DCA)
    #
    # Sämtliche Transaktionskosten werden dabei von der jeweiligen Position
    # absorbiert (Aktien-Legs in der Aktienposition, BTC-Legs in den Lots).
    _last = ts.index[-1]
    _row_last = prices_clean.loc[_last] if _last in prices_clean.index else None
    _smi_end = float(ts["smi_value"].iloc[-1])
    _btc_px = get_btc_price(_last)
    _fx_px = get_fx(_last)
    _pxchf = (_btc_px * _fx_px) if (_btc_px and _fx_px) else 0.0

    _btc_init_end = btc_u_init * _pxchf
    _btc_dca_end = btc_u_dca * _pxchf

    equity_gain = _smi_end - att_equity_invested
    btc_init_gain = (_btc_init_end + att_sold_gross_init) - att_btc_init_invested
    btc_dca_gain = (_btc_dca_end + att_sold_gross_dca) - att_btc_dca_invested

    _btc_tot = btc_init_gain + btc_dca_gain
    _dca_share = (btc_dca_gain / _btc_tot) if abs(_btc_tot) > 1e-9 else float("nan")

    _nav_end = float(ts["total_value"].iloc[-1])
    _pnl_gross = _nav_end - initial_capital
    _recon = equity_gain + att_div_income + btc_init_gain + btc_dca_gain

    ts.attrs["attribution"] = {
        "equity_gain": equity_gain,             # Aktien inkl. aller Aktien-Trading-Kosten
        "dividend_income": att_div_income,      # netto nach 35% Verrechnungssteuer
        "btc_initial_gain": btc_init_gain,      # Lump Sum Tag 1, isoliert
        "btc_dca_gain": btc_dca_gain,           # dividendenfinanziert, isoliert
        "total_pnl_gross": _pnl_gross,          # vor Mgmt-/Perf-Gebühren
        "reconciliation_error": _recon - _pnl_gross,   # muss ~0 sein
        "dca_share": _dca_share,                # DIE Zahl
        "btc_initial_invested": att_btc_init_invested,
        "btc_dca_invested": att_btc_dca_invested,
        "years": max((ts.index[-1] - ts.index[0]).days / 365.25, 1e-9),
    }
    ts.attrs["dividend_cashflows"] = (
        pd.DataFrame(dividend_cashflows)
        if dividend_cashflows
        else pd.DataFrame(columns=["date", "ticker", "cash_chf"])
    )
    return ts, txs, evts


def simulate_smi_benchmarks(prices, dividends_df, initial_capital, weights,
                             rebalance_dates_set):
    """Run two SMI benchmark portfolios:
       - Total Return: dividends reinvested into the same paying stock
       - Price Only: dividends discarded (price index behavior)
    Returns DataFrame with columns: smi_tr, smi_price
    """
    prices = _clean_index(prices.copy())
    rebalance_dates_set = {_norm_ts(x) for x in rebalance_dates_set}
    available = [t for t in weights if t in prices.columns]
    if not available:
        return pd.DataFrame()

    w = pd.Series({t: weights[t] for t in available})
    w = w / w.sum()
    # Same fix as in run_strategy: keep dates where any ticker has a price,
    # work per-day with the active universe (handles late spin-offs like Alcon).
    prices_clean = prices[available].dropna(how="all")
    if prices_clean.empty:
        return pd.DataFrame()
    # Same per-ticker ffill as run_strategy (single missing quotes must not
    # value a holding at zero for a day); leading NaNs stay for late listings.
    prices_clean = prices_clean.ffill()

    div_lookup = {}
    if not dividends_df.empty:
        for _, r in dividends_df.iterrows():
            # Net dividend after non-reclaimable 35% Swiss withholding tax (AMC wrapper),
            # applied to the SMI Total Return benchmark for a consistent comparison.
            div_lookup[(_norm_ts(r["date"]), r["ticker"])] = \
                r["dividend_per_share"] * DIVIDEND_NET_FACTOR

    first_day = prices_clean.index[0]

    # Subset of tickers active on day 0
    active_t0 = [t for t in available if pd.notna(prices_clean.loc[first_day, t])]
    if not active_t0:
        return pd.DataFrame()
    w_t0 = w[active_t0] / w[active_t0].sum()

    # Two parallel portfolios with identical starting allocations across day-0 active tickers
    shares_tr = {t: 0.0 for t in available}
    shares_price = {t: 0.0 for t in available}
    for t in active_t0:
        s = (initial_capital * w_t0[t]) / prices_clean.loc[first_day, t]
        shares_tr[t] = s
        shares_price[t] = s

    records = []
    for d in prices_clean.index:
        row = prices_clean.loc[d]
        active_today = [t for t in available if pd.notna(row[t])]

        # Total Return: reinvest dividends into the same stock at today's price
        for t in active_today:
            key = (d, t)
            if key in div_lookup:
                dps = div_lookup[key]
                cash = shares_tr[t] * dps
                if cash > 0 and row[t] > 0:
                    shares_tr[t] += cash / row[t]
                # Price Only: dividends discarded (no change)

        # Quarterly rebalance to target weights, renormalized over active tickers
        if d in rebalance_dates_set and d != first_day and active_today:
            tr_total = sum(shares_tr[t] * row[t] for t in active_today)
            pr_total = sum(shares_price[t] * row[t] for t in active_today)
            w_active = w[active_today] / w[active_today].sum()
            for t in active_today:
                shares_tr[t] = (tr_total * w_active[t]) / row[t]
                shares_price[t] = (pr_total * w_active[t]) / row[t]

        smi_tr = sum(shares_tr[t] * row[t] for t in active_today)
        smi_price = sum(shares_price[t] * row[t] for t in active_today)
        records.append({"date": d, "smi_tr": smi_tr, "smi_price": smi_price})

    return pd.DataFrame(records).set_index("date")


def run_static_blend(prices, dividends_df, btc_prices_usd, fx_chf_usd,
                      initial_capital, weights, btc_pct):
    """The TRUE alpha benchmark: a passive, unmanaged X% Bitcoin / (1-X)%
    Equity blend, bought once at day 0 and never touched again.

    Bitcoin leg: bought once, held forever — no DCA, no threshold sell-down,
    no rebalancing of any kind.
    Equity leg: standard total-return treatment (net dividends reinvested
    into the same paying stock), no quarterly rebalance either — this is
    deliberately the LAZIEST possible comparator.

    If OAK Swiss Blue Chip / Bitcoin's CAGR/Sharpe does not beat this at the
    SAME starting allocation, any outperformance elsewhere is coming from
    carrying more average Bitcoin exposure over time (beta), not from the
    DCA/threshold mechanism (alpha). This isolates the mechanism, not the
    allocation choice.
    """
    prices = _clean_index(prices.copy())
    available = [t for t in weights if t in prices.columns]
    if not available:
        return pd.DataFrame()
    w = pd.Series({t: weights[t] for t in available})
    w = w / w.sum()
    prices_clean = prices[available].dropna(how="all").ffill()
    if prices_clean.empty:
        return pd.DataFrame()

    div_lookup = {}
    if not dividends_df.empty:
        for _, r in dividends_df.iterrows():
            div_lookup[(_norm_ts(r["date"]), r["ticker"])] = \
                r["dividend_per_share"] * DIVIDEND_NET_FACTOR

    btc_prices_usd = _clean_index(btc_prices_usd.copy())
    fx_chf_usd = _clean_index(fx_chf_usd.copy())

    def get_btc_price(d):
        sub = btc_prices_usd[btc_prices_usd.index <= d]
        return float(sub.iloc[-1]) if not sub.empty else None

    def get_fx(d):
        sub = fx_chf_usd[fx_chf_usd.index <= d]
        return float(sub.iloc[-1]) if not sub.empty else None

    first_day = prices_clean.index[0]
    active_t0 = [t for t in available if pd.notna(prices_clean.loc[first_day, t])]
    if not active_t0:
        return pd.DataFrame()
    w_t0 = w[active_t0] / w[active_t0].sum()

    btc_price_0, fx_0 = get_btc_price(first_day), get_fx(first_day)
    equity_capital = initial_capital * (1 - btc_pct)
    btc_capital = initial_capital * btc_pct

    shares = {t: 0.0 for t in available}
    for t in active_t0:
        shares[t] = (equity_capital * w_t0[t]) / prices_clean.loc[first_day, t]

    btc_held = 0.0
    if btc_price_0 and fx_0 and btc_capital > 0:
        btc_held = (btc_capital / fx_0) / btc_price_0   # bought once, held forever

    records = []
    for d in prices_clean.index:
        row = prices_clean.loc[d]
        active_today = [t for t in available if pd.notna(row[t])]
        for t in active_today:
            key = (d, t)
            if key in div_lookup:
                dps = div_lookup[key]
                cash = shares[t] * dps
                if cash > 0 and row[t] > 0:
                    shares[t] += cash / row[t]   # reinvested into the same stock
        equity_val = sum(shares[t] * row[t] for t in active_today)
        btc_price_d, fx_d = get_btc_price(d), get_fx(d)
        btc_val = btc_held * btc_price_d * fx_d if (btc_price_d and fx_d) else 0.0
        records.append({"date": d, "equity_value": equity_val,
                        "btc_value_chf": btc_val, "total_value": equity_val + btc_val,
                        "btc_pct": (btc_val / (equity_val + btc_val)
                                   if (equity_val + btc_val) > 0 else 0.0)})
    return pd.DataFrame(records).set_index("date")


def risk_metrics(values, risk_free_rate=0.01):
    """Annualized vol, max drawdown, Sharpe, Calmar from a daily value series.
    Only meaningful for fully mark-to-market series (true here — SMI/BTC has
    no at-par sleeve, unlike RE/BTC or Private Debt/BTC)."""
    if values is None or len(values) < 30:
        return dict(vol=np.nan, max_dd=np.nan, sharpe=np.nan, calmar=np.nan)
    rets = values.pct_change().dropna()
    vol = float(rets.std() * np.sqrt(252))
    running_max = values.cummax()
    dd = (values / running_max - 1.0)
    max_dd = float(dd.min())
    yrs = max((values.index[-1] - values.index[0]).days / 365.25, 1e-9)
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1 / yrs) - 1
    sharpe = float((cagr - risk_free_rate) / vol) if vol > 1e-9 else np.nan
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-9 else np.nan
    return dict(vol=vol, max_dd=max_dd, sharpe=sharpe, calmar=calmar, cagr=cagr)


def apply_fees(gross_values, initial_capital, mgmt_fee_annual=0.015,
               perf_fee_rate=0.15, hwm_hurdle=0.05,
               crystallization_freq="Quarterly", hurdle_type="Hard Hurdle"):
    """Apply management fee (daily accrual) + performance fee (period-end, HWM).

    crystallization_freq: 'Quarterly', 'Semi-Annual', or 'Annual'.
    hurdle_type:
      - 'Hard Hurdle': performance fee charged only on the NAV gain ABOVE the
        hurdle-grown HWM (the hurdle return is fee-free).
      - 'Soft Hurdle': if NAV clears the hurdle-grown HWM, the fee applies to the
        ENTIRE gain above the plain HWM (catch-up over the hurdle).
      - 'No Hurdle (HWM only)': fee on all gains above the HWM (no hurdle).

    Returns (net_series, total_mgmt_chf, total_perf_chf, fee_events_df).
    """
    if gross_values is None or gross_values.empty:
        return gross_values, 0.0, 0.0, pd.DataFrame()

    if crystallization_freq == "Quarterly":
        crystal_months = {3, 6, 9, 12}
        periods_per_year = 4
    elif crystallization_freq == "Semi-Annual":
        crystal_months = {6, 12}
        periods_per_year = 2
    else:
        crystal_months = {12}
        periods_per_year = 1

    # Adaptive observation frequency: derive periods-per-year from the actual
    # index (≈252 on an equity calendar, ≈365 on the BTC/RE daily calendar) so
    # the management-fee accrual matches the stated annual rate exactly.
    _span_years = max((gross_values.index[-1] - gross_values.index[0]).days
                      / 365.25, 1e-9)
    obs_per_year = max((len(gross_values) - 1) / _span_years, 1.0)

    daily_mgmt = mgmt_fee_annual / obs_per_year
    net = pd.Series(index=gross_values.index, dtype=float)
    # Start the net series at the actual day-0 gross value so the initial
    # transaction-cost drag is reflected in the net NAV as well (previously
    # net was rebased to initial_capital, silently dropping that cost).
    net.iloc[0] = float(gross_values.iloc[0])
    hwm = float(initial_capital)            # plain high water mark (post-fee highs)
    prev_cryst_date = gross_values.index[0]  # for pro-rata hurdle on partial periods
    total_mgmt = 0.0
    period_mgmt = 0.0   # management fee accrued within the current crystallization period
    total_perf = 0.0
    fee_events = []

    for i in range(1, len(gross_values)):
        d = gross_values.index[i]
        gross_today = float(gross_values.iloc[i])
        gross_prev = float(gross_values.iloc[i - 1])
        gross_ret = (gross_today / gross_prev - 1.0) if gross_prev > 0 else 0.0

        nv = net.iloc[i - 1] * (1.0 + gross_ret)
        mgmt_today = nv * daily_mgmt
        nv -= mgmt_today
        total_mgmt += mgmt_today
        period_mgmt += mgmt_today

        is_last = (i == len(gross_values) - 1)
        if is_last:
            is_period_end = True
        else:
            next_d = gross_values.index[i + 1]
            is_period_end = (d.month in crystal_months and next_d.month != d.month)

        if is_period_end:
            quarter = (d.month - 1) // 3 + 1
            period_label = f"Q{quarter} {d.year}"

            # The hurdle-grown threshold the NAV must clear this period.
            # Pro-rated by the ACTUAL elapsed time since the last crystallization
            # so partial periods (esp. the final one) are not held to a full
            # period's hurdle.
            _frac = max((d - prev_cryst_date).days, 0) / 365.25
            hurdle_threshold = hwm * (1.0 + hwm_hurdle * _frac)

            perf_today = 0.0
            excess = 0.0
            if hurdle_type == "No Hurdle (HWM only)":
                if nv > hwm:
                    excess = nv - hwm
                    perf_today = excess * perf_fee_rate
            elif hurdle_type == "Soft Hurdle":
                # Must clear the hurdle; if so, fee on the WHOLE gain above HWM
                if nv > hurdle_threshold:
                    excess = nv - hwm
                    perf_today = excess * perf_fee_rate
            else:  # Hard Hurdle (default)
                # Fee only on the gain ABOVE the hurdle threshold
                if nv > hurdle_threshold:
                    excess = nv - hurdle_threshold
                    perf_today = excess * perf_fee_rate

            if perf_today > 0:
                nv_after = nv - perf_today
                total_perf += perf_today
                fee_events.append({
                    "date": d, "period": period_label, "year": d.year,
                    "nav_before_perf": nv, "hwm_before": hwm, "excess": excess,
                    "mgmt_fee": period_mgmt,
                    "perf_fee": perf_today, "nav_after_perf": nv_after,
                })
                hwm = max(hwm, nv_after)
                nv = nv_after
            else:
                fee_events.append({
                    "date": d, "period": period_label, "year": d.year,
                    "nav_before_perf": nv, "hwm_before": hwm,
                    "excess": nv - hwm, "mgmt_fee": period_mgmt,
                    "perf_fee": 0.0, "nav_after_perf": nv,
                })
                hwm = max(hwm, nv)

            prev_cryst_date = d
            period_mgmt = 0.0   # reset bucket for next period

        net.iloc[i] = nv

    return net, total_mgmt, total_perf, pd.DataFrame(fee_events)


def monthly_returns_matrix(values):
    """Return a DataFrame of monthly returns (rows: year, cols: month).
    The first month's return is measured against the series' starting value so
    no month (and no full-year figure) is silently dropped."""
    if values is None or values.empty:
        return pd.DataFrame()
    monthly = values.resample("ME").last()
    if len(monthly) < 1:
        return pd.DataFrame()
    # Prepend the starting value as an anchor so the first month gets a return
    start = values.iloc[0]
    anchor_idx = values.index[0] - pd.Timedelta(days=1)
    monthly_anchored = pd.concat([pd.Series([start], index=[anchor_idx]), monthly])
    mret = monthly_anchored.pct_change().dropna()
    if mret.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"ret": mret.values}, index=mret.index)
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Full-year column: compound the monthly returns actually present that year
    def _fy(row):
        vals = [v for v in row.values if pd.notna(v)]
        if not vals:
            return np.nan
        prod = 1.0
        for v in vals:
            prod *= (1 + v)
        return prod - 1
    pivot["YTD"] = pivot.apply(_fy, axis=1)
    return pivot


def compute_drawdown(values):
    """Drawdown series — percent below running peak."""
    if values.empty:
        return pd.Series(dtype=float)
    cummax = values.cummax()
    return (values - cummax) / cummax


def max_drawdown_info(values):
    """Max DD value, peak date, trough date, recovery date, duration in days."""
    if values.empty:
        return {"mdd": 0.0, "peak": None, "trough": None, "recovery": None, "duration": 0}
    cummax = values.cummax()
    dd = (values - cummax) / cummax
    mdd = float(dd.min())
    if mdd == 0:
        return {"mdd": 0.0, "peak": None, "trough": None, "recovery": None, "duration": 0}
    trough_date = dd.idxmin()
    peak_date = values.loc[:trough_date].idxmax()
    peak_value = float(values.loc[peak_date])
    post = values.loc[trough_date:]
    recovered = post[post >= peak_value]
    recovery_date = recovered.index[0] if not recovered.empty else None
    if recovery_date is not None:
        duration = (recovery_date - peak_date).days
    else:
        duration = (values.index[-1] - peak_date).days
    return {
        "mdd": mdd, "peak": peak_date, "trough": trough_date,
        "recovery": recovery_date, "duration": duration,
    }


def compute_risk_metrics(values, risk_free_rate=0.01, base_value=None):
    """Comprehensive risk metrics from a daily CHF value series.
    base_value: if given, total return and CAGR are measured against this
    (e.g. the investor's initial capital) instead of the first series value,
    so the figures match the KPI boxes exactly."""
    if values is None or values.empty or len(values) < 30:
        return {}
    returns = values.pct_change().dropna()
    if returns.empty:
        return {}
    n_days = len(returns)
    # CAGR must use CALENDAR time so it matches the KPI boxes exactly.
    cal_years = (values.index[-1] - values.index[0]).days / 365.25
    # Adaptive annualization: observations per calendar year from the index
    # itself (≈252 equity calendar, ≈365 BTC/RE daily calendar) — using a
    # hardcoded 252 understates volatility ~17% on a 365-day calendar.
    obs_per_year = max(n_days / cal_years, 1.0) if cal_years > 0 else 252.0

    start_val = float(base_value) if base_value else float(values.iloc[0])
    total_return = float(values.iloc[-1] / start_val - 1)
    cagr = float((values.iloc[-1] / start_val) ** (1 / cal_years) - 1) if cal_years > 0 else 0.0
    vol_ann = float(returns.std() * np.sqrt(obs_per_year))

    sharpe = (cagr - risk_free_rate) / vol_ann if vol_ann > 0 else 0.0

    downside = returns[returns < 0]
    downside_vol = (float(downside.std() * np.sqrt(obs_per_year))
                    if not downside.empty else 0.0)
    sortino = (cagr - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0

    dd_info = max_drawdown_info(values)
    max_dd = dd_info["mdd"]
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    monthly = values.resample("ME").last()
    mret = monthly.pct_change().dropna()
    best_month = float(mret.max()) if not mret.empty else 0.0
    worst_month = float(mret.min()) if not mret.empty else 0.0
    pct_pos = float((mret > 0).mean()) if not mret.empty else 0.0

    var_95 = float(mret.quantile(0.05)) if not mret.empty else 0.0
    cvar_subset = mret[mret <= var_95]
    cvar_95 = float(cvar_subset.mean()) if not cvar_subset.empty else 0.0

    return {
        "total_return": total_return, "cagr": cagr, "vol_ann": vol_ann,
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "downside_vol": downside_vol,
        "max_drawdown": max_dd, "dd_peak": dd_info["peak"],
        "dd_trough": dd_info["trough"], "dd_recovery": dd_info["recovery"],
        "dd_duration_days": dd_info["duration"],
        "best_month": best_month, "worst_month": worst_month,
        "pct_positive_months": pct_pos,
        "var_95_monthly": var_95, "cvar_95_monthly": cvar_95,
    }


def compute_benchmark_metrics(strategy, benchmark, risk_free_rate=0.01):
    """Strategy vs benchmark: alpha (Jensen), beta, tracking error, IR, correlation."""
    if strategy is None or benchmark is None or strategy.empty or benchmark.empty:
        return {}
    aligned = pd.concat([strategy, benchmark], axis=1, join="inner").dropna()
    if aligned.empty or len(aligned) < 30:
        return {}
    aligned.columns = ["s", "b"]
    s_ret = aligned["s"].pct_change().dropna()
    b_ret = aligned["b"].pct_change().dropna()
    combined = pd.concat([s_ret, b_ret], axis=1, join="inner").dropna()
    combined.columns = ["s", "b"]
    if combined.empty:
        return {}
    corr = float(combined["s"].corr(combined["b"]))
    cov = float(combined["s"].cov(combined["b"]))
    var_b = float(combined["b"].var())
    beta = cov / var_b if var_b > 0 else 0.0
    excess = combined["s"] - combined["b"]
    _bm_years = max((aligned.index[-1] - aligned.index[0]).days / 365.25, 1e-9)
    _bm_opy = max((len(aligned) - 1) / _bm_years, 1.0)
    te = float(excess.std() * np.sqrt(_bm_opy))
    info_ratio = float(excess.mean() * _bm_opy / te) if te > 0 else 0.0
    years = _bm_years
    s_cagr = float((aligned["s"].iloc[-1] / aligned["s"].iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0
    b_cagr = float((aligned["b"].iloc[-1] / aligned["b"].iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0
    alpha = s_cagr - (risk_free_rate + beta * (b_cagr - risk_free_rate))
    return {
        "correlation": corr, "r_squared": corr ** 2,
        "beta": beta, "alpha": alpha,
        "tracking_error": te, "information_ratio": info_ratio,
    }


def _fmt_pct(x, decimals=2):
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:+.{decimals}f}%" if x < 0 else f"{x*100:.{decimals}f}%"


def _fmt_num(x, decimals=2):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"


def fmt_chf(x):
    """Compact CHF for KPI cards. Abbreviates at >=1m so 8-10 digit values
    never overflow the box; decimals shrink as magnitude grows and the bracket
    is chosen rounding-safe, so every string stays <=13 chars. Below 1m: full
    thousands-separated."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(x):
        return "n/a"
    a = abs(x)
    if a >= 1e6:
        v, unit = (x / 1e9, "Mrd.") if a >= 1e9 else (x / 1e6, "Mio.")
        av = abs(v)
        if av >= 99.95:        # rounds to >=100 -> no decimals
            s = f"{v:,.0f}"
        elif av >= 9.995:      # rounds to >=10  -> 1 decimal
            s = f"{v:,.1f}"
        else:
            s = f"{v:,.2f}"
        return f"CHF {s} {unit}"
    return f"CHF {x:,.0f}"


def footer():
    st.markdown(
        f"""<div class='oak-footer'>
        For Illustrative Purposes · Not Investment Advice · Past Performance is no Guarantee of Future Results
        <span class='oak-mark'>Oakwood Capital · Quantitative Research</span>
        </div>""", unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if _show_results:
    tickers = list(SMI_CONSTITUENTS.keys())
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    if weighting_method.startswith("Equal"):
        weights = {t: 5.0 for t in tickers}
    else:
        weights = {t: v[1] for t, v in SMI_CONSTITUENTS.items()}
        weights = {t: min(w, 18.0) for t, w in weights.items()}

    with st.spinner("Loading SMI constituent prices ..."):
        prices = fetch_prices(tickers, start_str, end_str)
    with st.spinner("Loading dividend history ..."):
        divs = fetch_dividends(tickers, start_str, end_str)
    with st.spinner("Loading FX (USDCHF) ..."):
        fx = fetch_series("USDCHF=X", start_str, end_str)
    with st.spinner("Loading Bitcoin series ..."):
        btc_spot = fetch_series("BTC-USD", start_str, end_str)
        if btc_source.startswith("IBIT"):
            ibit = fetch_series("IBIT", "2024-01-11", end_str)
            if not btc_spot.empty and not ibit.empty:
                overlap = pd.concat([btc_spot, ibit], axis=1, join="inner").dropna()
                overlap.columns = ["btc", "ibit"]
                if not overlap.empty:
                    scale = overlap["btc"].iloc[0] / overlap["ibit"].iloc[0]
                    btc_series = ibit * scale
                else:
                    btc_series = ibit
            else:
                btc_series = ibit if not ibit.empty else btc_spot
        elif btc_source.startswith("BTC-USD bis"):
            cutoff = pd.Timestamp("2024-01-11")
            ibit = fetch_series("IBIT", "2024-01-11", end_str)
            if not ibit.empty and not btc_spot.empty:
                btc_pre = btc_spot[btc_spot.index < cutoff]
                btc_at_cut = btc_spot[btc_spot.index <= cutoff]
                if not btc_at_cut.empty:
                    scale = btc_at_cut.iloc[-1] / ibit.iloc[0]
                    btc_series = pd.concat([btc_pre, ibit * scale]).sort_index()
                    btc_series = _clean_index(btc_series)
                else:
                    btc_series = btc_spot
            else:
                btc_series = btc_spot
        else:
            btc_series = btc_spot
        btc_series = _clean_index(btc_series)

    if prices.empty:
        st.error("No price data received.")
        st.stop()

    # BTC and FX are mandatory for this strategy — without them the simulation is
    # meaningless. Fail with a clear message instead of crashing deeper down.
    _missing_feeds = []
    if btc_series is None or btc_series.empty:
        _missing_feeds.append("Bitcoin (BTC-USD)")
    if fx is None or fx.empty:
        _missing_feeds.append("FX (USDCHF=X)")
    if _missing_feeds:
        st.error(
            "⚠️ Keine Daten für: " + ", ".join(_missing_feeds) + ". "
            "Die Simulation braucht beide Reihen. Das ist meist eine temporäre "
            "Yahoo-Finance-Störung oder ein Rate-Limit — bitte in ein paar Minuten "
            "erneut versuchen (ggf. Cache leeren)."
        )
        st.stop()

    # Handle tickers that failed to load (delisted, ticker change, data outage)
    loaded_tickers = list(prices.columns)
    missing = [t for t in tickers if t not in loaded_tickers]
    if missing:
        missing_names = [f"{SMI_CONSTITUENTS[t][0]} ({t})" for t in missing if t in SMI_CONSTITUENTS]
        st.warning(
            f"⚠️ Price data unavailable for: {', '.join(missing_names)}. "
            f"The backtest continues with the remaining {len(loaded_tickers)} titles, "
            f"and their weights are renormalized to 100 %."
        )
        # Renormalize weights across the loaded tickers only
        weights = {t: w for t, w in weights.items() if t in loaded_tickers}
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {t: w / total_w * 100.0 for t, w in weights.items()}

    rebal_dates = get_rebalance_dates(prices.index, rebalance_freq)

    # =====================================================================
    # DATENQUALITÄT — Ausreisser-Diagnose
    # Findet den genauen Titel + das genaue Datum hinter unplausiblen
    # Tagesbewegungen (z.B. eine fehlerhafte Split-Zuordnung oder ein
    # schlechter Yahoo-Finance-Tick), BEVOR sie sich unbemerkt in die
    # Monatsrenditen-Heatmap durchschlagen. Kein Rätselraten — jede
    # auffällige Bewegung wird mit Titel, Datum und Vorher/Nachher-Kurs
    # ausgewiesen.
    # =====================================================================
    with st.expander("🔍 Datenqualität — Ausreisser-Diagnose (Kurssprünge)"):
        st.caption(
            "Prüft jeden geladenen Titel sowie Bitcoin und den USD/CHF-Kurs auf "
            "einzelne Tagesbewegungen über der Sanity-Schwelle. Eine reale SMI-"
            "Aktie bewegt sich praktisch nie um mehr als 25% an einem Tag ausserhalb "
            "eines Delistings/einer Fusion — ein Treffer hier ist meist eine "
            "fehlerhafte Split-Zuordnung oder ein Datenfehler des Anbieters, kein "
            "echtes Marktereignis.")
        _outlier_thresh = st.slider("Sanity-Schwelle (Tagesbewegung, %)", 10, 60, 25, 5,
                                    key="smi_dq_thresh") / 100.0
        _outlier_rows = []
        for _t in prices.columns:
            _s = prices[_t].dropna()
            if len(_s) < 2:
                continue
            _ret = _s.pct_change().dropna()
            _hits = _ret[_ret.abs() > _outlier_thresh]
            for _d, _r in _hits.items():
                _prev_idx = _s.index[_s.index.get_loc(_d) - 1]
                _outlier_rows.append({
                    "Titel": SMI_CONSTITUENTS.get(_t, (_t,))[0], "Ticker": _t,
                    "Datum": _d.strftime("%Y-%m-%d"), "Tagesbewegung": f"{_r*100:+.1f}%",
                    "Kurs davor": f"{_s.loc[_prev_idx]:.2f}", "Kurs danach": f"{_s.loc[_d]:.2f}",
                })
        # BTC und FX ebenfalls prüfen (andere, meist höhere Toleranz, da BTC volatiler ist)
        for _label, _series, _tol in [("Bitcoin (BTC-USD)", btc_series, max(_outlier_thresh, 0.35)),
                                       ("USD/CHF", fx, _outlier_thresh)]:
            _s = _series.dropna()
            if len(_s) < 2:
                continue
            _ret = _s.pct_change().dropna()
            _hits = _ret[_ret.abs() > _tol]
            for _d, _r in _hits.items():
                _prev_idx = _s.index[_s.index.get_loc(_d) - 1]
                _outlier_rows.append({
                    "Titel": _label, "Ticker": "—",
                    "Datum": _d.strftime("%Y-%m-%d"), "Tagesbewegung": f"{_r*100:+.1f}%",
                    "Kurs davor": f"{_s.loc[_prev_idx]:.2f}", "Kurs danach": f"{_s.loc[_d]:.2f}",
                })

        if _outlier_rows:
            _odf = pd.DataFrame(_outlier_rows)
            st.error(f"⚠️ **{len(_odf)} auffällige Tagesbewegung(en) gefunden** — "
                     "vor der weiteren Kalibrierung prüfen. Nicht jeder Treffer ist "
                     "ein Fehler: eine reale Kapitalmassnahme (Spin-off, Fusion) oder "
                     "ein historisch dokumentierter Markt-Crash können ebenfalls "
                     "grosse, aber ECHTE Tagesbewegungen erzeugen.")
            st.dataframe(_odf, use_container_width=True, hide_index=True)

            # Für jeden auffälligen AKTIENTITEL (nicht BTC/FX) zusätzlich die
            # ROHEN Split-Ereignisse zeigen, die Yahoo für diesen Ticker meldet —
            # damit sichtbar wird, ob eine Split-Ratio die Ursache ist, statt es
            # zu vermuten.
            _flagged_tickers = sorted(set(
                r["Ticker"] for r in _outlier_rows if r["Ticker"] != "—"))
            if _flagged_tickers:
                st.markdown("###### Rohe Split-Ereignisse der auffälligen Titel (Yahoo Finance)")
                for _ft in _flagged_tickers:
                    _sp = _get_split_series(_ft)
                    _fname = SMI_CONSTITUENTS.get(_ft, (_ft,))[0]
                    if _sp is None or _sp.empty:
                        st.caption(f"**{_fname} ({_ft})**: keine Split-Ereignisse in den "
                                   "Yahoo-Finance-Daten gefunden — die Ursache liegt dann "
                                   "vermutlich nicht bei der Split-Bereinigung.")
                    else:
                        _spdf = pd.DataFrame({
                            "Datum": [d.strftime("%Y-%m-%d") for d in _sp.index],
                            "Gemeldete Split-Ratio": [float(v) for v in _sp.values],
                        })
                        st.caption(f"**{_fname} ({_ft})** — {len(_sp)} gemeldete(s) "
                                   "Split-Ereignis(se):")
                        st.dataframe(_spdf, use_container_width=True, hide_index=True)
                        # KORREKTUR: nicht mehr nach Ratio-Grösse urteilen (Sika hatte
                        # einen echten 60:1-Split, das wäre fälschlich "unplausibel"
                        # gewesen) — stattdessen prüfen, ob die Ratio den tatsächlich
                        # beobachteten Kurssprung im Rohdatensatz erklärt. Derselbe
                        # Test wie in _apply_split_adjustment, hier nur zur Anzeige.
                        _raw_t = prices[_ft] if _ft in prices.columns else None
                        _sp_naive = _sp.copy()
                        try:
                            if _sp_naive.index.tz is not None:
                                _sp_naive.index = _sp_naive.index.tz_localize(None)
                        except (AttributeError, TypeError):
                            pass
                        for _d, _r in _sp_naive.items():
                            if _raw_t is None:
                                continue
                            _before = _raw_t[_raw_t.index < _d]
                            _after = _raw_t[_raw_t.index >= _d]
                            if _before.empty or _after.empty:
                                continue
                            _pb, _pa = _before.iloc[-1], _after.iloc[0]
                            if _pb <= 0 or _pa <= 0:
                                continue
                            _implied = _pb / _pa
                            _match = 0.6 <= (_implied / float(_r)) <= 1.6
                            if _match:
                                st.caption(f"✓ Ratio {_r:.2f} am {_d:%Y-%m-%d} erklärt den "
                                           f"beobachteten Kurssprung (implizite Ratio "
                                           f"{_implied:.2f}) — sieht nach echtem Split aus, "
                                           "wird angewendet.")
                            else:
                                st.warning(
                                    f"⚑ Ratio {_r:.2f} am {_d:%Y-%m-%d} erklärt den "
                                    f"beobachteten Kurssprung NICHT (implizite Ratio "
                                    f"{_implied:.2f}, weicht stark ab) — wird verworfen, "
                                    "vermutlich Datenfehler des Anbieters.")
        else:
            st.success("Keine Tagesbewegung über der Schwelle gefunden.")

    _tcf_map = {"Monatlich (Standard)": None, "Quartalsweise": "Quartalsweise",
                "Halbjährlich": "Halbjährlich"}
    _tcf = _tcf_map[threshold_check_freq]
    threshold_dates = (None if _tcf is None
                       else get_rebalance_dates(prices.index, _tcf))

    with st.spinner("Running integrated simulation ..."):
        ts, txs, evts = run_strategy(
            prices, divs, btc_series, fx,
            initial_capital, weights,
            initial_btc_pct, upper_threshold, target_btc_pct,
            rebal_dates, dca_months, tx_cost_bps=tx_cost_bps,
            threshold_check_dates_set=threshold_dates,
        )

    if ts is None or ts.empty or "total_value" not in ts.columns:
        st.error(
            "Strategy could not be executed — no valid price series was built. "
            "This is usually a temporary Yahoo Finance data issue. Please wait a "
            "moment and click 'Run Backtest' again, or try a shorter date range."
        )
        st.stop()

    with st.spinner("Computing SMI benchmarks ..."):
        bench = simulate_smi_benchmarks(prices, divs, initial_capital, weights, rebal_dates)

    with st.spinner("Applying fee structure ..."):
        ts_net, total_mgmt_fees, total_perf_fees, fee_events_df = apply_fees(
            ts["total_value"], initial_capital,
            mgmt_fee_annual=mgmt_fee_pct,
            perf_fee_rate=perf_fee_pct,
            hwm_hurdle=hwm_hurdle_pct,
            crystallization_freq=crystallization_freq,
            hurdle_type=hurdle_type,
        )
        ts["total_value_net"] = ts_net

    # =====================================================================
    # KPIs
    # =====================================================================
    st.markdown("## Performance-Übersicht")

    # ---- KPI Row 1: Performance vs benchmarks (NET of fees) ----
    smi_final = ts["smi_value"].iloc[-1]
    btc_final = ts["btc_value_chf"].iloc[-1]
    strategy_gross = ts["total_value"].iloc[-1]
    strategy_net = ts["total_value_net"].iloc[-1]
    years = (ts.index[-1] - ts.index[0]).days / 365.25
    strat_gross_cagr = (strategy_gross / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    strat_net_cagr = (strategy_net / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    smi_tr_final = float(bench["smi_tr"].iloc[-1]) if not bench.empty else initial_capital
    smi_price_final = float(bench["smi_price"].iloc[-1]) if not bench.empty else initial_capital
    smi_tr_cagr = (smi_tr_final / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    smi_price_cagr = (smi_price_final / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    excess_vs_tr = strat_net_cagr - smi_tr_cagr
    excess_vs_price = strat_net_cagr - smi_price_cagr
    fee_drag = strat_gross_cagr - strat_net_cagr

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategie (netto)", fmt_chf(strategy_net),
              f"{(strategy_net/initial_capital - 1)*100:+.1f}%")
    c2.metric("Strategie (brutto)", fmt_chf(strategy_gross),
              f"Gebührenlast: {fee_drag*100:.2f}% p.a.", delta_color="off")
    c3.metric("SMI Total Return", fmt_chf(smi_tr_final),
              f"{(smi_tr_final/initial_capital - 1)*100:+.1f}%")
    c4.metric("SMI Kursindex", fmt_chf(smi_price_final),
              f"{(smi_price_final/initial_capital - 1)*100:+.1f}%")

    # ---- KPI Row 2: CAGR comparison + alpha ----
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Netto-CAGR", f"{strat_net_cagr*100:.2f}%",
              f"nach allen Gebühren · {years:.1f} years", delta_color="off")
    c6.metric("Brutto-CAGR", f"{strat_gross_cagr*100:.2f}%",
              f"vor Gebühren", delta_color="off")
    c7.metric("Mehrrendite vs. SMI TR", f"{excess_vs_tr*100:+.2f}% p.a.",
              f"net of fees")
    c8.metric("Mehrrendite vs. Kursindex", f"{excess_vs_price*100:+.2f}% p.a.",
              f"net of fees")

    # ---- KPI Row 3: Fee breakdown ----
    total_tx_costs = float(ts.attrs.get("total_tx_costs", 0.0))
    total_wht = float(ts.attrs.get("total_wht", 0.0))
    fees_total = total_mgmt_fees + total_perf_fees + total_tx_costs
    fees_total_pct_initial = (fees_total / initial_capital) * 100
    n_perf_periods = int(fee_events_df["perf_fee"].gt(0).sum()) if not fee_events_df.empty else 0
    n_perf_total_periods = int(len(fee_events_df)) if not fee_events_df.empty else 0

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Management-Gebühren", fmt_chf(total_mgmt_fees),
              f"{mgmt_fee_pct*100:.2f}% p.a. on NAV", delta_color="off")
    c10.metric("Performance-Gebühren", fmt_chf(total_perf_fees),
               f"{perf_fee_pct*100:.0f}% × excess · {n_perf_periods} of {n_perf_total_periods} {crystallization_freq.lower()} periods charged", delta_color="off")
    c11.metric("Transaktionskosten", fmt_chf(total_tx_costs),
               f"{tx_cost_bps:.0f} bps per trade", delta_color="off")
    c12.metric("Gesamtkosten (inkl. TX)", fmt_chf(fees_total),
               f"{fees_total_pct_initial:.1f}% of initial capital", delta_color="off")

    st.caption(
        "Fee-Mechanik (bewusst abweichend von OAK RE/BTC und Private Debt): "
        "Gebühren werden hier als **nachgelagerter NAV-Abschlag** verbucht — "
        "ökonomisch ein proportionaler Trim beider Sleeves, nie ein gezielter "
        "BTC-Verkauf. Ein Stundungs-Wasserfall (wie bei den Produkten mit "
        "illiquidem Kern) ist hier nicht nötig: SMI-Aktien und Bitcoin sind beide "
        "liquide, es gibt keinen illiquiden Kern zu schützen, und die "
        "Netto-Dividende fliesst voll in den BTC-DCA (die Fee zehrt nicht am DCA).")

    # ---- KPI Row 4: Strategy mechanics ----
    n_thresholds = len(evts)
    total_btc_sold = float(-txs[txs["type"]=="SELL"]["btc_amount"].sum()) if not txs.empty else 0
    chf_redeployed = float(-txs[txs["type"]=="SELL"]["chf_amount"].sum()) if not txs.empty else 0
    n_buys = int((txs["type"] == "BUY").sum()) if not txs.empty else 0

    c13, c14, c15, c16 = st.columns(4)
    c13.metric("Threshold-Rebalancings", f"{n_thresholds}",
               f"trigger > {upper_threshold*100:.0f}%")
    c14.metric("BTC gekauft (total)", f"{ts['btc_held'].iloc[-1] + total_btc_sold:.4f}",
               f"{n_buys} transactions")
    c15.metric("BTC verkauft", f"{total_btc_sold:.4f}",
               f"CHF {chf_redeployed:,.0f} to SMI")
    c16.metric("Aktien / BTC (aktuell)",
               f"{smi_final/strategy_gross*100:.0f}% / {btc_final/strategy_gross*100:.0f}%",
               f"BTC: {ts['btc_held'].iloc[-1]:.4f}")

    # =====================================================================
    # Portfolio Evolution
    # =====================================================================
    # =====================================================================
    # RENDITEZERLEGUNG (ATTRIBUTION) — die zentrale Ehrlichkeits-Kennzahl.
    # Zeigt, wie viel der Rendite wirklich aus dem dividendenfinanzierten DCA
    # stammt und wie viel aus der Bitcoin-Startallokation vom Tag 1.
    # =====================================================================
    st.markdown("## Renditezerlegung")
    _att = ts.attrs.get("attribution", {})
    if _att:
        _yrs = _att["years"]

        def _pp(chf):
            return (chf / initial_capital) / _yrs * 100

        st.markdown(
            "<p style='color:#A9B5A4;margin-top:-6px'>Zerlegung der Brutto-P&amp;L "
            "in ihre Quellen (vor Management- und Performance-Gebühren). Die "
            "Positionen summieren sich exakt auf die Gesamt-P&amp;L; sämtliche "
            "Transaktionskosten sind in den jeweiligen Positionen enthalten.</p>",
            unsafe_allow_html=True)

        _rows = [
            ("Aktien-Kapitalwertentwicklung (SMI)", _att["equity_gain"]),
            ("Dividendenerträge (netto, nach 35% VSt)", _att["dividend_income"]),
            ("Bitcoin — Startallokation (Tag 1)", _att["btc_initial_gain"]),
            ("Bitcoin — dividendenfinanzierter DCA", _att["btc_dca_gain"]),
        ]
        _html = ["<table class='oak-metrics-table'><thead><tr>"
                 "<th>Beitrag</th><th>CHF</th><th>%-Punkte p.a.</th></tr></thead><tbody>"]
        for _lab, _v in _rows:
            _col = OAK_GOLD if "DCA" in _lab else OAK_CREAM
            _html.append(
                f"<tr><td class='metric-label'>{_lab}</td>"
                f"<td class='strategy-col' style='color:{_col}'>{_v:+,.0f}</td>"
                f"<td style='color:{_col}'>{_pp(_v):+.2f}</td></tr>")
        _html.append(
            f"<tr class='oak-section'><td>Total brutto (= NAV − Startkapital)</td>"
            f"<td>{_att['total_pnl_gross']:+,.0f}</td>"
            f"<td>{_pp(_att['total_pnl_gross']):+.2f}</td></tr>")
        _html.append("</tbody></table>")
        st.markdown("".join(_html), unsafe_allow_html=True)

        _dca = _att["dca_share"]
        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric("DCA-Anteil am BTC-Gewinn",
                      "n/a" if _dca != _dca else f"{_dca*100:.1f}%")
            st.caption("DCA / (DCA + Startallokation)")
        with b2:
            st.metric("BTC Startallokation", fmt_chf(_att["btc_initial_invested"]))
            st.caption("am Tag 1 investiert")
        with b3:
            st.metric("BTC via Dividenden investiert", fmt_chf(_att["btc_dca_invested"]))
            st.caption("über die gesamte Laufzeit")

        if _dca == _dca:
            if _dca < 0.30:
                st.warning(
                    f"⚠️ **Der DCA-Anteil liegt bei {_dca*100:.1f}%.** Der weit "
                    "überwiegende Teil des Bitcoin-Gewinns stammt aus der "
                    "Startallokation vom ersten Tag, nicht aus dem "
                    "dividendenfinanzierten DCA. Unterhalb von ~30% beschreibt "
                    "«dividendenfinanzierte BTC-Allokation» eher das Etikett als "
                    "den Mechanismus. Wichtig zur Einordnung: der DCA-Anteil ist "
                    "**invers zum Einstiegsglück** — je schlechter der "
                    "Einstiegszeitpunkt, desto grösser der Beitrag des DCA. Ein "
                    "tiefer Wert bedeutet hier vor allem, dass der Backtest-"
                    "Zeitraum für die Startallokation günstig lag.")
            else:
                st.success(
                    f"✅ Der DCA-Anteil liegt bei {_dca*100:.1f}% — der "
                    "Dividendenmechanismus trägt den Bitcoin-Beitrag substanziell.")

        st.caption(
            f"Abstimmdifferenz der Zerlegung: "
            f"{_att['reconciliation_error']:+.2f} CHF · Gebühren "
            f"({fmt_chf(total_mgmt_fees + total_perf_fees)}) werden nachgelagert "
            "auf die Brutto-Kurve angewandt und sind hier nicht enthalten.")

    st.markdown("## Portfolioentwicklung vs. Benchmarks")
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    # Net strategy (primary, gold, filled)
    fig.add_trace(go.Scatter(x=ts.index, y=ts["total_value_net"],
                             name="Strategie (netto)",
                             line=dict(color=OAK_GOLD, width=3, shape="spline", smoothing=0.5),
                             fill="tozeroy", fillcolor="rgba(201,169,97,0.10)"))
    # Gross strategy (faded dotted)
    fig.add_trace(go.Scatter(x=ts.index, y=ts["total_value"],
                             name="Strategie (brutto)",
                             line=dict(color=OAK_GOLD, width=1.2, dash="dot"),
                             opacity=0.55))
    if not bench.empty:
        fig.add_trace(go.Scatter(x=bench.index, y=bench["smi_tr"],
                                 name="SMI Total Return",
                                 line=dict(color=OAK_SAGE, width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=bench.index, y=bench["smi_price"],
                                 name="SMI Kursindex",
                                 line=dict(color=OAK_SAGE_DIM, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=ts.index, y=ts["smi_value"],
                             name="Strategie · Aktien-Sleeve",
                             line=dict(color=OAK_CREAM, width=1.2),
                             opacity=0.7))
    fig.add_trace(go.Scatter(x=ts.index, y=ts["btc_value_chf"],
                             name="Strategie · BTC-Sleeve",
                             line=dict(color=OAK_BTC, width=1.2),
                             opacity=0.7))
    # Mark threshold rebalances
    if not evts.empty:
        evts_with_values = evts.copy()
        evts_with_values["total_at_event"] = evts_with_values["date"].map(
            lambda d: ts.loc[d, "total_value_net"] if d in ts.index else None
        )
        fig.add_trace(go.Scatter(
            x=evts_with_values["date"], y=evts_with_values["total_at_event"],
            mode="markers", name="Threshold-Rebalancing",
            marker=dict(symbol="diamond", size=11,
                        color=OAK_RED, line=dict(color=OAK_CREAM, width=1.5)),
        ))
    # Mark performance fee events
    if not fee_events_df.empty:
        perf_paid = fee_events_df[fee_events_df["perf_fee"] > 0]
        if not perf_paid.empty:
            fig.add_trace(go.Scatter(
                x=perf_paid["date"], y=perf_paid["nav_after_perf"],
                mode="markers", name="Performance-Gebühr belastet",
                marker=dict(symbol="triangle-down", size=11,
                            color=OAK_CREAM, line=dict(color=OAK_GOLD, width=1.5)),
            ))
    fig = style_plotly(fig, height=580)
    fig.update_yaxes(title_text="Value (CHF)", tickformat=",.0f")

    # Endpoint value labels for the main series, with vertical anti-overlap
    # spreading so close endpoints never collide.
    _ep = [(ts.index[-1], float(ts["total_value_net"].iloc[-1]), OAK_GOLD)]
    if not bench.empty:
        _ep.append((bench.index[-1], float(bench["smi_tr"].iloc[-1]), OAK_SAGE))
        _ep.append((bench.index[-1], float(bench["smi_price"].iloc[-1]), OAK_SAGE_DIM))
    _ys = sorted(range(len(_ep)), key=lambda i: _ep[i][1])
    _lo_y = min(v for _, v, _c in _ep)
    _hi_y = max(v for _, v, _c in _ep)
    _min_gap = max((_hi_y - _lo_y), _hi_y * 0.02) * 0.045
    _pos = {}
    _prev = None
    for _i in _ys:
        _y = _ep[_i][1]
        if _prev is not None and _y - _prev < _min_gap:
            _y = _prev + _min_gap
        _pos[_i] = _y
        _prev = _y
    for _i, (_x, _v, _c) in enumerate(_ep):
        fig.add_annotation(x=_x, y=_pos[_i], text=fmt_chf(_v), showarrow=False,
                           xanchor="left", xshift=8, yanchor="middle",
                           font=dict(family="'Inter', sans-serif", size=11, color=_c))
    st.plotly_chart(fig, use_container_width=True)

    # =====================================================================
    # RISK ANALYTICS
    # =====================================================================
    st.markdown("## Risikoanalyse")

    # Compute metrics for all three series — Strategy is NET of fees.
    # All measured against initial_capital so CAGR/Total Return match the KPI boxes.
    strat_m = compute_risk_metrics(ts["total_value_net"], risk_free_rate, base_value=initial_capital)
    tr_m = compute_risk_metrics(bench["smi_tr"], risk_free_rate, base_value=initial_capital) if not bench.empty else {}
    pr_m = compute_risk_metrics(bench["smi_price"], risk_free_rate, base_value=initial_capital) if not bench.empty else {}
    bm_tr = compute_benchmark_metrics(ts["total_value_net"],
                                       bench["smi_tr"] if not bench.empty else pd.Series(dtype=float),
                                       risk_free_rate)

    # ---- Master risk metrics table (HTML for full styling control) ----
    def _row(label, key, fmt="pct", hint=""):
        if fmt == "pct":
            s = _fmt_pct(strat_m.get(key))
            tr = _fmt_pct(tr_m.get(key))
            pr = _fmt_pct(pr_m.get(key))
        else:
            s = _fmt_num(strat_m.get(key))
            tr = _fmt_num(tr_m.get(key))
            pr = _fmt_num(pr_m.get(key))
        hint_html = f"<span class='hint'>{hint}</span>" if hint else ""
        return (f"<tr><td class='metric-label'>{label}{hint_html}</td>"
                f"<td class='strategy-col'>{s}</td><td>{tr}</td><td>{pr}</td></tr>")

    def _section(title):
        return f"<tr class='oak-section'><td colspan='4'>{title}</td></tr>"

    table_html = f"""
    <table class="oak-metrics-table">
        <thead>
            <tr><th>Metric</th><th>Strategy (Net)</th><th>SMI Total Return</th><th>SMI Price Index</th></tr>
        </thead>
        <tbody>
            {_section("Return")}
            {_row("Total Return", "total_return")}
            {_row("Annualized Return (CAGR)", "cagr")}
            {_section("Risk")}
            {_row("Annualized Volatility", "vol_ann", hint="Std. dev. of daily returns × √252")}
            {_row("Downside Deviation", "downside_vol", hint="Volatility of negative returns only")}
            {_row("Maximum Drawdown", "max_drawdown", hint="Largest peak-to-trough loss")}
            {_section("Risk-Adjusted Performance")}
            {_row("Sharpe Ratio", "sharpe", "num", "(CAGR − Rf) / Volatility")}
            {_row("Sortino Ratio", "sortino", "num", "(CAGR − Rf) / Downside Vol")}
            {_row("Calmar Ratio", "calmar", "num", "CAGR / |Max DD|")}
            {_section("Tail Risk · Monthly")}
            {_row("Value at Risk (95%)", "var_95_monthly", hint="5th-percentile monthly return")}
            {_row("Expected Shortfall (95%)", "cvar_95_monthly", hint="Avg. return in worst 5% of months")}
            {_row("Worst Month", "worst_month")}
            {_section("Consistency")}
            {_row("Best Month", "best_month")}
            {_row("Positive Months", "pct_positive_months", hint="% of months with positive return")}
        </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown(
        f"<p style='color:{OAK_SAGE_DIM}; font-size:11px; margin-top:-8px;'>"
        f"Risk-free rate assumption: {risk_free_rate*100:.2f}% p.a. · "
        f"Adjust in sidebar to recalculate."
        "</p>",
        unsafe_allow_html=True
    )

    # ---- Strategy vs SMI TR benchmark metrics (4 KPI tiles) ----
    st.markdown("### Strategie vs. SMI Total Return")
    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("Alpha (Jensen, annualisiert)",
               _fmt_pct(bm_tr.get("alpha")),
               "Excess return adj. for beta")
    bc2.metric("Beta", _fmt_num(bm_tr.get("beta")),
               "Sensitivity to SMI TR")
    bc3.metric("Tracking Error",
               _fmt_pct(bm_tr.get("tracking_error")),
               "Std. dev. of excess returns")
    bc4.metric("Information Ratio",
               _fmt_num(bm_tr.get("information_ratio")),
               "Excess return / TE")

    bc5, bc6 = st.columns([1, 3])
    bc5.metric("Korrelation",
               _fmt_num(bm_tr.get("correlation")),
               f"R² = {_fmt_num(bm_tr.get('r_squared'))}")
    with bc6:
        if strat_m.get("dd_peak") and strat_m.get("dd_trough"):
            peak = pd.Timestamp(strat_m["dd_peak"]).strftime("%Y-%m-%d")
            trough = pd.Timestamp(strat_m["dd_trough"]).strftime("%Y-%m-%d")
            rec = pd.Timestamp(strat_m["dd_recovery"]).strftime("%Y-%m-%d") if strat_m.get("dd_recovery") else "not yet recovered"
            days = strat_m.get("dd_duration_days", 0)
            st.markdown(
                f"<div style='background:{OAK_GREEN_2}; padding:16px 20px; "
                f"border:1px solid {OAK_BORDER}; border-left:3px solid {OAK_RED}; "
                f"border-radius:9px; margin-top:0;'>"
                f"<div style='color:{OAK_SAGE}; font-size:10px; text-transform:uppercase; "
                f"letter-spacing:0.12em; font-weight:600;'>Strategy Max Drawdown Episode</div>"
                f"<div style='color:{OAK_CREAM}; font-family:Cormorant Garamond, serif; "
                f"font-size:22px; margin-top:6px;'>{_fmt_pct(strat_m['max_drawdown'])}</div>"
                f"<div style='color:{OAK_CREAM_DIM}; font-size:11px; margin-top:6px;'>"
                f"Peak: <strong style='color:{OAK_CREAM};'>{peak}</strong> · "
                f"Trough: <strong style='color:{OAK_CREAM};'>{trough}</strong> · "
                f"Recovery: <strong style='color:{OAK_CREAM};'>{rec}</strong> · "
                f"Duration: <strong style='color:{OAK_CREAM};'>{days} days</strong>"
                f"</div></div>",
                unsafe_allow_html=True
            )

    # ---- Drawdown Chart (Underwater) ----
    st.markdown("### Drawdown-Analyse")
    dd_strat = compute_drawdown(ts["total_value_net"]) * 100
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd_strat.index, y=dd_strat.values, name="Strategie (netto)",
        line=dict(color=OAK_GOLD, width=2),
        fill="tozeroy", fillcolor="rgba(201,169,97,0.2)",
    ))
    if not bench.empty:
        dd_tr = compute_drawdown(bench["smi_tr"]) * 100
        fig_dd.add_trace(go.Scatter(
            x=dd_tr.index, y=dd_tr.values, name="SMI Total Return",
            line=dict(color=OAK_SAGE, width=1.5, dash="dash"),
        ))
        dd_pr = compute_drawdown(bench["smi_price"]) * 100
        fig_dd.add_trace(go.Scatter(
            x=dd_pr.index, y=dd_pr.values, name="SMI Kursindex",
            line=dict(color=OAK_SAGE_DIM, width=1, dash="dot"),
        ))
    fig_dd = style_plotly(fig_dd, height=350)
    fig_dd.update_yaxes(title_text="Drawdown from Peak", ticksuffix="%")
    st.plotly_chart(fig_dd, use_container_width=True)

    # ---- Rolling Volatility Chart ----
    st.markdown("### Rollierende Volatilität (60-Tage-Fenster, annualisiert)")
    strat_ret = ts["total_value_net"].pct_change().dropna()
    roll_strat = strat_ret.rolling(60).std() * np.sqrt(252) * 100
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=roll_strat.index, y=roll_strat.values, name="Strategie (netto)",
        line=dict(color=OAK_GOLD, width=2),
    ))
    if not bench.empty:
        tr_ret = bench["smi_tr"].pct_change().dropna()
        roll_tr = tr_ret.rolling(60).std() * np.sqrt(252) * 100
        fig_vol.add_trace(go.Scatter(
            x=roll_tr.index, y=roll_tr.values, name="SMI Total Return",
            line=dict(color=OAK_SAGE, width=1.5, dash="dash"),
        ))
    fig_vol = style_plotly(fig_vol, height=320)
    fig_vol.update_yaxes(title_text="Annualized Volatility", ticksuffix="%")
    st.plotly_chart(fig_vol, use_container_width=True)

    # ---- Monthly Returns Heatmap ----
    st.markdown("### Monatsrenditen · Strategie (netto)")
    matrix = monthly_returns_matrix(ts["total_value_net"])
    if not matrix.empty:
        # Build heatmap with custom colorscale (red → cream → sage/green)
        z = matrix.values.astype(float) * 100  # to percent
        years_idx = matrix.index.astype(str).tolist()
        cols = matrix.columns.tolist()
        # Custom diverging colorscale
        colorscale = [
            [0.0, "#7A2A1F"],
            [0.25, "#B85042"],
            [0.5, OAK_GREEN_2],
            [0.75, "#7A8975"],
            [1.0, OAK_SAGE],
        ]
        # Use symmetric range so 0 is in the middle
        vmax = max(abs(np.nanmin(z)), abs(np.nanmax(z)))
        text = [[f"{v:+.1f}%" if not np.isnan(v) else "" for v in row] for row in z]
        fig_hm = go.Figure(data=go.Heatmap(
            z=z, x=cols, y=years_idx,
            colorscale=colorscale, zmid=0, zmin=-vmax, zmax=vmax,
            text=text, texttemplate="%{text}",
            textfont=dict(size=11, color=OAK_CREAM, family="Inter"),
            xgap=2, ygap=2,
            colorbar=dict(
                title=dict(text="Return %", font=dict(color=OAK_CREAM, size=11)),
                tickfont=dict(color=OAK_CREAM_DIM, size=10),
                outlinecolor=OAK_BORDER, outlinewidth=1,
                len=0.85, thickness=12,
            ),
            hovertemplate="%{y} · %{x}: <b>%{z:+.2f}%</b><extra></extra>",
        ))
        fig_hm = style_plotly(fig_hm, height=max(280, 38 * len(years_idx) + 80))
        fig_hm.update_xaxes(side="top", showgrid=False, ticks="")
        fig_hm.update_yaxes(showgrid=False, ticks="", autorange="reversed")
        st.plotly_chart(fig_hm, use_container_width=True)

    # ---- Yearly Returns Bar Chart with HWM ----
    if not fee_events_df.empty:
        st.markdown("### Jahresperformance & High Water Mark")
        yearly_net = ts["total_value_net"].resample("YE").last()
        yearly_ret = yearly_net.pct_change()
        # First-year return: compute from start
        first_year_ret = yearly_net.iloc[0] / initial_capital - 1
        yearly_ret.iloc[0] = first_year_ret

        years_list = yearly_net.index.year.tolist()
        rets_pct = (yearly_ret.values * 100).tolist()
        bar_colors = [OAK_SAGE if r >= 0 else OAK_RED for r in rets_pct]

        fig_yr = go.Figure()
        fig_yr.add_trace(go.Bar(
            x=years_list, y=rets_pct, marker=dict(color=bar_colors,
                                                   line=dict(color=OAK_GREEN_2, width=1)),
            name="Jahresrendite Strategie (netto)",
            text=[f"{r:+.1f}%" for r in rets_pct],
            textposition="outside",
            textfont=dict(color=OAK_CREAM, size=11),
        ))
        # Hurdle line for year 1
        fig_yr.add_hline(y=hwm_hurdle_pct * 100,
                         line=dict(color=OAK_GOLD, width=1.5, dash="dash"),
                         annotation_text=f"Year-1 Hurdle {hwm_hurdle_pct*100:.0f}%",
                         annotation_position="top right",
                         annotation_font=dict(color=OAK_GOLD, size=11))
        fig_yr.add_hline(y=0, line=dict(color=OAK_SAGE_DIM, width=1))
        fig_yr = style_plotly(fig_yr, height=380)
        fig_yr.update_xaxes(title_text="Year", dtick=1)
        fig_yr.update_yaxes(title_text="Annual Return (Net)", ticksuffix="%")
        st.plotly_chart(fig_yr, use_container_width=True)

    # ======================================================================
    # KALIBRIERUNG — Datenfenster-Wahl (Start-Sensitivität)
    # Beantwortet regelbasiert, welcher Backtest-Startpunkt verwendet werden
    # soll — nicht per Augenmass, sondern über zwei mechanische Tests:
    #   A) Anker-Extremität: liegt der Startpunkt selbst an einem lokalen
    #      Kurs-Extrem (Hoch oder Tief)? Das würde die daraus resultierenden
    #      rollierenden Fenster systematisch verzerren.
    #   B) Stabilität: bleibt der Regime-Befund (Δ-CAGR in Crash-Fenstern)
    #      über verschiedene Kandidaten-Startdaten stabil, oder kippt er?
    # Regel: frühester Kandidat, der (A) nicht extrem ist UND (B) im stabilen
    # Bereich liegt — maximiert die Anzahl nutzbarer Fenster, ohne Verzerrung.
    # ======================================================================
    st.markdown("---")
    st.markdown("## Kalibrierung — Datenfenster-Wahl (Start-Sensitivität)")
    st.markdown(
        "<p style='color:#A9B5A4;margin-top:-6px'>Der Backtest-Startpunkt wird "
        "hier selbst regelbasiert bestimmt, nicht per Augenmass — sonst wäre "
        "die Kalibrierung an ihrer eigenen Wurzel diskretionär. Zwei Tests: "
        "(A) liegt ein Kandidat-Startdatum an einem lokalen Kurs-Extrem? "
        "(B) bleibt der Regime-Befund stabil, egal welcher nicht-extreme "
        "Kandidat gewählt wird? Empfehlung = frühester Kandidat, der beide "
        "Tests besteht — maximiert die Fensterzahl, ohne Anker-Verzerrung.</p>",
        unsafe_allow_html=True)

    def _anchor_percentile(_btc, cand_date, lookback_m=18, lookahead_m=18):
        lo = pd.Timestamp(cand_date) - pd.DateOffset(months=lookback_m)
        hi = pd.Timestamp(cand_date) + pd.DateOffset(months=lookahead_m)
        seg = _btc[(_btc.index >= lo) & (_btc.index <= hi)]
        if seg.empty:
            return np.nan
        idx_le = _btc.index[_btc.index <= pd.Timestamp(cand_date)]
        if len(idx_le) == 0:
            return np.nan
        p = _btc.loc[idx_le[-1]]
        return float((seg < p).mean())

    @st.cache_data(ttl=3600, show_spinner=False)
    def compute_start_date_sensitivity(_prices, _divs, _btc, _fx, _weights, cap,
                                       candidates, allocs, band_width, win_years,
                                       step_months, dd_crash_threshold):
        """Für jeden Kandidaten-Startpunkt: (A) Anker-Perzentil, (B) Regime-
        Befund (Median Δ-CAGR und %-positiv in Fenstern mit schwerem
        BTC-Drawdown), auf den Daten AB diesem Startpunkt."""
        full_all = _prices.index
        rows = []
        for cand in candidates:
            anchor_pct = _anchor_percentile(_btc, cand)
            full = full_all[full_all >= pd.Timestamp(cand)]
            if len(full) < 400:
                rows.append({"start": cand, "anchor_pct": anchor_pct,
                            "n_windows": 0, "n_crash": 0,
                            "median_delta": np.nan, "pos_pct": np.nan})
                continue
            starts = pd.date_range(full[0], full[-1] - pd.DateOffset(years=win_years),
                                   freq=f"{step_months}MS")
            deltas, dds = [], []
            for alloc in allocs:
                target = alloc
                upper = min(alloc + band_width, 0.95)
                for s in starts:
                    e = s + pd.DateOffset(years=win_years)
                    w = full[(full >= s) & (full <= e)]
                    if len(w) < 300:
                        continue
                    try:
                        _ts, _, _ = run_strategy(
                            _prices.loc[w], _divs, _btc, _fx,
                            initial_capital=cap, weights=_weights,
                            initial_btc_pct=alloc, upper_threshold=upper,
                            target_btc_pct=target, rebalance_dates_set=set(),
                            dca_months=12, tx_cost_bps=10.0)
                        _bl = run_static_blend(_prices.loc[w], _divs, _btc, _fx,
                                               cap, _weights, alloc)
                    except Exception:
                        continue
                    if _ts.empty or _bl.empty:
                        continue
                    rm_s = risk_metrics(_ts["total_value"])
                    rm_b = risk_metrics(_bl["total_value"])
                    # KORREKTUR: _btc und w (aus _prices.index) überlappen
                    # nicht zwingend exakt (unabhängige Kalender) — reindex+
                    # ffill statt .loc[w], sonst KeyError bei fehlendem Datum.
                    seg = _btc.reindex(w).ffill().dropna()
                    if len(seg) < 2:
                        continue
                    dd = float((seg / seg.cummax() - 1.0).min())
                    deltas.append(rm_s["cagr"] - rm_b["cagr"])
                    dds.append(dd)
            _d = pd.Series(deltas); _dd = pd.Series(dds)
            _crash_mask = _dd <= -dd_crash_threshold
            _n_crash = int(_crash_mask.sum())
            _med = float(_d[_crash_mask].median()) if _n_crash else np.nan
            _pos = float((_d[_crash_mask] > 0).mean()) if _n_crash else np.nan
            rows.append({"start": cand, "anchor_pct": anchor_pct,
                        "n_windows": len(_d), "n_crash": _n_crash,
                        "median_delta": _med, "pos_pct": _pos})
        return pd.DataFrame(rows)

    dw1, dw2, dw3 = st.columns(3)
    with dw1:
        _dw_earliest = st.selectbox("Früheste Kandidatin", [2013, 2014, 2015], index=0,
                                    key="smi_dw_earliest")
    with dw2:
        _dw_dd = st.slider("Crash-Schwelle für den Regime-Test (%)", 20, 60, 40, 5,
                           key="smi_dw_dd") / 100.0
    with dw3:
        st.caption("")
        _dwgo = st.button("Datenfenster-Test starten", key="smi_dw_go")

    if _dwgo:
        st.session_state["smi_dw_has_run"] = True

    if st.session_state.get("smi_dw_has_run"):
        _dw_candidates = [f"{y}-01-01" for y in range(_dw_earliest, 2021)]
        with st.spinner("Teste Kandidaten-Startdaten (Anker-Extremität + "
                         "Regime-Stabilität)…"):
            dwres = compute_start_date_sensitivity(
                prices, divs, btc_series, fx, weights, initial_capital,
                _dw_candidates, [0.10, 0.20], 0.10, 3, 12, _dw_dd)

        if dwres.empty:
            st.warning("Zu wenig Daten für den Sensitivitätstest.")
        else:
            # KORREKTUR: ein Kandidat ohne auswertbares Anker-Perzentil (zu
            # wenig BTC-Historie vor diesem Datum, z.B. vor dem Beginn der
            # Yahoo-Finance-BTC-USD-Reihe ~09/2014) ist NICHT automatisch
            # "ok" — NaN < 0.15 und NaN > 0.85 werten beide als False, was
            # einen nicht testbaren Kandidaten fälschlich wie einen
            # bestandenen behandeln würde. Explizit als eigener Status.
            dwres["insufficient_data"] = dwres["anchor_pct"].isna()
            dwres["extreme"] = ((dwres["anchor_pct"] < 0.15) | (dwres["anchor_pct"] > 0.85)) & ~dwres["insufficient_data"]

            st.markdown("##### Schritt A — Anker-Extremität je Kandidat")
            _dispA = dwres.copy()
            _dispA["Startdatum"] = _dispA["start"]
            _dispA["Lokales Perzentil"] = _dispA["anchor_pct"].apply(
                lambda v: "keine Daten" if pd.isna(v) else f"{v*100:.1f}%")
            _dispA["Status"] = _dispA.apply(
                lambda r: "⚐ zu wenig Historie" if r["insufficient_data"]
                else ("⚑ extrem" if r["extreme"] else "ok"), axis=1)
            st.dataframe(_dispA[["Startdatum", "Lokales Perzentil", "Status"]],
                        use_container_width=True, hide_index=True)

            st.markdown("##### Schritt B — Stabilität des Regime-Befunds je Kandidat")
            figdw = go.Figure()
            figdw.add_trace(go.Scatter(
                x=dwres["start"], y=dwres["median_delta"] * 100, mode="markers+lines",
                marker=dict(size=10, color=[
                    (OAK_CREAM_DIM if ins else (OAK_BTC if e else OAK_GOLD))
                    for e, ins in zip(dwres["extreme"], dwres["insufficient_data"])]),
                line=dict(color=OAK_SAGE, dash="dot"), name="Median Δ-CAGR in Crash-Fenstern"))
            figdw.add_hline(y=0, line=dict(color=OAK_CREAM_DIM, dash="dot"))
            figdw.update_xaxes(title_text="Kandidat-Startdatum")
            figdw.update_yaxes(title_text="Median Δ-CAGR in Crash-Fenstern (pp)")
            figdw = style_plotly(figdw, height=360)
            st.plotly_chart(figdw, use_container_width=True)
            st.caption("Orange = nicht-extreme Kandidaten, Bitcoin-orange = an Schritt A "
                       "gescheitert, Cremeweiss = zu wenig Historie (Schritt A nicht "
                       "auswertbar). Flach über mehrere Kandidaten = stabil.")

            _stab_thresh = st.slider(
                "Stabilitäts-Schwelle (max. Sprung zwischen Nachbar-Kandidaten, pp)",
                1.0, 15.0, 5.0, 0.5, key="smi_dw_stabthresh",
                help="Vorab festgelegt, nicht nachträglich an ein gewünschtes "
                     "Ergebnis angepasst. Ein Kandidat gehört zum stabilen "
                     "Bereich nur, wenn sich der Regime-Befund zum nächsten "
                     "(chronologisch benachbarten, nicht ausgeschlossenen) "
                     "Kandidaten um höchstens diesen Wert unterscheidet.") / 1.0

            _eligible = dwres[~dwres["extreme"] & ~dwres["insufficient_data"]
                              & dwres["n_crash"].ge(3)].reset_index(drop=True)
            if _eligible.empty:
                st.error("⚠️ Kein Kandidat besteht Schritt A — Zeitraum oder "
                         "Crash-Schwelle anpassen.")
            else:
                # ECHTER Schritt-B-Filter: rückwärts vom jüngsten zulässigen
                # Kandidaten aus laufen und so lange in den "stabilen Block"
                # aufnehmen, wie der Sprung zum nächsten Nachbarn unter der
                # Schwelle bleibt. Bricht beim ersten (rückwärts gesehenen)
                # Sprung über der Schwelle ab — alles davor gehört NICHT zum
                # stabilen Bereich, auch wenn es Schritt A bestanden hat.
                _n = len(_eligible)
                _plateau_start_idx = _n - 1
                for i in range(_n - 1, 0, -1):
                    _jump = abs(_eligible.loc[i, "median_delta"]
                               - _eligible.loc[i - 1, "median_delta"]) * 100
                    if _jump <= _stab_thresh:
                        _plateau_start_idx = i - 1
                    else:
                        break
                _plateau = _eligible.iloc[_plateau_start_idx:]
                _excluded_unstable = _eligible.iloc[:_plateau_start_idx]
                _spread = _plateau["median_delta"].max() - _plateau["median_delta"].min()
                _rec = _plateau.iloc[0]

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("Empfohlenes Startdatum", _rec["start"])
                    st.caption("frühester Kandidat im stabilen Block")
                with r2:
                    st.metric("Nutzbare Fenster", int(_rec["n_windows"]))
                with r3:
                    st.metric("Streuung im stabilen Block", f"{_spread*100:.2f}pp")
                    st.caption("klein = Befund robust gegen Startdatum-Wahl")

                if not _excluded_unstable.empty:
                    st.warning(
                        f"⚠️ **{len(_excluded_unstable)} früherer Kandidat(en) "
                        f"({', '.join(_excluded_unstable['start'])}) bestehen zwar "
                        f"Schritt A, fallen aber bei Schritt B raus** — der Sprung "
                        f"zum nächsten Nachbarn überschreitet die "
                        f"{_stab_thresh:.1f}pp-Schwelle. Sie werden NICHT für die "
                        "Empfehlung verwendet, obwohl sie einzeln unauffällig "
                        "aussehen.")

                st.info(
                    f"**Regel angewendet:** {_rec['start']} ist der früheste Kandidat "
                    f"in einem ununterbrochenen Block aufeinanderfolgender Kandidaten "
                    f"bis zum jüngsten zulässigen Kandidaten, innerhalb dessen sich "
                    f"der Regime-Befund von Nachbar zu Nachbar um höchstens "
                    f"{_stab_thresh:.1f}pp unterscheidet. Für den weiteren Live-Test "
                    "diesen Wert als Backtest-Startdatum in der Sidebar übernehmen.")

        st.warning(
            "⚠️ **Provisorisch, solange mit synthetischen Testpfaden gerechnet "
            "wird.** Im Deployment mit echten Kursen automatisch neu bestimmt — "
            "diese Sektion sollte bei jeder grösseren Neukalibrierung erneut "
            "laufen, nicht nur einmalig.")

    # =====================================================================
    # Parameter Sensitivity Analysis (Heatmap)
    # =====================================================================
    # ======================================================================
    # ROBUSTHEIT — vereinfachtes Grid + Startdatum-Sensitivität
    #
    # Trick: die SMI-Engine ist GEBÜHRENUNABHÄNGIG (Fees werden nachgelagert
    # via apply_fees auf die Brutto-Kurve gelegt und beeinflussen weder die
    # BTC-Lots noch das Rebalancing). Also läuft die teure Engine nur einmal
    # je (Startallokation × Fenster); die vier Fee-Stufen werden danach quasi
    # gratis daraufgelegt. Das viertelt die Laufzeit.
    #
    # Folge daraus: der DCA-Anteil ist beim SMI per Konstruktion fee-unabhängig
    # — deshalb hier eine Balkengrafik statt einer Heatmap.
    # ======================================================================
    st.markdown("## Robustheit — Grid & Startdatum")
    st.markdown(
        "<p style='color:#A9B5A4;margin-top:-6px'>Ein einzelnes Startdatum ist "
        "keine Evidenz. Die Engine läuft über mehrere Startallokationen und viele "
        "rollierende Startzeitpunkte — ausgewiesen wird die Verteilung, nicht der "
        "Bestwert.</p>", unsafe_allow_html=True)

    @st.cache_data(ttl=3600, show_spinner=False)
    def compute_smi_robustness(_prices, _divs, _btc, _fx, _weights, cap,
                               allocs, fees, upper, target, dca_m, txbps,
                               win_years, step_months, cryst, hurdle_t, hurdle_r,
                               perf_rate):
        """Engine EINMAL je (alloc, window); Fees danach analytisch drauf."""
        full = _prices.index
        if len(full) < 400:
            return pd.DataFrame()
        starts = pd.date_range(full[0], full[-1] - pd.DateOffset(years=win_years),
                               freq=f"{step_months}MS")
        rows = []
        for alloc in allocs:
            for s in starts:
                e = s + pd.DateOffset(years=win_years)
                w = full[(full >= s) & (full <= e)]
                if len(w) < 300:
                    continue
                try:
                    _ts, _, _ = run_strategy(
                        _prices.loc[w], _divs, _btc, _fx,
                        initial_capital=cap, weights=_weights,
                        initial_btc_pct=alloc, upper_threshold=upper,
                        target_btc_pct=min(target, upper - 0.01),
                        rebalance_dates_set=set(), dca_months=dca_m,
                        tx_cost_bps=txbps)
                except Exception:
                    continue
                if _ts.empty:
                    continue
                _gross = _ts["total_value"]
                _att = _ts.attrs.get("attribution", {})
                _yrs = max((_gross.index[-1] - _gross.index[0]).days / 365.25, 1e-9)
                for f in fees:   # billig: nur die Fee-Schicht
                    _net, _, _, _ = apply_fees(
                        _gross, cap, mgmt_fee_annual=f, perf_fee_rate=perf_rate,
                        hwm_hurdle=hurdle_r, crystallization_freq=cryst,
                        hurdle_type=hurdle_t)
                    _cagr = (_net.iloc[-1] / cap) ** (1 / _yrs) - 1
                    rows.append({"alloc": alloc, "fee": f, "start": s,
                                 "net_cagr": _cagr,
                                 "dca_share": _att.get("dca_share", np.nan)})
        return pd.DataFrame(rows)

    _s1, _s2, _s3 = st.columns(3)
    with _s1:
        _sw = st.selectbox("Fensterlänge (Jahre)", [3, 5], index=0, key="smi_rb_win")
    with _s2:
        _sstep = st.selectbox("Fenster-Schritt", ["halbjährlich", "quartalsweise"],
                              index=0, key="smi_rb_step",
                              help="Quartalsweise ist gründlicher, dauert aber "
                                   "rund doppelt so lange.")
    with _s3:
        st.caption("")
        _sgo = st.button("Robustheitsanalyse starten", key="smi_rb_go")

    if _sgo:
        st.session_state["smi_rb_has_run"] = True

    if st.session_state.get("smi_rb_has_run"):
        _sallocs = [0.0, 0.05, 0.10, 0.20]
        _sfees = [0.0200, 0.0150, 0.0100, 0.0075]
        _sm = 6 if _sstep == "halbjährlich" else 3

        with st.spinner("Rechne Grid über alle rollierenden Fenster…"):
            sgrid = compute_smi_robustness(
                prices, divs, btc_series, fx, weights, initial_capital,
                _sallocs, _sfees, upper_threshold, target_btc_pct, dca_months,
                tx_cost_bps, _sw, _sm, crystallization_freq, hurdle_type,
                hwm_hurdle_pct, perf_fee_pct)

        if sgrid.empty:
            st.warning("Zu wenig überlappende Daten für die Fensteranalyse.")
        else:
            _nw = sgrid["start"].nunique()
            st.caption(
                f"{sgrid['alloc'].nunique() * _nw:,} Engine-Läufe · {_nw} rollierende "
                f"{_sw}-Jahres-Fenster · Gebühren nachgelagert aufgelegt (die Engine "
                f"ist gebührenunabhängig)")

            # ---- 1) Netto-CAGR: Startallokation × Fee ----------------------
            st.markdown("##### Netto-CAGR (Median über alle Fenster) — Startallokation × Management Fee")
            spiv = (sgrid.groupby(["alloc", "fee"])["net_cagr"].median().unstack() * 100)
            figs = go.Figure(data=go.Heatmap(
                z=spiv.values,
                x=[f"{f*100:.2f}%" for f in spiv.columns],
                y=[f"{a*100:.0f}%" for a in spiv.index],
                colorscale=[[0, OAK_GREEN_2], [0.5, OAK_SAGE], [1, OAK_GOLD]],
                text=[[f"{v:.1f}%" for v in r] for r in spiv.values],
                texttemplate="%{text}", showscale=False))
            figs.update_xaxes(title_text="Management Fee (p.a.)", type="category")
            figs.update_yaxes(title_text="Startallokation BTC", type="category")
            figs = style_plotly(figs, height=320)
            st.plotly_chart(figs, use_container_width=True)

            # ---- 2) DCA-Anteil (fee-unabhängig -> Balken statt Heatmap) ----
            st.markdown("##### DCA-Anteil am BTC-Gewinn (Median) — hält der Name, was er verspricht?")
            sd = sgrid.groupby("alloc")["dca_share"].median() * 100
            figd = go.Figure(go.Bar(
                x=[f"{a*100:.0f}%" for a in sd.index], y=sd.values,
                marker_color=[OAK_GOLD if v >= 30 else "#8C3A2B" for v in sd.values],
                text=[f"{v:.0f}%" for v in sd.values], textposition="outside"))
            figd.add_hline(y=30, line=dict(color=OAK_SAGE, dash="dot"),
                           annotation_text="30%-Schwelle",
                           annotation_font=dict(color=OAK_SAGE, size=10))
            figd.update_xaxes(title_text="Startallokation BTC", type="category")
            figd.update_yaxes(title_text="DCA-Anteil (%)")
            figd = style_plotly(figd, height=320)
            st.plotly_chart(figd, use_container_width=True)
            st.caption(
                "Der DCA-Anteil ist beim SMI **fee-unabhängig** (die Gebühren werden "
                "auf die Brutto-Kurve gelegt und treffen beide BTC-Lots gleich). "
                "Wichtig: er ist **invers zum Einstiegsglück** — ein tiefer Wert heisst "
                "vor allem, dass der Zeitraum für die Startallokation günstig lag.")

            # ---- 3) Verteilung + Streuung ---------------------------------
            st.markdown("##### Verteilung der Netto-CAGR je Startallokation")
            figb = go.Figure()
            for a in _sallocs:
                v = sgrid.loc[sgrid["alloc"] == a, "net_cagr"] * 100
                figb.add_trace(go.Box(y=v, name=f"{a*100:.0f}%", marker_color=OAK_GOLD,
                                      line_color=OAK_SAGE, boxmean=True))
            figb.update_xaxes(title_text="Startallokation BTC", type="category")
            figb.update_yaxes(title_text="Netto-CAGR (%)")
            figb = style_plotly(figb, height=360)
            figb.update_layout(showlegend=False)
            st.plotly_chart(figb, use_container_width=True)

            sdist = (sgrid.groupby("alloc")["net_cagr"]
                     .agg(Minimum="min", P25=lambda s: s.quantile(.25), Median="median",
                          P75=lambda s: s.quantile(.75), Maximum="max") * 100).round(2)
            sdist["Streuung"] = (sdist["Maximum"] - sdist["Minimum"]).round(2)
            sdist.index = [f"{a*100:.0f}%" for a in sdist.index]
            sdist.index.name = "Startallokation"
            st.dataframe(sdist.style.format("{:.2f}%"), use_container_width=True)
            st.caption(
                "⚠️ **Das Minimum ist kein Risikomass** — die Datenreihe enthält kein "
                "3-Jahres-Fenster mit einem Bitcoin-Kollaps ohne Erholung. Das "
                "belastbare Signal ist die **Streuung**: sie misst, wie stark das "
                "Ergebnis vom Einstiegszeitpunkt abhängt.")

            # ---- 4) Worst-Entry -------------------------------------------
            st.markdown("##### Worst-Entry — der Investor mit dem schlechtesten Einstieg")
            _cur = min(_sallocs, key=lambda a: abs(a - initial_btc_pct))
            _sg = sgrid[(sgrid["alloc"] == _cur)
                        & (np.isclose(sgrid["fee"], mgmt_fee_pct))]
            if _sg.empty:
                _sg = sgrid[sgrid["alloc"] == _cur]
            if not _sg.empty:
                _wst = _sg.loc[_sg["net_cagr"].idxmin()]
                w1, w2, w3, w4 = st.columns(4)
                with w1:
                    st.metric("Schlechtestes Fenster", f"{_wst['net_cagr']*100:.1f}% p.a.")
                    st.caption(f"Einstieg {_wst['start']:%b %Y}")
                with w2:
                    _wd = _wst["dca_share"]
                    st.metric("DCA-Anteil dort",
                              "n/a" if _wd != _wd else f"{_wd*100:.0f}%")
                    st.caption("Mechanismus im Stressfall")
                with w3:
                    st.metric("Median", f"{_sg['net_cagr'].median()*100:.1f}% p.a.")
                    st.caption("mittleres Fenster")
                with w4:
                    _sp = (_sg["net_cagr"].max() - _sg["net_cagr"].min()) * 100
                    st.metric("Streuung", f"{_sp:.0f} pp")
                    st.caption("Max − Min über alle Fenster")
                st.caption(
                    f"Bei {_cur*100:.0f}% Startallokation und {mgmt_fee_pct*100:.2f}% Fee. "
                    "Je schlechter der Einstieg, desto wichtiger wird der "
                    "dividendenfinanzierte DCA — er kauft antizyklisch nach, während "
                    "der Aktienkern unangetastet weiterläuft.")

            st.session_state["smi_rb_dist"] = sdist

    # ======================================================================
    # KALIBRIERUNG — Schwellenprüfung-Frequenz (Bitcoin-Band)
    # Beantwortet konkret: unter welchen Bedingungen (Startallokation) macht
    # monatliche vs. quartalsweise vs. halbjährliche Prüfung Sinn? Dieselbe
    # Rolling-Window-Methodik wie oben, aber threshold_check_dates_set als
    # zusätzliche Grid-Dimension. Bandbreite (upper/target) bleibt auf dem
    # aktuellen Sidebar-Wert fixiert, um die Fragestellung fokussiert zu
    # halten — das ist NICHT dieselbe Frage wie "welche Startallokation".
    # ======================================================================
    st.markdown("---")
    st.markdown("## Kalibrierung — Schwellenprüfung-Frequenz")
    st.markdown(
        "<p style='color:#A9B5A4;margin-top:-6px'>Beantwortet konkret: unter "
        "welchen Bedingungen macht eine seltenere Prüfung des Bitcoin-Bands "
        "Sinn? Dieselbe Rolling-Window-Methodik wie oben, jetzt mit der "
        "Prüf-Frequenz als zusätzlicher Dimension. Bandbreite bleibt auf dem "
        "aktuellen Sidebar-Wert fixiert — das ist eine andere Frage als "
        "\u201ewelche Startallokation\u201c oben.</p>", unsafe_allow_html=True)

    @st.cache_data(ttl=3600, show_spinner=False)
    def compute_smi_threshold_freq_grid(_prices, _divs, _btc, _fx, _weights, cap,
                                        allocs, freqs, upper, target, dca_m, txbps,
                                        win_years, step_months, fee):
        """Engine je (Frequenz, Startallokation, Fenster). Erfasst zusätzlich
        die Überschreitung über der oberen Schwelle bei Auslösung — das ist
        die Kennzahl, die die Prüf-Frequenz direkt sichtbar macht."""
        full = _prices.index
        if len(full) < 400:
            return pd.DataFrame()
        starts = pd.date_range(full[0], full[-1] - pd.DateOffset(years=win_years),
                               freq=f"{step_months}MS")
        rows = []
        for freq_label in freqs:
            for alloc in allocs:
                for s in starts:
                    e = s + pd.DateOffset(years=win_years)
                    w = full[(full >= s) & (full <= e)]
                    if len(w) < 300:
                        continue
                    thr_dates = (None if freq_label == "Monatlich (Standard)"
                                else get_rebalance_dates(w, freq_label))
                    try:
                        _ts, _, _evts = run_strategy(
                            _prices.loc[w], _divs, _btc, _fx,
                            initial_capital=cap, weights=_weights,
                            initial_btc_pct=alloc, upper_threshold=upper,
                            target_btc_pct=min(target, upper - 0.01),
                            rebalance_dates_set=set(), dca_months=dca_m,
                            tx_cost_bps=txbps, threshold_check_dates_set=thr_dates)
                    except Exception:
                        continue
                    if _ts.empty:
                        continue
                    _gross = _ts["total_value"]
                    _att = _ts.attrs.get("attribution", {})
                    _yrs = max((_gross.index[-1] - _gross.index[0]).days / 365.25, 1e-9)
                    _net, _, _, _ = apply_fees(
                        _gross, cap, mgmt_fee_annual=fee, perf_fee_rate=perf_fee_pct,
                        hwm_hurdle=hwm_hurdle_pct, crystallization_freq=crystallization_freq,
                        hurdle_type=hurdle_type)
                    _cagr = (_net.iloc[-1] / cap) ** (1 / _yrs) - 1
                    _n_events = len(_evts) if _evts is not None else 0
                    _avg_overshoot = (float((_evts["btc_pct_before"] - upper).mean())
                                      if _n_events else 0.0)
                    _max_overshoot = (float((_evts["btc_pct_before"] - upper).max())
                                      if _n_events else 0.0)
                    rows.append({"freq": freq_label, "alloc": alloc, "start": s,
                                 "net_cagr": _cagr,
                                 "dca_share": _att.get("dca_share", np.nan),
                                 "n_events": _n_events,
                                 "avg_overshoot": _avg_overshoot,
                                 "max_overshoot": _max_overshoot})
        return pd.DataFrame(rows)

    tf1, tf2, tf3 = st.columns(3)
    with tf1:
        _tfw = st.selectbox("Fensterlänge (Jahre)", [3, 5], index=0, key="smi_tf_win")
    with tf2:
        _tfstep = st.selectbox("Fenster-Schritt", ["halbjährlich", "quartalsweise"],
                               index=0, key="smi_tf_step")
    with tf3:
        st.caption("")
        _tfgo = st.button("Frequenz-Kalibrierung starten", key="smi_tf_go")

    if _tfgo:
        st.session_state["smi_tf_has_run"] = True

    if st.session_state.get("smi_tf_has_run"):
        _tf_allocs = [0.05, 0.10, 0.15, 0.20]
        _tf_freqs = ["Monatlich (Standard)", "Quartalsweise", "Halbjährlich"]
        _tf_sm = 6 if _tfstep == "halbjährlich" else 3

        with st.spinner("Rechne Grid über Frequenz × Startallokation × rollierende Fenster… "
                         "(mehr Läufe als oben, kann länger dauern)"):
            tfgrid = compute_smi_threshold_freq_grid(
                prices, divs, btc_series, fx, weights, initial_capital,
                _tf_allocs, _tf_freqs, upper_threshold, target_btc_pct, dca_months,
                tx_cost_bps, _tfw, _tf_sm, mgmt_fee_pct)

        if tfgrid.empty:
            st.warning("Zu wenig überlappende Daten für die Fensteranalyse.")
        else:
            _tf_nw = tfgrid["start"].nunique()
            st.caption(f"{len(tfgrid):,} Engine-Läufe · {_tf_nw} rollierende "
                       f"{_tfw}-Jahres-Fenster je Kombination · Bandbreite "
                       f"{upper_threshold*100:.0f}% / {target_btc_pct*100:.0f}% "
                       "(aktueller Sidebar-Wert)")

            st.markdown("##### Vergleichstabelle — Median über alle Fenster")
            _summary = (tfgrid.groupby(["freq", "alloc"]).agg(
                Median_CAGR=("net_cagr", "median"),
                P5_CAGR=("net_cagr", lambda x: x.quantile(.05)),
                Median_DCA=("dca_share", "median"),
                Overshoot_avg=("avg_overshoot", "median"),
                Overshoot_max=("max_overshoot", "max"),
                Events_pro_Fenster=("n_events", "mean"),
            ).reset_index())
            _disp = _summary.copy()
            _disp["Startallokation"] = (_disp["alloc"] * 100).round(0).astype(int).astype(str) + "%"
            _disp["Median-CAGR"] = (_disp["Median_CAGR"] * 100).round(2).astype(str) + "%"
            _disp["P5-CAGR"] = (_disp["P5_CAGR"] * 100).round(2).astype(str) + "%"
            _disp["DCA-Anteil"] = (_disp["Median_DCA"] * 100).round(0).astype(str) + "%"
            _disp["Ø Überschreitung"] = (_disp["Overshoot_avg"] * 100).round(2).astype(str) + "pp"
            _disp["Max Überschreitung"] = (_disp["Overshoot_max"] * 100).round(2).astype(str) + "pp"
            _disp["Events/Fenster"] = _disp["Events_pro_Fenster"].round(2)
            _disp = _disp.rename(columns={"freq": "Frequenz"})
            st.dataframe(_disp[["Frequenz", "Startallokation", "Median-CAGR", "P5-CAGR",
                                "DCA-Anteil", "Ø Überschreitung", "Max Überschreitung",
                                "Events/Fenster"]],
                        use_container_width=True, hide_index=True)

            st.markdown("##### Median-CAGR nach Frequenz und Startallokation")
            _piv = _summary.pivot(index="alloc", columns="freq", values="Median_CAGR") * 100
            _piv = _piv[[f for f in _tf_freqs if f in _piv.columns]]
            figtf = go.Figure()
            _colors = {"Monatlich (Standard)": OAK_GOLD, "Quartalsweise": OAK_SAGE,
                      "Halbjährlich": OAK_BTC}
            for fcol in _piv.columns:
                figtf.add_trace(go.Bar(name=fcol, x=[f"{a*100:.0f}%" for a in _piv.index],
                                       y=_piv[fcol].values,
                                       marker_color=_colors.get(fcol, OAK_SAGE)))
            figtf.update_layout(barmode="group")
            figtf.update_xaxes(title_text="Startallokation BTC", type="category")
            figtf.update_yaxes(title_text="Median Netto-CAGR (%)")
            figtf = style_plotly(figtf, height=380)
            st.plotly_chart(figtf, use_container_width=True)

            st.caption(
                "**Lesehilfe:** Wenn die CAGR-Balken je Startallokation nah beieinander "
                "liegen, macht die Frequenz für die Rendite kaum einen Unterschied — "
                "dann entscheidet die Überschreitungs-Spalte (Konzentrationsrisiko "
                "zwischen den Prüfterminen). Ein systematischer CAGR-Vorteil einer "
                "Frequenz über ALLE Startallokationen hinweg wäre ein Hinweis auf "
                "Overfitting an den Testzeitraum, kein robuster Befund.")

    # ======================================================================
    # KALIBRIERUNG — Risiko/Rendite-Profil (Alpha-Test)
    # Beantwortet: bringt der Mechanismus (DCA + Schwellen-Rebalancing) einen
    # echten Mehrwert gegenüber einer simplen, unverwalteten Static-Blend-
    # Position mit DERSELBEN Startallokation? Trennt "mehr Rendite durch mehr
    # Bitcoin-Beta" von "mehr Rendite durch den Mechanismus selbst (Alpha)".
    # Erweitert das Allokations-Raster zusätzlich in die höhere Risikozone.
    # ======================================================================
    st.markdown("---")
    st.markdown("## Kalibrierung — Risiko/Rendite-Profil (Alpha-Test)")
    st.markdown(
        "<p style='color:#A9B5A4;margin-top:-6px'>Eine höhere Startallokation "
        "erzeugt fast immer eine höhere Rendite in einem Bitcoin-Bullenfenster "
        "— das ist <strong>Beta</strong> (mehr Marktrisiko), kein Verdienst des "
        "Mechanismus. Diese Sektion isoliert die eigentliche Alpha-Frage: schlägt "
        "die Strategie (DCA + Schwellen-Rebalancing) eine simple, unverwaltete "
        "Static-Blend-Position mit <em>derselben</em> Startallokation — bei "
        "identischem Bitcoin-Exposure am Tag 1? Nur P1 kann diesen Test sauber "
        "liefern (kein at-par-Sleeve wie bei RE/BTC oder Private Debt/BTC, "
        "Vol/Sharpe/MaxDD sind hier real).</p>", unsafe_allow_html=True)

    def _window_btc_regime(_btc, w):
        """Charakterisiert das Marktregime des Fensters an seinem EIGENEN
        Bitcoin-Verlauf — kein handverlesenes Bär/Bulle-Label, sondern direkt
        gemessen: Gesamtrendite und Peak-to-Trough-Drawdown innerhalb des
        Fensters.

        KORREKTUR: _btc und die Aktienkalender (w, aus _prices.index) sind
        zwei UNABHÄNGIG geladene Serien (Bitcoin handelt 24/7, SIX hat eigene
        Feiertage) — sie überlappen nicht zwingend exakt. Ein direktes
        seg = _btc.loc[w] wirft KeyError, sobald ein Datum in w in _btc fehlt.
        Reindex+ffill nutzt dieselbe "letzter verfügbarer Kurs"-Konvention,
        die auch die Haupt-Engine (get_btc_price) verwendet.
        """
        seg = _btc.reindex(w).ffill().dropna()
        if len(seg) < 2 or seg.iloc[0] <= 0:
            return np.nan, np.nan
        ret = float(seg.iloc[-1] / seg.iloc[0] - 1.0)
        running_max = seg.cummax()
        dd = float((seg / running_max - 1.0).min())
        return ret, dd

    @st.cache_data(ttl=3600, show_spinner=False)
    def compute_smi_alpha_grid(_prices, _divs, _btc, _fx, _weights, cap,
                               allocs, band_width, dca_m, txbps,
                               win_years, step_months, fee):
        """Strategie vs. Static-Blend, EIN Engine-Lauf je (alloc, window) für
        jede Seite. Band skaliert mit der Allokation (target = alloc, upper =
        alloc + band_width), damit auch hohe Allokationen ein sinnvolles Band
        haben statt sofort über einer fixen Schwelle zu liegen. Erfasst
        zusätzlich die REGIME-Charakteristik jedes Fensters (eigene
        Bitcoin-Rendite/-Drawdown), um zu zeigen, UNTER WELCHEN BEDINGUNGEN
        der Mechanismus eine simple Static-Blend-Position schlägt."""
        full = _prices.index
        if len(full) < 400:
            return pd.DataFrame()
        starts = pd.date_range(full[0], full[-1] - pd.DateOffset(years=win_years),
                               freq=f"{step_months}MS")
        rows = []
        for alloc in allocs:
            target = alloc
            upper = min(alloc + band_width, 0.95)
            for s in starts:
                e = s + pd.DateOffset(years=win_years)
                w = full[(full >= s) & (full <= e)]
                if len(w) < 300:
                    continue
                try:
                    _ts, _, _ = run_strategy(
                        _prices.loc[w], _divs, _btc, _fx,
                        initial_capital=cap, weights=_weights,
                        initial_btc_pct=alloc, upper_threshold=upper,
                        target_btc_pct=target, rebalance_dates_set=set(),
                        dca_months=dca_m, tx_cost_bps=txbps)
                    _bl = run_static_blend(_prices.loc[w], _divs, _btc, _fx,
                                           cap, _weights, alloc)
                except Exception:
                    continue
                if _ts.empty or _bl.empty:
                    continue
                _net, _, _, _ = apply_fees(
                    _ts["total_value"], cap, mgmt_fee_annual=fee, perf_fee_rate=0.0,
                    hwm_hurdle=0.05, crystallization_freq="Quarterly", hurdle_type="Hard Hurdle")
                _rm_strat = risk_metrics(_net)
                _rm_bl = risk_metrics(_bl["total_value"])
                _avg_btc_pct = float(_ts["btc_pct"].mean())
                _w_ret, _w_dd = _window_btc_regime(_btc, w)
                rows.append({
                    "alloc": alloc, "start": s,
                    "strat_cagr": _rm_strat["cagr"], "strat_vol": _rm_strat["vol"],
                    "strat_sharpe": _rm_strat["sharpe"], "strat_maxdd": _rm_strat["max_dd"],
                    "bl_cagr": _rm_bl["cagr"], "bl_vol": _rm_bl["vol"],
                    "bl_sharpe": _rm_bl["sharpe"], "bl_maxdd": _rm_bl["max_dd"],
                    "avg_realized_btc_pct": _avg_btc_pct,
                    "window_btc_return": _w_ret, "window_btc_maxdd": _w_dd,
                })
        return pd.DataFrame(rows)

    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        _acw = st.selectbox("Fensterlänge (Jahre)", [3, 5], index=0, key="smi_ac_win")
    with ac2:
        _acstep = st.selectbox("Fenster-Schritt", ["halbjährlich", "quartalsweise"],
                               index=0, key="smi_ac_step")
    with ac3:
        st.caption("")
        _acgo = st.button("Alpha-Test starten", key="smi_ac_go")

    if _acgo:
        st.session_state["smi_ac_has_run"] = True

    if st.session_state.get("smi_ac_has_run"):
        _ac_allocs = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
        _ac_sm = 6 if _acstep == "halbjährlich" else 3

        with st.spinner("Rechne Strategie vs. Static-Blend über alle Fenster… "
                         "(2 Engine-Läufe je Kombination)"):
            agrid = compute_smi_alpha_grid(
                prices, divs, btc_series, fx, weights, initial_capital,
                _ac_allocs, 0.10, dca_months, tx_cost_bps, _acw, _ac_sm, mgmt_fee_pct)

        if agrid.empty:
            st.warning("Zu wenig überlappende Daten für die Fensteranalyse.")
        else:
            _an = agrid["start"].nunique()
            st.caption(f"{len(agrid):,} Fenster-Kombinationen ({_an} rollierende "
                       f"{_acw}-Jahres-Fenster) · Band = Startallokation bis "
                       "+10pp · Statische Vergleichsposition: gleiche Startallokation, "
                       "Bitcoin nie verkauft/nachgekauft, Aktien-Dividenden normal "
                       "reinvestiert, kein Rebalancing")

            summ = agrid.groupby("alloc").agg(
                Strat_CAGR=("strat_cagr", "median"), Static_CAGR=("bl_cagr", "median"),
                Strat_Vol=("strat_vol", "median"), Static_Vol=("bl_vol", "median"),
                Strat_Sharpe=("strat_sharpe", "median"), Static_Sharpe=("bl_sharpe", "median"),
                Strat_MaxDD=("strat_maxdd", "median"), Static_MaxDD=("bl_maxdd", "median"),
                Avg_BTC_Quote=("avg_realized_btc_pct", "median"),
            ).reset_index()
            summ["Delta_CAGR"] = summ["Strat_CAGR"] - summ["Static_CAGR"]
            summ["Delta_Sharpe"] = summ["Strat_Sharpe"] - summ["Static_Sharpe"]

            st.markdown("##### Strategie vs. Static-Blend — Median über alle Fenster")
            _disp = summ.copy()
            _disp["Startallokation"] = (_disp["alloc"] * 100).round(0).astype(int).astype(str) + "%"
            _disp["Ø realisierte BTC-Quote"] = (_disp["Avg_BTC_Quote"] * 100).round(1).astype(str) + "%"
            for c1, c2, lbl in [("Strat_CAGR", "Static_CAGR", "CAGR"),
                                 ("Strat_Vol", "Static_Vol", "Vol"),
                                 ("Strat_Sharpe", "Static_Sharpe", "Sharpe"),
                                 ("Strat_MaxDD", "Static_MaxDD", "MaxDD")]:
                if lbl == "Sharpe":
                    _disp[f"Strategie {lbl}"] = _disp[c1].round(2)
                    _disp[f"Static {lbl}"] = _disp[c2].round(2)
                else:
                    _disp[f"Strategie {lbl}"] = (_disp[c1] * 100).round(2).astype(str) + "%"
                    _disp[f"Static {lbl}"] = (_disp[c2] * 100).round(2).astype(str) + "%"
            _disp["Δ CAGR (Alpha-Signal)"] = (_disp["Delta_CAGR"] * 100).round(2).astype(str) + "pp"
            _disp["Δ Sharpe"] = _disp["Delta_Sharpe"].round(2)
            st.dataframe(_disp[["Startallokation", "Ø realisierte BTC-Quote",
                                "Strategie CAGR", "Static CAGR", "Δ CAGR (Alpha-Signal)",
                                "Strategie Vol", "Static Vol",
                                "Strategie Sharpe", "Static Sharpe", "Δ Sharpe",
                                "Strategie MaxDD", "Static MaxDD"]],
                        use_container_width=True, hide_index=True)

            st.markdown("##### Risiko/Rendite-Frontier — Strategie vs. Static-Blend")
            figac = go.Figure()
            figac.add_trace(go.Scatter(
                x=summ["Strat_Vol"] * 100, y=summ["Strat_CAGR"] * 100, mode="markers+lines+text",
                name="Strategie (DCA + Schwellen-Rebalancing)",
                text=[f"{a*100:.0f}%" for a in summ["alloc"]], textposition="top center",
                marker=dict(size=12, color=OAK_GOLD), line=dict(color=OAK_GOLD, dash="dot")))
            figac.add_trace(go.Scatter(
                x=summ["Static_Vol"] * 100, y=summ["Static_CAGR"] * 100, mode="markers+lines+text",
                name="Static-Blend (unverwaltet)",
                text=[f"{a*100:.0f}%" for a in summ["alloc"]], textposition="bottom center",
                marker=dict(size=12, color=OAK_SAGE), line=dict(color=OAK_SAGE, dash="dot")))
            figac.update_xaxes(title_text="Annualisierte Volatilität (%)")
            figac.update_yaxes(title_text="Median Netto-CAGR (%)")
            figac = style_plotly(figac, height=440)
            st.plotly_chart(figac, use_container_width=True)
            st.caption(
                "**Lesehilfe:** Liegt die Gold-Linie (Strategie) bei GLEICHER Vola "
                "über der Salbei-Linie (Static-Blend), ist das echtes Alpha — der "
                "Mechanismus bringt bei identischem Risiko mehr Rendite. Liegen die "
                "Linien praktisch übereinander, kommt jede Mehrrendite ausschliesslich "
                "aus höherer Startallokation (Beta), nicht aus dem Mechanismus. Die "
                "Spalte 'Ø realisierte BTC-Quote' zeigt, ob die Strategie über die Zeit "
                "strukturell mehr Bitcoin trägt als die Startallokation vermuten lässt "
                "(DCA baut kontinuierlich zu) — ein fairer Vergleich muss das einordnen.")

            st.markdown("##### Unter welchen Bedingungen gewinnt der Mechanismus? — Regime-Test")
            st.markdown(
                "<p style='color:#A9B5A4;margin-top:-6px'>Keine handverlesenen "
                "Bär-/Bullen-Label — das Regime jedes Fensters wird direkt an "
                "dessen EIGENEM Bitcoin-Verlauf gemessen (Drawdown, Gesamtrendite "
                "innerhalb des Fensters) und mit dem Delta zwischen Strategie und "
                "Static-Blend korreliert.</p>", unsafe_allow_html=True)

            agrid["delta_cagr"] = agrid["strat_cagr"] - agrid["bl_cagr"]
            _corr_dd = agrid["delta_cagr"].corr(agrid["window_btc_maxdd"])
            _corr_ret = agrid["delta_cagr"].corr(agrid["window_btc_return"])

            figreg = go.Figure()
            figreg.add_trace(go.Scatter(
                x=agrid["window_btc_maxdd"] * 100, y=agrid["delta_cagr"] * 100,
                mode="markers",
                marker=dict(size=7, color=agrid["alloc"], colorscale=[[0, OAK_SAGE], [1, OAK_GOLD]],
                           showscale=True, colorbar=dict(title="Alloc.")),
                name="Fenster"))
            figreg.add_hline(y=0, line=dict(color=OAK_CREAM_DIM, dash="dot"))
            figreg.update_xaxes(title_text="Bitcoin Max-Drawdown IM Fenster (%)")
            figreg.update_yaxes(title_text="Δ CAGR — Strategie minus Static-Blend (pp)")
            figreg = style_plotly(figreg, height=420)
            st.plotly_chart(figreg, use_container_width=True)
            st.caption(
                f"Korrelation Δ-CAGR ↔ Fenster-Drawdown: {_corr_dd:+.2f} · "
                f"Δ-CAGR ↔ Fenster-Gesamtrendite: {_corr_ret:+.2f}. "
                "Punkte rechts der Null-Linie oben = Mechanismus gewinnt; "
                "Punkte unten = Static-Blend gewinnt.")

            st.markdown("###### Anteil Fenster mit Mechanismus-Vorteil, nach Drawdown-Schwere")
            _bins = [-1.01, -0.6, -0.4, -0.2, -0.05, 0.01]
            _labels = ["≤ −60%", "−60% bis −40%", "−40% bis −20%", "−20% bis −5%", "> −5%"]
            agrid["dd_bucket"] = pd.cut(agrid["window_btc_maxdd"], bins=_bins, labels=_labels)
            _regime_tbl = agrid.groupby("dd_bucket", observed=True).agg(
                Median_Delta_CAGR=("delta_cagr", "median"),
                Anteil_positiv=("delta_cagr", lambda x: (x > 0).mean()),
                n=("delta_cagr", "size"),
            ).reset_index()
            _regime_tbl["Fenster-Drawdown"] = _regime_tbl["dd_bucket"].astype(str)
            _regime_tbl["Median Δ-CAGR"] = (_regime_tbl["Median_Delta_CAGR"] * 100).round(2).astype(str) + "pp"
            _regime_tbl["Anteil Fenster mit Vorteil"] = (_regime_tbl["Anteil_positiv"] * 100).round(0).astype(str) + "%"
            _regime_tbl["Anzahl Fenster"] = _regime_tbl["n"]
            st.dataframe(_regime_tbl[["Fenster-Drawdown", "Median Δ-CAGR",
                                      "Anteil Fenster mit Vorteil", "Anzahl Fenster"]],
                        use_container_width=True, hide_index=True)
            st.caption(
                "**Ökonomische Lesart — das ist Versicherungslogik:** in ruhigen/rein "
                "bullischen Fenstern kostet der Mechanismus eine kleine Prämie (die "
                "Gewinnmitnahme verpasst etwas Fortsetzung der Rally). In Fenstern mit "
                "einem echten Bitcoin-Crash gewinnt er häufiger — die Bandlogik hat "
                "rechtzeitig verkauft bzw. der DCA hat günstiger nachgekauft. Das "
                "Modell macht ökonomisch dort Sinn, wo Crash-Risiko real eingepreist "
                "werden soll — nicht als genereller Rendite-Booster über Buy-and-Hold.")

            st.warning(
                "⚠️ **Provisorisch, solange mit synthetischen Testpfaden gerechnet "
                "wird**, und Einzelrealisierung eines Bitcoin-Pfads — kein Ersatz für "
                "die Auswertung mit echten Kursen im Deployment. Höhere Allokationen "
                "(30–40%) sind hier bewusst zur Exploration eingeschlossen; das sind "
                "keine Empfehlungen, sondern Datenpunkte für die Positionierungs-"
                "Entscheidung.")

    # ======================================================================
    # KALIBRIERUNG — Optimales Risiko/Rendite-Profil (Sharpe/Calmar-Grid)
    # Andere Zielfunktion als der Alpha-Test oben: nicht "schlägt der
    # Mechanismus Buy-and-Hold" (beantwortet), sondern "welche Kombination
    # aus Startallokation × Bandbreite × DCA-Fenster liefert das beste
    # Rendite-pro-Risiko-Verhältnis, ohne das Upside künstlich zu kappen".
    # Sharpe/Calmar statt fixer Downside-Constraints, weil das Risiko und
    # Rendite gleichzeitig gewichtet statt eine willkürliche Präferenz
    # festzulegen. Rebalancing-Frequenz, Gewichtung, Kosten und Gebühren
    # bleiben fixiert, um die drei eigentlichen Hebel isoliert zu testen.
    # ======================================================================
    st.markdown("---")
    st.markdown("## Kalibrierung — Optimales Risiko/Rendite-Profil")
    st.markdown(
        "<p style='color:#A9B5A4;margin-top:-6px'>Zielfunktion: höchste "
        "risikoadjustierte Rendite (Sharpe/Calmar), NICHT höchste Rendite "
        "und NICHT primär Downside-Schutz. Rastert Startallokation × "
        "Bandbreite × Ernte-/DCA-Fenster; Schwellenprüfung-Frequenz, "
        "Aktien-Rebalancing, Gewichtung, Kosten und Gebühren bleiben auf "
        "dem aktuellen Sidebar-Wert fixiert, um die drei eigentlichen Hebel "
        "isoliert zu testen.</p>", unsafe_allow_html=True)

    @st.cache_data(ttl=3600, show_spinner=False)
    def compute_smi_sharpe_grid(_prices, _divs, _btc, _fx, _weights, cap,
                                allocs, widths, dca_opts, txbps,
                                win_years, step_months, max_dd_ceiling):
        full = _prices.index
        if len(full) < 400:
            return pd.DataFrame()
        starts = pd.date_range(full[0], full[-1] - pd.DateOffset(years=win_years),
                               freq=f"{step_months}MS")
        rows = []
        for alloc in allocs:
            for width in widths:
                target = alloc
                upper = min(alloc + width, 0.95)
                for dca_m in dca_opts:
                    for s in starts:
                        e = s + pd.DateOffset(years=win_years)
                        w = full[(full >= s) & (full <= e)]
                        if len(w) < 300:
                            continue
                        try:
                            _ts, _, _ = run_strategy(
                                _prices.loc[w], _divs, _btc, _fx,
                                initial_capital=cap, weights=_weights,
                                initial_btc_pct=alloc, upper_threshold=upper,
                                target_btc_pct=target, rebalance_dates_set=set(),
                                dca_months=dca_m, tx_cost_bps=txbps)
                        except Exception:
                            continue
                        if _ts.empty:
                            continue
                        _rm = risk_metrics(_ts["total_value"])
                        _att = _ts.attrs.get("attribution", {})
                        rows.append({
                            "alloc": alloc, "width": width, "dca_m": dca_m, "start": s,
                            "cagr": _rm["cagr"], "vol": _rm["vol"],
                            "sharpe": _rm["sharpe"], "max_dd": _rm["max_dd"],
                            "calmar": _rm["calmar"], "dca_share": _att.get("dca_share", np.nan),
                        })
        return pd.DataFrame(rows)

    sh1, sh2, sh3, sh4 = st.columns(4)
    with sh1:
        _shw = st.selectbox("Fensterlänge (Jahre)", [3, 5], index=0, key="smi_sh_win")
    with sh2:
        _shstep = st.selectbox("Fenster-Schritt", ["halbjährlich", "quartalsweise"],
                               index=0, key="smi_sh_step")
    with sh3:
        _dd_ceiling = st.slider("Max-Drawdown-Obergrenze (%)", 20, 80, 45, 5,
                                key="smi_sh_ddc",
                                help="Kombinationen mit einem schlechteren Max-"
                                     "Drawdown (Median über alle Fenster) als "
                                     "diese Obergrenze werden ausgeschlossen.") / 100.0
    with sh4:
        st.caption("")
        _shgo = st.button("Risiko/Rendite-Grid starten", key="smi_sh_go")

    if _shgo:
        st.session_state["smi_sh_has_run"] = True

    if st.session_state.get("smi_sh_has_run"):
        _sh_allocs = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
        _sh_widths = [0.05, 0.10, 0.15, 0.20]
        _sh_dca = [6, 12, 18, 24]
        _sh_sm = 6 if _shstep == "halbjährlich" else 3

        with st.spinner("Rechne Startallokation × Bandbreite × DCA-Fenster über "
                         "alle rollierenden Fenster… (grosses Grid, kann mehrere "
                         "Minuten dauern)"):
            shgrid = compute_smi_sharpe_grid(
                prices, divs, btc_series, fx, weights, initial_capital,
                _sh_allocs, _sh_widths, _sh_dca, tx_cost_bps, _shw, _sh_sm,
                _dd_ceiling)

        if shgrid.empty:
            st.warning("Zu wenig überlappende Daten für die Fensteranalyse.")
        else:
            _shn = shgrid["start"].nunique()
            st.caption(f"{len(shgrid):,} Engine-Läufe · {shgrid.groupby(['alloc','width','dca_m']).ngroups} "
                       f"Parameterkombinationen × {_shn} rollierende {_shw}-Jahres-Fenster · "
                       "Gebühren nicht angewendet (Mechanismus isoliert getestet)")

            shsumm = shgrid.groupby(["alloc", "width", "dca_m"]).agg(
                Median_CAGR=("cagr", "median"), Median_Vol=("vol", "median"),
                Median_Sharpe=("sharpe", "median"), Median_Calmar=("calmar", "median"),
                Median_MaxDD=("max_dd", "median"), Median_DCA=("dca_share", "median"),
            ).reset_index()
            shsumm["feasible"] = shsumm["Median_MaxDD"] >= -_dd_ceiling
            shsumm = shsumm.sort_values("Median_Sharpe", ascending=False)
            _n_feas = int(shsumm["feasible"].sum())

            st.caption(f"**{_n_feas} von {len(shsumm)}** Kombinationen innerhalb der "
                       f"Drawdown-Obergrenze ({_dd_ceiling*100:.0f}%)")

            if _n_feas == 0:
                st.error("⚠️ Keine Kombination bleibt unter der gewählten Drawdown-"
                         "Obergrenze. Obergrenze lockern oder Bandbreite/Allokation "
                         "enger fassen.")
            else:
                _best = shsumm[shsumm["feasible"]].iloc[0]
                b1, b2, b3, b4, b5 = st.columns(5)
                with b1:
                    st.metric("Beste Startallokation", f"{_best['alloc']*100:.1f}%")
                with b2:
                    st.metric("Beste Bandbreite", f"{_best['width']*100:.0f}pp")
                with b3:
                    st.metric("Bestes DCA-Fenster", f"{_best['dca_m']:.0f} Mte")
                with b4:
                    st.metric("Sharpe (Median)", f"{_best['Median_Sharpe']:.2f}")
                with b5:
                    st.metric("Calmar (Median)", f"{_best['Median_Calmar']:.2f}")
                st.caption(
                    f"Median-CAGR {_best['Median_CAGR']*100:.1f}% · Median-Vol "
                    f"{_best['Median_Vol']*100:.1f}% · Median-MaxDD "
                    f"{_best['Median_MaxDD']*100:.1f}% · DCA-Anteil "
                    f"{_best['Median_DCA']*100:.0f}% — höchste Sharpe Ratio unter "
                    "allen Kombinationen, die die Drawdown-Obergrenze einhalten.")

                st.markdown("##### Top 10 nach Sharpe Ratio (innerhalb der Drawdown-Obergrenze)")
                _disp = shsumm[shsumm["feasible"]].head(10).copy()
                _disp["Startallokation"] = (_disp["alloc"] * 100).round(1).astype(str) + "%"
                _disp["Bandbreite"] = (_disp["width"] * 100).round(0).astype(int).astype(str) + "pp"
                _disp["DCA-Fenster"] = _disp["dca_m"].astype(int).astype(str) + " Mte"
                _disp["CAGR"] = (_disp["Median_CAGR"] * 100).round(2).astype(str) + "%"
                _disp["Vol"] = (_disp["Median_Vol"] * 100).round(2).astype(str) + "%"
                _disp["Sharpe"] = _disp["Median_Sharpe"].round(2)
                _disp["Calmar"] = _disp["Median_Calmar"].round(2)
                _disp["MaxDD"] = (_disp["Median_MaxDD"] * 100).round(1).astype(str) + "%"
                _disp["DCA-Anteil"] = (_disp["Median_DCA"] * 100).round(0).astype(str) + "%"
                st.dataframe(_disp[["Startallokation", "Bandbreite", "DCA-Fenster",
                                    "CAGR", "Vol", "Sharpe", "Calmar", "MaxDD", "DCA-Anteil"]],
                            use_container_width=True, hide_index=True)

                st.markdown("##### Sharpe Ratio nach Startallokation und Bandbreite (bestes DCA-Fenster je Zelle)")
                _best_per_cell = shgrid.groupby(["alloc", "width"]).apply(
                    lambda g: g.groupby("dca_m")["sharpe"].median().max(),
                    include_groups=False).reset_index(name="best_sharpe")
                _piv = _best_per_cell.pivot(index="alloc", columns="width", values="best_sharpe")
                figsh = go.Figure(data=go.Heatmap(
                    z=_piv.values,
                    x=[f"{v*100:.0f}pp" for v in _piv.columns],
                    y=[f"{a*100:.1f}%" for a in _piv.index],
                    colorscale=[[0, OAK_GREEN_2], [0.5, OAK_SAGE], [1, OAK_GOLD]],
                    text=[[f"{v:.2f}" for v in r] for r in _piv.values],
                    texttemplate="%{text}", showscale=False))
                figsh.update_xaxes(title_text="Bandbreite", type="category")
                figsh.update_yaxes(title_text="Startallokation BTC", type="category")
                figsh = style_plotly(figsh, height=340)
                st.plotly_chart(figsh, use_container_width=True)
                st.caption(
                    "Bestes Sharpe-Ratio über alle getesteten DCA-Fenster je Zelle. "
                    "Kein Downside-Constraint hier — nur die Drawdown-Obergrenze oben. "
                    "Das Upside wird nicht künstlich gekappt: eine hohe Startallokation "
                    "mit entsprechend hoher Rendite bleibt zulässig, solange der "
                    "Drawdown unter der Obergrenze bleibt.")

        st.warning(
            "⚠️ **Provisorisch, solange mit synthetischen Testpfaden gerechnet "
            "wird.** Grosses Grid — im Deployment mit echten Kursen entsprechend "
            "länger laufend, aber derselbe Code, dieselbe Zielfunktion.")

    st.markdown("## Parameter-Sensitivität")
    st.markdown(
        f"<p style='color:{OAK_CREAM_DIM}; font-size:13px;'>"
        "Robustness check: re-runs the backtest across a grid of initial BTC "
        "allocations and rebalancing thresholds, holding all other parameters "
        "fixed. Shows how net CAGR and maximum drawdown respond to the two key "
        "risk levers — a single strong path means little if nearby parameters "
        "collapse.</p>",
        unsafe_allow_html=True
    )

    if st.button("Sensitivitätsanalyse starten (Grid-Backtest)", key="sens_btn"):
        # Grids: initial BTC weight × upper threshold
        btc_grid = [0.05, 0.10, 0.15, 0.20, 0.25]
        thr_grid = [0.20, 0.25, 0.30, 0.35]
        # Ensure target < threshold for each cell; keep target = current target
        # but clamp below the threshold being tested.
        cagr_matrix = []
        dd_matrix = []
        prog = st.progress(0.0, text="Running grid backtests ...")
        total_cells = len(btc_grid) * len(thr_grid)
        done = 0
        for b in btc_grid:
            cagr_row = []
            dd_row = []
            for thr in thr_grid:
                tgt = min(target_btc_pct, thr - 0.05)
                if tgt <= 0:
                    tgt = thr * 0.6
                try:
                    ts_g, _, _ = run_strategy(
                        prices, divs, btc_series, fx,
                        initial_capital, weights,
                        b, thr, tgt,
                        rebal_dates, dca_months, tx_cost_bps=tx_cost_bps
                    )
                    if ts_g is not None and not ts_g.empty and "total_value" in ts_g.columns:
                        net_g, _, _, _ = apply_fees(
                            ts_g["total_value"], initial_capital,
                            mgmt_fee_annual=mgmt_fee_pct, perf_fee_rate=perf_fee_pct,
                            hwm_hurdle=hwm_hurdle_pct,
                            crystallization_freq=crystallization_freq,
                            hurdle_type=hurdle_type,
                        )
                        m_g = compute_risk_metrics(net_g, risk_free_rate)
                        cagr_row.append(m_g.get("cagr", float("nan")) * 100)
                        dd_row.append(m_g.get("max_drawdown", float("nan")) * 100)
                    else:
                        cagr_row.append(float("nan"))
                        dd_row.append(float("nan"))
                except Exception:
                    cagr_row.append(float("nan"))
                    dd_row.append(float("nan"))
                done += 1
                prog.progress(done / total_cells, text=f"Running grid backtests ... {done}/{total_cells}")
            cagr_matrix.append(cagr_row)
            dd_matrix.append(dd_row)
        prog.empty()

        x_labels = [f"{int(t*100)}%" for t in thr_grid]
        y_labels = [f"{int(b*100)}%" for b in btc_grid]

        def _mark_current(figh):
            """Outline the cell of the CURRENT sidebar parameters, if on the grid."""
            _cx = f"{int(round(upper_threshold*100))}%"
            _cy = f"{int(round(initial_btc_pct*100))}%"
            if _cx in x_labels and _cy in y_labels:
                figh.add_annotation(x=_cx, y=_cy, text="◉", showarrow=False,
                                    yshift=-1, font=dict(size=18, color=OAK_CREAM))
                figh.add_annotation(x=_cx, y=_cy, text="current", showarrow=False,
                                    yshift=-17, font=dict(size=9, color=OAK_CREAM))
            return figh

        sens_col1, sens_col2 = st.columns(2)
        with sens_col1:
            fig_cagr = go.Figure(data=go.Heatmap(
                z=cagr_matrix, x=x_labels, y=y_labels,
                colorscale=[[0, OAK_RED], [0.5, OAK_GREEN_3], [1, OAK_GOLD]],
                text=[[f"{v:.1f}%" for v in row] for row in cagr_matrix],
                texttemplate="%{text}", textfont=dict(size=11, color=OAK_CREAM),
                colorbar=dict(title="CAGR (%)", tickfont=dict(color=OAK_CREAM)),
                hovertemplate="BTC init %{y} · Threshold %{x}<br>Net CAGR %{z:.2f}%<extra></extra>",
            ))
            fig_cagr.update_layout(title="Netto-CAGR (%)")
            fig_cagr = style_plotly(_mark_current(fig_cagr), height=380)
            fig_cagr.update_xaxes(title_text="Upper Threshold")
            fig_cagr.update_yaxes(title_text="Initial BTC %")
            st.plotly_chart(fig_cagr, use_container_width=True)

        with sens_col2:
            fig_dd = go.Figure(data=go.Heatmap(
                z=dd_matrix, x=x_labels, y=y_labels,
                colorscale=[[0, OAK_RED], [1, OAK_GREEN_3]],
                text=[[f"{v:.1f}%" for v in row] for row in dd_matrix],
                texttemplate="%{text}", textfont=dict(size=11, color=OAK_CREAM),
                colorbar=dict(title="Max. Drawdown (%)", tickfont=dict(color=OAK_CREAM)),
                hovertemplate="BTC init %{y} · Threshold %{x}<br>Max Drawdown %{z:.2f}%<extra></extra>",
            ))
            fig_dd.update_layout(title="Maximum Drawdown (%)")
            fig_dd = style_plotly(_mark_current(fig_dd), height=380)
            fig_dd.update_xaxes(title_text="Upper Threshold")
            fig_dd.update_yaxes(title_text="Initial BTC %")
            st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown(
            f"<p style='color:{OAK_SAGE_DIM}; font-size:11px;'>"
            "Rows: initial BTC allocation · Columns: BTC upper threshold. "
            "All other parameters held at current sidebar values. The rebalance "
            "target is clamped to stay below each tested threshold.</p>",
            unsafe_allow_html=True
        )

    # =====================================================================
    # Monte-Carlo Forward Projection
    # =====================================================================
    st.markdown("## Monte-Carlo-Projektion")
    st.markdown(
        f"<p style='color:{OAK_CREAM_DIM}; font-size:13px;'>"
        "Forward-looking simulation: bootstraps the strategy's historical daily "
        "net returns to generate thousands of possible future paths, showing the "
        "range of outcomes as percentile bands. This is a statistical "
        "illustration based on past behaviour — <strong>not a forecast</strong>.</p>",
        unsafe_allow_html=True
    )

    mc_col1, mc_col2, mc_col3 = st.columns(3)
    with mc_col1:
        mc_years = st.slider("Projection Horizon (years)", 1, 10, 5, key="mc_years")
    with mc_col2:
        mc_paths = st.select_slider("Number of Paths", options=[500, 1000, 2000, 5000],
                                    value=1000, key="mc_paths")
    with mc_col3:
        mc_method = st.selectbox("Method", ["Bootstrap (historical)", "Normal (parametric)"],
                                 key="mc_method",
                                 help="Bootstrap resamples actual historical daily returns "
                                      "(keeps fat tails). Normal assumes Gaussian returns "
                                      "with the same mean/volatility.")

    if st.button("Monte-Carlo-Simulation starten", key="mc_btn"):
        net_series = ts["total_value_net"]
        daily_ret = net_series.pct_change().dropna().values
        if len(daily_ret) < 30:
            st.warning("Not enough history for a meaningful projection.")
        else:
            start_value = float(net_series.iloc[-1])
            horizon_days = int(mc_years * 252)
            n_paths = int(mc_paths)
            rng = np.random.default_rng(42)

            if mc_method.startswith("Bootstrap"):
                # Resample daily returns with replacement
                sampled = rng.choice(daily_ret, size=(n_paths, horizon_days), replace=True)
            else:
                mu = float(np.mean(daily_ret))
                sigma = float(np.std(daily_ret))
                sampled = rng.normal(mu, sigma, size=(n_paths, horizon_days))

            # Cumulative paths
            cum = start_value * np.cumprod(1.0 + sampled, axis=1)
            # Percentile bands across paths at each time step
            pcts = [5, 25, 50, 75, 95]
            bands = {p: np.percentile(cum, p, axis=0) for p in pcts}

            future_idx = pd.bdate_range(net_series.index[-1], periods=horizon_days + 1, freq="B")[1:]

            fig_mc = go.Figure()
            # Shaded 5-95 band
            fig_mc.add_trace(go.Scatter(
                x=future_idx, y=bands[95], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_mc.add_trace(go.Scatter(
                x=future_idx, y=bands[5], mode="lines", fill="tonexty",
                fillcolor="rgba(153,167,150,0.15)", line=dict(width=0),
                name="5.–95. Perzentil"))
            # 25-75 band
            fig_mc.add_trace(go.Scatter(
                x=future_idx, y=bands[75], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_mc.add_trace(go.Scatter(
                x=future_idx, y=bands[25], mode="lines", fill="tonexty",
                fillcolor="rgba(153,167,150,0.30)", line=dict(width=0),
                name="25.–75. Perzentil"))
            # Median
            fig_mc.add_trace(go.Scatter(
                x=future_idx, y=bands[50], mode="lines",
                line=dict(color=OAK_GOLD, width=2.5), name="Median-Pfad"))
            fig_mc = style_plotly(fig_mc, height=420)
            fig_mc.update_xaxes(title_text="Projected Date")
            fig_mc.update_yaxes(title_text="Projected Value (CHF)", tickformat=",.0f")
            st.plotly_chart(fig_mc, use_container_width=True)

            # Summary table of terminal outcomes
            terminal = cum[:, -1]
            t1, t2, t3, t4, t5 = st.columns(5)
            t1.metric("5. Perzentil", fmt_chf(np.percentile(terminal,5)))
            t2.metric("25. Perzentil", fmt_chf(np.percentile(terminal,25)))
            t3.metric("Median", fmt_chf(np.percentile(terminal,50)))
            t4.metric("75. Perzentil", fmt_chf(np.percentile(terminal,75)))
            t5.metric("95. Perzentil", fmt_chf(np.percentile(terminal,95)))

            prob_loss = float(np.mean(terminal < start_value)) * 100
            st.markdown(
                f"<p style='color:{OAK_SAGE_DIM}; font-size:12px;'>"
                f"Starting from the current net value of CHF {start_value:,.0f}, "
                f"over a {mc_years}-year horizon across {n_paths:,} simulated paths: "
                f"<strong>{prob_loss:.1f}%</strong> of paths end below today's value. "
                "Bootstrapping preserves the historical return distribution including "
                "its tails; results are illustrative and assume the future resembles "
                "the backtest period — which it may not.</p>",
                unsafe_allow_html=True
            )

    # ---- Fee Detail Section ----
    st.markdown("## Gebührenstruktur & Kostendetail")
    fee_col_a, fee_col_b = st.columns([1, 2])
    with fee_col_a:
        st.markdown(
            f"<div style='background:{OAK_GREEN_2}; padding:20px 24px; "
            f"border:1px solid {OAK_BORDER}; border-left:3px solid {OAK_GOLD}; "
            f"border-radius:10px;'>"
            f"<div style='color:{OAK_SAGE}; font-size:10px; text-transform:uppercase; "
            f"letter-spacing:0.14em; font-weight:600;'>Fee Structure</div>"
            f"<div style='color:{OAK_CREAM_DIM}; font-size:13px; margin-top:12px; line-height:1.8;'>"
            f"<strong style='color:{OAK_CREAM};'>Management Fee:</strong> {mgmt_fee_pct*100:.2f}% p.a. "
            f"· CHF {total_mgmt_fees:,.0f}<br>"
            f"<span style='font-size:11px; color:{OAK_SAGE_DIM};'>Accrued daily (1/252 per trading day)</span><br><br>"
            f"<strong style='color:{OAK_CREAM};'>Performance Fee:</strong> {perf_fee_pct*100:.0f}% "
            f"· CHF {total_perf_fees:,.0f}<br>"
            f"<span style='font-size:11px; color:{OAK_SAGE_DIM};'>Crystallized {crystallization_freq.lower()} on gains above HWM</span><br><br>"
            f"<strong style='color:{OAK_CREAM};'>HWM Hurdle:</strong> {hwm_hurdle_pct*100:.1f}% (Year 1) · {hurdle_type}<br>"
            f"<span style='font-size:11px; color:{OAK_SAGE_DIM};'>Initial HWM = Initial × (1 + Hurdle)</span><br><br>"
            f"<strong style='color:{OAK_CREAM};'>Transaction Costs:</strong> {tx_cost_bps:.0f} bps/trade "
            f"· CHF {total_tx_costs:,.0f}<br>"
            f"<span style='font-size:11px; color:{OAK_SAGE_DIM};'>Already reflected in gross NAV</span><br><br>"
            f"<strong style='color:{OAK_CREAM};'>Total Fees Paid:</strong> CHF {fees_total:,.0f}<br>"
            f"<span style='font-size:11px; color:{OAK_SAGE_DIM};'>"
            f"= Mgmt + Perf + TX · {fees_total_pct_initial:.2f}% of initial capital over {years:.1f} years"
            f"</span><br><br>"
            f"<div style='border-top:1px solid {OAK_BORDER}; margin:4px 0 12px;'></div>"
            f"<strong style='color:{OAK_CREAM};'>Dividend Withholding Tax:</strong> "
            f"{int(WITHHOLDING_TAX*100)}% · CHF {total_wht:,.0f}<br>"
            f"<span style='font-size:11px; color:{OAK_SAGE_DIM};'>Non-reclaimable (AMC) · a tax drag, "
            f"not a fee — applied equally to the SMI TR benchmark</span>"
            f"</div></div>",
            unsafe_allow_html=True
        )
    with fee_col_b:
        if not fee_events_df.empty:
            # Build per-period fee ledger (mgmt + perf), not just perf-fee events
            fed = fee_events_df.copy()
            fed["date"] = pd.to_datetime(fed["date"]).dt.strftime("%Y-%m-%d")
            if "mgmt_fee" not in fed.columns:
                fed["mgmt_fee"] = 0.0
            fed["period_cost"] = fed["mgmt_fee"].fillna(0) + fed["perf_fee"].fillna(0)
            fed_disp = fed.rename(columns={
                "date": "Period-End", "period": "Period", "year": "Year",
                "nav_before_perf": "NAV before Fees",
                "hwm_before": "HWM",
                "excess": "Excess over HWM",
                "mgmt_fee": "Mgmt Fee",
                "perf_fee": "Perf Fee",
                "period_cost": "Total Cost",
                "nav_after_perf": "NAV after Fees",
            })
            for col in ["NAV before Fees", "HWM", "Excess over HWM",
                        "Mgmt Fee", "Perf Fee", "Total Cost", "NAV after Fees"]:
                fed_disp[col] = fed_disp[col].apply(lambda x: f"CHF {x:,.0f}")
            display_cols = ["Period", "NAV before Fees", "HWM", "Excess over HWM",
                            "Mgmt Fee", "Perf Fee", "Total Cost", "NAV after Fees"]
            st.dataframe(fed_disp[display_cols],
                         use_container_width=True, hide_index=True, height=320)
            st.caption(
                "Per-period cost ledger. Mgmt fee accrues daily and is shown summed "
                "per crystallization period; perf fee crystallizes at period end on "
                "gains above the HWM. Transaction costs and the dividend withholding "
                "tax are already reflected in the NAV (see panel at left).")

    # =====================================================================
    # BTC Weight Over Time
    # =====================================================================
    st.markdown("## Bitcoin-Quote & Schwellenwert")
    fig_w = go.Figure()
    fig_w.add_trace(go.Scatter(x=ts.index, y=ts["btc_pct"] * 100,
                               name="BTC in % des Portfolios",
                               line=dict(color=OAK_BTC, width=2.5),
                               fill="tozeroy", fillcolor="rgba(247,147,26,0.1)"))
    # Threshold lines
    fig_w.add_hline(y=upper_threshold * 100, line=dict(color=OAK_RED, width=2, dash="dash"),
                    annotation_text=f"Upper Threshold {upper_threshold*100:.0f}%",
                    annotation_position="top right",
                    annotation_font=dict(color=OAK_RED, size=11))
    fig_w.add_hline(y=target_btc_pct * 100, line=dict(color=OAK_SAGE, width=1.5, dash="dot"),
                    annotation_text=f"Target {target_btc_pct*100:.0f}%",
                    annotation_position="bottom right",
                    annotation_font=dict(color=OAK_SAGE, size=11))
    fig_w.add_hline(y=initial_btc_pct * 100, line=dict(color=OAK_CREAM_DIM, width=1, dash="dot"),
                    annotation_text=f"Initial {initial_btc_pct*100:.0f}%",
                    annotation_position="bottom left",
                    annotation_font=dict(color=OAK_CREAM_DIM, size=11))
    if not evts.empty:
        evts2 = evts.copy()
        evts2["btc_pct_pct"] = evts2["btc_pct_before"] * 100
        fig_w.add_trace(go.Scatter(
            x=evts2["date"], y=evts2["btc_pct_pct"], mode="markers",
            name="Verkaufs-Trigger",
            marker=dict(symbol="diamond", size=12, color=OAK_RED,
                        line=dict(color=OAK_CREAM, width=1.5)),
        ))
    fig_w = style_plotly(fig_w, height=380)
    fig_w.update_yaxes(title_text="BTC % of Portfolio", ticksuffix="%")
    st.plotly_chart(fig_w, use_container_width=True)

    # =====================================================================
    # BTC Accumulation
    # =====================================================================
    st.markdown("## Bitcoin-Bestand vs. Marktpreis")
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=ts.index, y=ts["btc_held"],
                              name="BTC-Bestand", line=dict(color=OAK_BTC, width=2.5),
                              fill="tozeroy", fillcolor="rgba(247,147,26,0.12)"),
                   secondary_y=False)
    fig2.add_trace(go.Scatter(x=btc_series.index, y=btc_series.values,
                              name="BTC-Preis (USD)",
                              line=dict(color=OAK_CREAM, width=1.5, dash="dot")),
                   secondary_y=True)
    fig2 = style_plotly(fig2, height=400)
    fig2.update_yaxes(title_text="BTC holding", secondary_y=False, tickformat=",.4f")
    fig2.update_yaxes(title_text="BTC price (USD)", secondary_y=True,
                      tickformat=",.0f", showgrid=False)
    st.plotly_chart(fig2, use_container_width=True)

    # =====================================================================
    # Dividends by Year — actual harvested cashflow from the simulation,
    # computed on the live (evolving) share counts rather than a frozen
    # initial-share approximation. Net of the non-reclaimable 35% withholding
    # tax; this is the exact cash that funded the BTC DCA.
    # =====================================================================
    div_cf = ts.attrs.get("dividend_cashflows")
    if div_cf is not None and not div_cf.empty:
        st.markdown("## Dividendenerträge nach Jahr")
        st.markdown(
            f"<p style='color:{OAK_CREAM_DIM}; font-size:13px; margin-top:-8px;'>"
            f"Net of {int(WITHHOLDING_TAX*100)}% Swiss withholding tax "
            "(non-reclaimable in the AMC wrapper) — i.e. the amount actually "
            "available for reinvestment into the BTC sleeve. Reflects the real "
            "holdings over time, including portfolio growth and rebalances.</p>",
            unsafe_allow_html=True
        )
        div_cf = div_cf.copy()
        div_cf["year"] = pd.to_datetime(div_cf["date"]).dt.year
        agg = div_cf.groupby(["year", "ticker"])["cash_chf"].sum().reset_index()
        year_totals = agg.groupby("year")["cash_chf"].sum()

        fig3 = go.Figure()
        tickers_sorted = sorted(agg["ticker"].unique(),
                                key=lambda t: -agg[agg["ticker"] == t]["cash_chf"].sum())
        for i, t in enumerate(tickers_sorted):
            sub = agg[agg["ticker"] == t]
            name = SMI_CONSTITUENTS.get(t, (t,))[0]
            fig3.add_trace(go.Bar(
                x=sub["year"], y=sub["cash_chf"], name=name,
                marker=dict(color=CHART_BAR_COLORS[i % len(CHART_BAR_COLORS)],
                            line=dict(color=OAK_GREEN_2, width=0.5)),
                hovertemplate="%{fullData.name}: CHF %{y:,.0f}<extra></extra>"))
        fig3.update_layout(barmode="stack")
        fig3 = style_plotly(fig3, height=440)
        fig3.update_xaxes(title_text="Year", dtick=1)
        fig3.update_yaxes(title_text="Dividends (CHF, net)", tickformat=",.0f")

        # Per-year total above each stacked bar. Compact CHF formatting
        # (e.g. "CHF 326k" / "CHF 1.20M") so adjacent labels don't collide.
        def _compact_chf(v):
            if v >= 1e6:
                return f"CHF {v / 1e6:.2f}M"
            if v >= 1e3:
                return f"CHF {v / 1e3:.0f}k"
            return f"CHF {v:,.0f}"

        for yr, tot in year_totals.items():
            fig3.add_annotation(
                x=int(yr), y=float(tot), text=_compact_chf(tot),
                showarrow=False, yshift=10, xanchor="center", yanchor="bottom",
                font=dict(family="'Inter', sans-serif", size=10, color=OAK_CREAM))
        # Headroom so the topmost total label isn't clipped
        _ymax = float(year_totals.max()) if len(year_totals) else 0.0
        if _ymax > 0:
            fig3.update_yaxes(range=[0, _ymax * 1.13])

        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            "Actual dividend cashflow harvested in the simulation, on the holdings "
            "as they evolved (initial allocation, threshold reallocations and "
            "quarterly rebalances) — the cash that funded the BTC DCA.")

    # =====================================================================
    # Detail tables
    # =====================================================================
    st.markdown("## Transaktionsdetails")
    with st.expander("BTC-Transaktionen (Kauf & Verkauf)"):
        if not txs.empty:
            tx_disp = txs.copy()
            tx_disp["date"] = pd.to_datetime(tx_disp["date"]).dt.strftime("%Y-%m-%d")
            st.dataframe(tx_disp, use_container_width=True, height=400)
            st.download_button("Download CSV", tx_disp.to_csv(index=False).encode(),
                               "btc_transactions.csv", "text/csv")

    with st.expander("Threshold-Rebalancing-Ereignisse"):
        if not evts.empty:
            evt_disp = evts.copy()
            evt_disp["date"] = pd.to_datetime(evt_disp["date"]).dt.strftime("%Y-%m-%d")
            evt_disp["btc_pct_before"] = (evt_disp["btc_pct_before"] * 100).round(2).astype(str) + "%"
            evt_disp["btc_pct_after"] = (evt_disp["btc_pct_after"] * 100).round(2).astype(str) + "%"
            st.dataframe(evt_disp, use_container_width=True, height=300)
            st.download_button("Download CSV", evts.to_csv(index=False).encode(),
                               "threshold_events.csv", "text/csv")
        else:
            st.info("No threshold rebalances triggered in this period.")

    with st.expander("Tägliche Portfolio-Zeitreihe"):
        df_export = ts.reset_index()
        st.dataframe(df_export.tail(50), use_container_width=True)
        st.download_button("Download Full CSV", df_export.to_csv(index=False).encode(),
                           "portfolio_timeseries.csv", "text/csv")

    with st.expander("Benchmarks · Tagesreihen (SMI TR & Kursindex)"):
        if not bench.empty:
            bench_export = bench.reset_index()
            st.dataframe(bench_export.tail(50), use_container_width=True)
            st.download_button("Download Benchmarks CSV",
                               bench_export.to_csv(index=False).encode(),
                               "benchmarks.csv", "text/csv")

    # =====================================================================
    # PDF Tearsheet Export
    # =====================================================================
    st.markdown("## Export")
    st.markdown(
        f"<p style='color:{OAK_CREAM_DIM}; font-size:13px;'>"
        "Generate a presentation-ready PDF tearsheet with all key metrics, charts, "
        "methodology and disclosures — suitable for internal review or qualified "
        "investor discussions.</p>",
        unsafe_allow_html=True
    )

    if st.button("Generate PDF Tearsheet", use_container_width=False):
        with st.spinner("Building PDF tearsheet ..."):
            try:
                from pdf_report import (build_tearsheet, build_bilingual_tearsheet,
                                        render_line_chart,
                                        render_bar_chart, render_scatter_chart,
                                        compute_period_returns, identify_top_drawdowns,
                                        get_font_status)

                # Render charts with matplotlib (stable, no headless browser).
                pdf_figures = []
                # 1. Portfolio evolution (net strategy + benchmarks)
                _evo = [
                    ("Strategy (Net of Fees)", ts["total_value_net"], OAK_GOLD, {"lw": 2.2}),
                ]
                if not bench.empty:
                    _evo.append(("SMI Total Return", bench["smi_tr"], OAK_SAGE, {"lw": 1.5, "ls": "--"}))
                    _evo.append(("SMI Price Index", bench["smi_price"], "#7D8A78", {"lw": 1.2, "ls": ":"}))
                png1 = render_line_chart(_evo, ylabel="Value (CHF)", fill_first=True,
                                         annotate_end=True)
                pdf_figures.append(("Portfolio Evolution vs. Benchmarks", png1))

                # 2. Drawdown — with crisis-phase shading where they fall in range
                dd_strat = compute_drawdown(ts["total_value_net"])
                _dd = [("Strategy (Net)", dd_strat, OAK_GOLD, {"lw": 1.8})]
                if not bench.empty:
                    _dd.append(("SMI Total Return", compute_drawdown(bench["smi_tr"]), OAK_SAGE, {"lw": 1.3, "ls": "--"}))
                _all_crises = [
                    ("2020-02-19", "2020-04-07", "COVID-19"),
                    ("2022-01-03", "2022-10-20", "2022 Bear"),
                ]
                _t0, _t1 = ts.index[0], ts.index[-1]
                _crises = [(s, e, lbl) for (s, e, lbl) in _all_crises
                           if pd.Timestamp(e) >= _t0 and pd.Timestamp(s) <= _t1]
                png2 = render_line_chart(_dd, ylabel="Drawdown", percent=True,
                                         fill_first=True, crisis_phases=_crises)
                pdf_figures.append(("Drawdown Analysis", png2))

                # 3. Yearly returns bar chart
                try:
                    yearly_net = ts["total_value_net"].resample("YE").last()
                    yearly_ret = yearly_net.pct_change()
                    yearly_ret.iloc[0] = yearly_net.iloc[0] / initial_capital - 1
                    # Flag partial first/last years (backtest doesn't span the whole year)
                    first_dt, last_dt = ts.index[0], ts.index[-1]
                    yr_labels = []
                    for y in yearly_net.index.year:
                        partial = ((y == first_dt.year and (first_dt.month, first_dt.day) > (1, 7))
                                   or (y == last_dt.year and (last_dt.month, last_dt.day) < (12, 24)))
                        yr_labels.append(f"{y}*" if partial else str(y))
                    yr_vals = list(yearly_ret.values * 100)
                    png3 = render_bar_chart(yr_labels, yr_vals, ylabel="Annual Return (Net)",
                                            hurdle=hwm_hurdle_pct * 100)
                    pdf_figures.append(("Yearly Net Performance", png3))
                except Exception:
                    pass

                # 4. Risk/Return scatter — Strategy vs benchmarks
                pdf_scatter = None
                try:
                    sc_points = [("Strategy", strat_m.get("vol_ann", 0) * 100,
                                  strat_m.get("cagr", 0) * 100, OAK_GOLD, "o")]
                    if tr_m:
                        sc_points.append(("SMI Total Return", tr_m.get("vol_ann", 0) * 100,
                                          tr_m.get("cagr", 0) * 100, OAK_SAGE, "s"))
                    if pr_m:
                        sc_points.append(("SMI Price Index", pr_m.get("vol_ann", 0) * 100,
                                          pr_m.get("cagr", 0) * 100, "#9AA595", "^"))
                    pdf_scatter = render_scatter_chart(sc_points)
                except Exception:
                    pdf_scatter = None

                # Key takeaways — data-driven bullet points (bilingual)
                pdf_takeaways_en = []
                pdf_takeaways_de = []
                try:
                    _exc = excess_vs_tr * 100
                    _rel_en = "outperformed" if _exc >= 0 else "trailed"
                    _rel_de = "übertraf" if _exc >= 0 else "lag unter"
                    pdf_takeaways_en.append(
                        f"Net CAGR of {strat_net_cagr*100:.1f}% over the full backtest, "
                        f"{_rel_en} the SMI Total Return benchmark by {abs(_exc):.1f}% p.a.")
                    pdf_takeaways_de.append(
                        f"Netto-CAGR von {strat_net_cagr*100:.1f}% über den gesamten Backtest, "
                        f"{_rel_de} dem SMI Total Return Benchmark um {abs(_exc):.1f}% p.a.")
                    pdf_takeaways_en.append(
                        f"Sharpe ratio of {_fmt_num(strat_m.get('sharpe'))} and maximum drawdown of "
                        f"{_fmt_pct(strat_m.get('max_drawdown'))}, reflecting the structural Bitcoin allocation.")
                    pdf_takeaways_de.append(
                        f"Sharpe Ratio von {_fmt_num(strat_m.get('sharpe'))} und maximaler Drawdown von "
                        f"{_fmt_pct(strat_m.get('max_drawdown'))} — Ausdruck der strukturellen Bitcoin-Allokation.")
                    pdf_takeaways_en.append(
                        f"Total fees of CHF {fees_total:,.0f} ({fee_drag*100:.1f}% p.a. drag), "
                        f"net of {int(WITHHOLDING_TAX*100)}% non-reclaimable dividend withholding tax.")
                    pdf_takeaways_de.append(
                        f"Gesamtgebühren von CHF {fees_total:,.0f} ({fee_drag*100:.1f}% p.a. Drag), "
                        f"nach Abzug der {int(WITHHOLDING_TAX*100)}% nicht rückforderbaren Dividenden-Quellensteuer.")
                except Exception:
                    pdf_takeaways_en = None
                    pdf_takeaways_de = None

                def _row(metric, key, fmt):
                    s = strat_m.get(key)
                    t = tr_m.get(key) if tr_m else None
                    p = pr_m.get(key) if pr_m else None
                    return [metric, fmt(s), fmt(t), fmt(p)]

                pct = lambda x: _fmt_pct(x) if x is not None else "—"
                num = lambda x: _fmt_num(x) if x is not None else "—"

                risk_rows = [
                    _row("Total Return", "total_return", pct),
                    _row("CAGR", "cagr", pct),
                    _row("Annualized Volatility", "vol_ann", pct),
                    _row("Downside Deviation", "downside_vol", pct),
                    _row("Max Drawdown", "max_drawdown", pct),
                    _row("Sharpe Ratio", "sharpe", num),
                    _row("Sortino Ratio", "sortino", num),
                    _row("Calmar Ratio", "calmar", num),
                    _row("VaR 95% (monthly)", "var_95_monthly", pct),
                    _row("CVaR 95% (monthly)", "cvar_95_monthly", pct),
                ]

                fee_rows = []
                if not fee_events_df.empty:
                    for _, r in fee_events_df.iterrows():
                        _mgmt = float(r.get("mgmt_fee", 0.0) or 0.0)
                        _perf = float(r.get("perf_fee", 0.0) or 0.0)
                        fee_rows.append([
                            r.get("period", str(r.get("year", ""))),
                            f"CHF {_mgmt:,.0f}",
                            f"CHF {_perf:,.0f}",
                            f"CHF {_mgmt + _perf:,.0f}",
                        ])

                # Investment universe: the digital-asset sleeve (Bitcoin) first,
                # then the SMI equity replication. BTC weight is portfolio-level
                # (target + cap); SMI weights are within the equity sleeve.
                btc_weight_label = (
                    f"{target_btc_pct*100:.0f}% \u00b7 cap {upper_threshold*100:.0f}%")
                universe_rows = (
                    [["Bitcoin", "BTC", "Digital Assets", btc_weight_label]]
                    + [[v[0], t, v[2], v[1]] for t, v in SMI_CONSTITUENTS.items()]
                )

                # Build monthly returns dict {year: [12 values in %]} for the PDF heatmap
                pdf_monthly = {}
                try:
                    mtx = monthly_returns_matrix(ts["total_value_net"])
                    month_cols = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    for yr in mtx.index:
                        row = []
                        for mc in month_cols:
                            v = mtx.loc[yr, mc] if mc in mtx.columns else np.nan
                            row.append(None if pd.isna(v) else float(v) * 100.0)
                        pdf_monthly[int(yr)] = row
                except Exception:
                    pdf_monthly = None

                # Data-driven executive summary (bilingual)
                _exc = excess_vs_tr * 100
                _verb_en = "outperforming" if _exc >= 0 else "trailing"
                if _exc >= 0:
                    _perf_de = f"und übertraf den SMI Total Return Benchmark um {abs(_exc):.1f}% pro Jahr"
                else:
                    _perf_de = f"und lag damit {abs(_exc):.1f}% pro Jahr unter dem SMI Total Return Benchmark"

                pdf_exec_en = (
                    f"The strategy combines a full Swiss Market Index replication with a structural "
                    f"Bitcoin allocation, harvesting equity dividends (net of the 35% non-reclaimable "
                    f"withholding tax) to fund a disciplined dollar-cost-averaging programme into "
                    f"digital assets. Over the backtest period it delivered a {strat_net_cagr*100:.1f}% "
                    f"net CAGR, {_verb_en} the SMI Total Return benchmark by {abs(_exc):.1f}% per annum, "
                    f"with a Sharpe ratio of {_fmt_num(strat_m.get('sharpe'))} and a maximum drawdown "
                    f"of {_fmt_pct(strat_m.get('max_drawdown'))}. A threshold-based rebalancing rule "
                    f"caps Bitcoin concentration to control risk."
                )
                pdf_exec_de = (
                    f"Die Strategie kombiniert eine vollständige SMI-Replikation mit einer strukturellen "
                    f"Bitcoin-Allokation und nutzt Aktiendividenden (nach Abzug der 35% nicht "
                    f"rückforderbaren Quellensteuer) zur Finanzierung eines disziplinierten "
                    f"Dollar-Cost-Averaging-Programms in digitale Vermögenswerte. Im Backtest-Zeitraum "
                    f"erzielte sie einen Netto-CAGR von {strat_net_cagr*100:.1f}% {_perf_de} — bei "
                    f"einem Sharpe Ratio von {_fmt_num(strat_m.get('sharpe'))} und einem maximalen "
                    f"Drawdown von {_fmt_pct(strat_m.get('max_drawdown'))}. Eine "
                    f"schwellenwertbasierte Rebalancing-Regel begrenzt die Bitcoin-Konzentration zur "
                    f"Risikokontrolle."
                )

                # ---------- IB-style fact-sheet add-ons ----------
                # Strategy Snapshot — the canonical IB fact box
                snapshot_data = [
                    ("sn_inception",  ts.index[0].strftime("%d %b %Y")),
                    ("sn_currency",   "CHF"),
                    ("sn_benchmark",  "SMI Total Return"),
                    ("sn_style",      "Multi-Asset (Equity + BTC)"),
                    ("sn_domicile",   "Switzerland"),
                    ("sn_frequency",  "Daily"),
                ]

                # Performance per Period (1M / 3M / 6M / YTD / 1Y / 3Y / ITD)
                _bench_series = bench["smi_tr"] if not bench.empty else None
                pdf_period_returns = compute_period_returns(
                    ts["total_value_net"], _bench_series
                )

                # Top 5 Drawdowns
                pdf_top_drawdowns = identify_top_drawdowns(
                    ts["total_value_net"], n=5, min_depth_pct=2.0
                )

                # DE->EN translation for two methodology values that arrive in
                # German from the UI selectboxes — keeps the PDF fully English
                # without touching the Streamlit UI labels.
                _PARAM_DE_EN = {
                    "Marktkapitalisierung (Approx. + 18% Cap)": "Market cap (approx., 18% cap)",
                    "Equal Weight (5 % je Titel)":              "Equal weight (5% per holding)",
                    "Quartalsweise":                            "Quarterly",
                    "Halbjährlich":                             "Semi-annual",
                    "Jährlich":                                 "Annual",
                    "Keine":                                    "None",
                }
                _weighting_method_en  = _PARAM_DE_EN.get(weighting_method, weighting_method)
                _rebalance_freq_en    = _PARAM_DE_EN.get(rebalance_freq,   rebalance_freq)
                _crystallization_en   = _PARAM_DE_EN.get(crystallization_freq, crystallization_freq)

                # ---- Renditezerlegung als PDF-Sektion ---------------------
                def _xtabs_smi(lang):
                    de = (lang == "de")
                    _a = ts.attrs.get("attribution", {})
                    if not _a:
                        return []
                    _yy = _a["years"]
                    def _pp(v):
                        return f"{(v / initial_capital) / _yy * 100:+.2f}"
                    labels = ([("Aktien-Kapitalwertentwicklung (SMI)", "equity_gain"),
                               ("Dividendenerträge (netto, nach 35% VSt)", "dividend_income"),
                               ("Bitcoin — Startallokation (Tag 1)", "btc_initial_gain"),
                               ("Bitcoin — dividendenfinanzierter DCA", "btc_dca_gain")]
                              if de else
                              [("Equity capital appreciation (SMI)", "equity_gain"),
                               ("Dividend income (net of 35% WHT)", "dividend_income"),
                               ("Bitcoin — initial allocation (day 1)", "btc_initial_gain"),
                               ("Bitcoin — dividend-funded DCA", "btc_dca_gain")])
                    rows = [[lab, f"{_a.get(k, 0.0):+,.0f}", _pp(_a.get(k, 0.0))]
                            for lab, k in labels]
                    rows.append([("Total brutto (= NAV − Startkapital)" if de else
                                  "Total gross (= NAV − initial capital)"),
                                 f"{_a['total_pnl_gross']:+,.0f}",
                                 _pp(_a["total_pnl_gross"])])
                    _ds = _a.get("dca_share")
                    _dstxt = "n/a" if _ds != _ds else f"{_ds*100:.1f}%"
                    _out = [{
                        "eyebrow": "08",
                        "title": "Renditezerlegung" if de else "Return Attribution",
                        "subtitle": (
                            f"DCA-Anteil am BTC-Gewinn: {_dstxt} — der Rest stammt aus der "
                            f"Startallokation vom ersten Tag. Vor Management- und "
                            f"Performance-Gebühren; Transaktionskosten sind in den "
                            f"jeweiligen Positionen enthalten."
                            if de else
                            f"DCA share of the BTC gain: {_dstxt} — the remainder comes from "
                            f"the day-1 initial allocation. Before management and performance "
                            f"fees; transaction costs are absorbed by the respective lines."),
                        "headers": (["Beitrag", "CHF", "%-Punkte p.a."] if de else
                                    ["Contribution", "CHF", "pp p.a."]),
                        "rows": rows,
                        "note": (
                            "Der DCA-Anteil misst, wie viel des Bitcoin-Gewinns aus dem "
                            "dividendenfinanzierten Mechanismus stammt und wie viel aus der "
                            "Startallokation. Er ist invers zum Einstiegsglück: je schlechter "
                            "der Einstiegszeitpunkt, desto grösser der Beitrag des DCA. Ein "
                            "tiefer Wert zeigt daher primär an, dass der Backtest-Zeitraum "
                            "für die Startallokation günstig lag."
                            if de else
                            "The DCA share measures how much of the Bitcoin gain came from the "
                            "dividend-funded mechanism versus the initial allocation. It is "
                            "inverse to entry luck: the worse the entry point, the larger the "
                            "DCA contribution. A low value therefore mainly indicates that the "
                            "backtest period was favourable for the initial allocation."),
                    }]
                    _sd = st.session_state.get("smi_rb_dist")
                    if _sd is not None and not _sd.empty:
                        _out.append({
                            "eyebrow": "09",
                            "title": ("Robustheit — Verteilung über rollierende Fenster"
                                      if de else
                                      "Robustness — Distribution across rolling windows"),
                            "subtitle": (
                                "Netto-CAGR je Startallokation über alle rollierenden "
                                "Fenster und alle Gebührenstufen."
                                if de else
                                "Net CAGR by initial allocation across all rolling windows "
                                "and all fee levels."),
                            "headers": ([("Startallokation" if de else "Initial allocation")]
                                        + list(_sd.columns)),
                            "rows": [[str(i)] + [f"{v:.2f}%" for v in _sd.loc[i].tolist()]
                                     for i in _sd.index],
                            "note": (
                                "Das Minimum ist KEIN Risikomass — die Datenreihe enthält kein "
                                "Fenster mit einem Bitcoin-Kollaps ohne Erholung. Das belastbare "
                                "Signal ist die Streuung: sie misst, wie stark das Ergebnis vom "
                                "Einstiegszeitpunkt abhängt."
                                if de else
                                "The minimum is NOT a risk measure — the sample contains no "
                                "window with a Bitcoin collapse without recovery. The meaningful "
                                "signal is the spread: it measures how strongly the outcome "
                                "depends on the entry point."),
                        })
                    return _out

                pdf_bytes = build_bilingual_tearsheet(
                    strategy_name="OAK Swiss Blue Chip / Bitcoin",
                    strategy_subtitle_de=(
                        "Disziplinierte SMI-Replikation mit struktureller BTC-Allokation, "
                        "dividendenfinanzierter DCA und schwellenwertbasiertem Risikomanagement."
                    ),
                    strategy_subtitle_en=(
                        "Disciplined SMI replication with structural BTC allocation, "
                        "dividend-funded DCA and threshold-based risk management."
                    ),
                    period_str=f"{ts.index[0].strftime('%Y-%m-%d')} to {ts.index[-1].strftime('%Y-%m-%d')}",
                    kpis_performance=[
                        ("Strategy (Net)", f"CHF {strategy_net:,.0f}"),
                        ("Net CAGR", f"{strat_net_cagr*100:.2f}%"),
                        ("SMI Total Return", f"CHF {smi_tr_final:,.0f}"),
                        ("Excess vs SMI TR", f"{excess_vs_tr*100:+.2f}% p.a."),
                    ],
                    kpis_risk=[
                        ("Sharpe Ratio", _fmt_num(strat_m.get("sharpe"))),
                        ("Sortino Ratio", _fmt_num(strat_m.get("sortino"))),
                        ("Max Drawdown", _fmt_pct(strat_m.get("max_drawdown"))),
                        ("Volatility", _fmt_pct(strat_m.get("vol_ann"))),
                    ],
                    fee_summary=[
                        ("Mgmt Fees", f"CHF {total_mgmt_fees:,.0f}"),
                        ("Perf Fees", f"CHF {total_perf_fees:,.0f}"),
                        ("Total Fees", f"CHF {fees_total:,.0f}"),
                        ("Fee Drag", f"{fee_drag*100:.2f}% p.a."),
                    ],
                    risk_table_headers=["Metric", "Strategy (Net)", "SMI Total Return", "SMI Price Index"],
                    risk_table_rows=risk_rows,
                    fee_table_headers=["Period", "Mgmt Fee", "Perf Fee", "Total Cost"],
                    fee_table_rows=fee_rows,
                    figures=pdf_figures,
                    params_summary=[
                        ("Allocation Framework",
                         ("OAK Yield Bridge (pure) — the satellite is funded exclusively "
                          "by net dividend income; the equity core is never sold"
                          if initial_btc_pct <= 0 else
                          f"OAK Yield Bridge with strategic initial allocation — "
                          f"{initial_btc_pct*100:.0f}% of capital is allocated to Bitcoin "
                          f"on day 1; net dividend income funds all further purchases")),
                        ("DCA Share of BTC Gain",
                         ("n/a" if ts.attrs.get("attribution", {}).get("dca_share")
                          != ts.attrs.get("attribution", {}).get("dca_share")
                          else f"{ts.attrs['attribution']['dca_share']*100:.1f}% "
                               f"(dividend-funded vs. day-1 lump sum)")),
                        ("Initial Capital", f"CHF {initial_capital:,.0f}"),
                        ("Initial Allocation", f"{(1-initial_btc_pct)*100:.0f}% Equity / {initial_btc_pct*100:.0f}% BTC"),
                        ("BTC Upper Threshold", f"{upper_threshold*100:.0f}%"),
                        ("BTC Target after Rebalance", f"{target_btc_pct*100:.0f}%"),
                        ("Equity Weighting", _weighting_method_en),
                        ("Rebalancing Frequency", _rebalance_freq_en),
                        ("DCA Window", f"{dca_months} months per dividend"),
                        ("Transaction Cost", f"{tx_cost_bps:.0f} bps per trade"),
                        ("Dividend Withholding Tax", f"{int(WITHHOLDING_TAX*100)}% (non-reclaimable, AMC)"),
                        ("Management Fee", f"{mgmt_fee_pct*100:.2f}% p.a."),
                        ("Performance Fee", f"{perf_fee_pct*100:.0f}% ({_crystallization_en})"),
                        ("Hurdle", f"{hurdle_type}, {hwm_hurdle_pct*100:.1f}% (Year 1)"),
                        ("Risk-Free Rate", f"{risk_free_rate*100:.2f}%"),
                    ],
                    universe_rows=universe_rows,
                    monthly_returns=pdf_monthly,
                    exec_summary_de=pdf_exec_de,
                    exec_summary_en=pdf_exec_en,
                    key_takeaways_de=pdf_takeaways_de,
                    key_takeaways_en=pdf_takeaways_en,
                    scatter_png=pdf_scatter,
                    snapshot_data=snapshot_data,
                    period_returns=pdf_period_returns,
                    top_drawdowns=pdf_top_drawdowns,
                    extra_tables_de=_xtabs_smi("de"),
                    extra_tables_en=_xtabs_smi("en"),
                )

                st.download_button(
                    "Download PDF Tearsheet",
                    data=pdf_bytes,
                    file_name=f"OAK_Swiss_BlueChip_BTC_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                )
                st.success("PDF generated. Click the download button above.")

                # Surface whether the embedded brand fonts loaded, or whether
                # the report silently fell back to Times/Helvetica (e.g. when
                # assets/fonts was not committed to the repo).
                _fs = get_font_status()
                if _fs["crimson_pro"] and _fs["work_sans"]:
                    st.caption("✓ Brand fonts embedded: Crimson Pro + Work Sans.")
                else:
                    missing = []
                    if not _fs["crimson_pro"]:
                        missing.append("Crimson Pro")
                    if not _fs["work_sans"]:
                        missing.append("Work Sans")
                    st.warning(
                        "⚠ PDF is using the Times/Helvetica fallback — "
                        f"{', '.join(missing)} not found. "
                        f"Expected TTFs in `{_fs['fonts_dir']}` "
                        f"(directory {'exists' if _fs['dir_exists'] else 'is missing'}). "
                        "Commit the `assets/fonts/` folder to the repo to embed the brand fonts."
                    )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    footer()

else:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("### Strategy Logic")
        st.markdown(f"""
<div style='color:{OAK_CREAM_DIM}; line-height:1.7;'>
<strong style='color:{OAK_CREAM};'>Initial allocation.</strong> Capital split at day 0
between Equity Sleeve (SMI 20 by chosen weighting) and Bitcoin Sleeve (target % via spot purchase).<br><br>
<strong style='color:{OAK_CREAM};'>Dividend harvesting.</strong> Each dividend collected in CHF,
reduced by the 35% Swiss withholding tax (non-reclaimable in the AMC wrapper, so only the
net 65% is available), and split into N monthly tranches (DCA), bought at month-end into BTC via USDCHF FX.<br><br>
<strong style='color:{OAK_CREAM};'>Equity rebalancing.</strong> Quarterly return to target SMI weights —
mirroring the SIX index review cycle.<br><br>
<strong style='color:{OAK_CREAM};'>Risk management — Threshold rebalance.</strong>
At each month-end, if BTC sleeve exceeds upper threshold, sell down to target weight.
Proceeds reinvested across SMI titles by current target weights.<br><br>
<strong style='color:{OAK_CREAM};'>Result.</strong> Long Swiss equity income + structural BTC exposure
with mechanical profit-taking on outsized crypto appreciation.<br><br>
<strong style='color:{OAK_CREAM};'>Benchmarks.</strong> Strategy (net of fees) is compared against
<em>SMI Total Return</em> (dividends, net of the same 35% withholding tax, reinvested into the
same stocks, quarterly rebalanced) and the <em>SMI Price Index</em> (no dividend reinvestment).<br><br>
<strong style='color:{OAK_CREAM};'>Fees.</strong> Management fee accrued daily, performance fee
charged annually on returns above a High Water Mark with a Year-1 hurdle. All risk metrics
computed on the net-of-fees series.
</div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("### Default Parameters")
        st.markdown(f"""
<div style='color:{OAK_CREAM_DIM}; line-height:1.9;'>
<strong style='color:{OAK_SAGE};'>Initial Allocation</strong><br>
85% SMI · 15% BTC<br><br>
<strong style='color:{OAK_SAGE};'>Upper Threshold</strong><br>
25% — sell-down trigger<br><br>
<strong style='color:{OAK_SAGE};'>Target</strong><br>
15% — post-rebalance weight<br><br>
<strong style='color:{OAK_SAGE};'>DCA Window</strong><br>
12 months per dividend<br><br>
<strong style='color:{OAK_SAGE};'>Rebalancing</strong><br>
Quarterly (SMI) · Monthly (BTC check)
</div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Configure parameters in the sidebar, then click **Run Backtest** to begin analysis.")
    footer()
