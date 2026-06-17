"""
OAK RE/BTC — AMC Backtesting (Wohnimmobilien Schweiz + strukturelle BTC-Allokation)

Konzept:
  * RE-Sleeve: parametrisches Direktimmobilien-Modell für CH-Wohnliegenschaften.
      - Kapitalwert-Entwicklung: SNB-Immobilienpreisindex (data.snb.ch, Cube
        'plimoinchq', quartalsweise, täglich interpoliert) — echte Daten.
      - Netto-Eigenkapitalrendite: eigene Parametrik (Bruttomietrendite,
        Leerstand, Bewirtschaftung, Hypothek LTV/Zins/Amortisation).
  * BTC-Sleeve: identische Mechanik wie 'SMI Income meets Digital Assets' —
      Netto-Mieterträge fliessen monatlich via DCA in BTC; Threshold-Regel
      verkauft auf Zielquote zurück, sobald die BTC-Quote den Cap überschreitet.
      Verkaufserlöse + Über-Cap-Miete sammeln sich als CHF-Cash-Puffer; an
      Quartalsenden wird der Puffer blockweise in Immobilien reinvestiert,
      sobald die Blockgrösse erreicht ist (Rest bleibt Cash).
  * AMC-Schicht: identisch (Mgmt-Fee täglich, Perf-Fee je Periode auf HWM).

WICHTIG — Methodischer Charakter:
  Der RE-Teil ist eine PARAMETRISCHE SIMULATION auf Basis eines geglätteten
  Bewertungsindex, kein marktdatenbasierter Backtest. Volatilität, Sharpe und
  Drawdowns sind NICHT mit kotierten Strategien (z.B. dem SMI-Produkt)
  vergleichbar. Einzig der BTC-Sleeve basiert auf echten Marktpreisen.
"""

import base64
import io
from datetime import date

from dateutil.relativedelta import relativedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from pdf_report import (build_bilingual_tearsheet,
                        render_line_chart, render_bar_chart,
                        render_scatter_chart, render_stacked_bar_chart,
                        compute_period_returns, identify_top_drawdowns,
                        get_font_status)

st.set_page_config(page_title="OAK RE/BTC — AMC Backtesting",
                   page_icon="🏠", layout="wide")

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


def load_logo_base64():
    here = Path(__file__).parent.parent / "assets"
    for name in ("oakwood_logo.png", "logo.png", "OAKWOOD-CAPITAL-LOGO-DARK.png"):
        path = here / name
        if path.exists():
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
    return None


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




# ===========================================================================
# REUSED-FROM-SMI-PAGE BLOCK (injected verbatim by build script)
# ===========================================================================
def _clean_index(obj):
    if obj is None:
        return obj
    if hasattr(obj, "empty") and obj.empty:
        return obj
    if hasattr(obj.index, "tz") and obj.index.tz is not None:
        obj.index = obj.index.tz_localize(None)
    obj.index = pd.to_datetime(obj.index).normalize()
    obj = obj[~obj.index.duplicated(keep="last")]
    obj = obj.sort_index()
    return obj


def _to_series(x):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] >= 1:
            return x.iloc[:, 0]
    return x



@st.cache_data(ttl=21600, show_spinner=False)
def fetch_series(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False,
                     auto_adjust=False, threads=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        col = ("Adj Close", ticker) if ("Adj Close", ticker) in df.columns else df.columns[0]
        s = df[col]
    else:
        s = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    s = _to_series(s)
    s = _clean_index(s)
    return s.dropna()



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




# ===========================================================================
# SNB data layer — Immobilienpreisindizes (Cube plimoinchq)
# ===========================================================================
SNB_CUBE = "plimoinchq"
SNB_CSV_URL = f"https://data.snb.ch/api/cube/{SNB_CUBE}/data/csv/de"
SNB_DIM_URL = f"https://data.snb.ch/api/cube/{SNB_CUBE}/dimensions/de"
SNB_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/csv,application/json;q=0.9,*/*;q=0.8",
}


def _parse_snb_csv(text):
    """Parse an SNB data-portal CSV. Real-world format (live-verified on the
    sibling cube snbfxtr — the API envelope is uniform across cubes):

        \ufeff"CubeId";"plimoinchq"
        "PublishingDate";"2026-05-29 09:00"
        <leer>
        "Date";"D0";"Value"
        "2022-Q4";"XYZ";"123.4"

    i.e. UTF-8 BOM, ALLE Felder in Anführungszeichen, Metadaten-Zeilen vor
    dem Header. Erkennung dynamisch: erste Zeile, die entquotet mit 'Date;'
    beginnt. pd.read_csv übernimmt das Quote-Handling der Datenzeilen."""
    text = text.lstrip("\ufeff")
    lines = text.splitlines()
    header_i = None
    for i, ln in enumerate(lines):
        if ln.replace('"', "").strip().startswith("Date;"):
            header_i = i
            break
    if header_i is None:
        raise ValueError("SNB-CSV: keine 'Date;…'-Headerzeile gefunden — "
                         f"Antwort beginnt mit: {lines[:3]!r}")
    body = "\n".join(lines[header_i:])
    df = pd.read_csv(io.StringIO(body), sep=";")
    df.columns = [str(c).replace('"', "").strip() for c in df.columns]
    if "Value" not in df.columns:
        raise ValueError(f"SNB-CSV: 'Value'-Spalte fehlt — Spalten: {list(df.columns)}")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])
    return df


def _extract_dim_items(node, out):
    """Recursively collect {id: name} from the /dimensions JSON
    ({"dimensions":[{"id","name","dimensionItems":[{"id","name"},…]}]},
    tolerant of nested hierarchies)."""
    if isinstance(node, dict):
        nid, name = node.get("id"), node.get("name")
        if nid is not None and name is not None:
            out[str(nid)] = str(name)
        for v in node.values():
            _extract_dim_items(v, out)
    elif isinstance(node, list):
        for v in node:
            _extract_dim_items(v, out)


def _quarter_to_timestamp(qstr):
    """'2020-Q3' -> Timestamp of quarter END (valuation effective date)."""
    qstr = str(qstr).strip()
    per = pd.Period(qstr.replace("-Q", "Q"), freq="Q")
    return per.to_timestamp(how="end").normalize()


def snb_series_catalog(df, code_map=None):
    """From a parsed SNB frame, build {series_label: quarterly pd.Series}.
    Die SNB-CSV enthält Dimensions-CODES (z.B. 'T0'); code_map (aus dem
    /dimensions-Endpoint) übersetzt sie in Klartext-Labels. Ohne Mapping
    werden die Codes angezeigt — funktional, nur weniger lesbar."""
    code_map = code_map or {}
    dim_cols = [c for c in df.columns if c not in ("Date", "Value")]
    out = {}
    if dim_cols:
        for key, sub in df.groupby(dim_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            label = " · ".join(code_map.get(str(k), str(k))
                               for k in key if pd.notna(k) and str(k).strip())
            s = pd.Series(sub["Value"].values,
                          index=[_quarter_to_timestamp(d) for d in sub["Date"]])
            out[label or "Serie"] = s.sort_index()
    else:
        s = pd.Series(df["Value"].values,
                      index=[_quarter_to_timestamp(d) for d in df["Date"]])
        out["Immobilienpreisindex"] = s.sort_index()
    # keep only sufficiently long series
    return {k: v for k, v in out.items() if len(v.dropna()) >= 8}


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_snb_catalog():
    """Fetch the SNB cube (CSV) plus the dimension labels (JSON) and return
    the series catalog with readable labels. Cached for a day."""
    r = requests.get(SNB_CSV_URL, headers=SNB_HEADERS, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} von data.snb.ch — "
                           f"Antwort: {r.text[:200]!r}")
    df = _parse_snb_csv(r.text)
    code_map = {}
    try:  # Labels sind kosmetisch — Codes funktionieren auch
        rd = requests.get(SNB_DIM_URL, headers=SNB_HEADERS, timeout=30)
        if rd.status_code == 200:
            _extract_dim_items(rd.json(), code_map)
    except Exception:
        pass
    return snb_series_catalog(df, code_map)


def interpolate_quarterly_to_daily(qseries, daily_index):
    """Linear interpolation of a quarterly valuation index onto a daily
    grid (with edge fill). NOTE: interpolation smooths the path — this is a
    valuation series, not market pricing; documented in the disclosures."""
    s = qseries.dropna().sort_index()
    combined = s.reindex(s.index.union(daily_index)).interpolate(method="time")
    out = combined.reindex(daily_index).ffill().bfill()
    return out


# ===========================================================================
# Engine — parametric residential RE + BTC (DCA from net rents, threshold)
# ===========================================================================
def run_re_btc(prop_index_daily, btc_chf, params):
    """Daily simulation — unlevered residential property + BTC + CHF cash,
    with band-based rebalancing of the net rental income.

    params (dict):
      initial_capital      total CHF at t0
      initial_btc_pct      fraction of capital in BTC at t0
      net_yield            net rental yield p.a. on the property COST BASIS —
                           capital actually invested at purchase (initial
                           allocation + reinvested blocks), NOT the SNB
                           valuation. Constant per CHF invested; total monthly
                           rent steps up when cash is reinvested into property.
      lower_threshold      BTC weight below which the boosted rate applies
      upper_threshold      BTC weight cap
      base_invest_rate     share of accumulated net rent invested into BTC
      boost_invest_rate    share invested while BTC weight < lower_threshold
      rent_to_btc_freq     "M" (monthly) or "Q" (quarterly) — DCA dates
      btc_to_cash_freq     "M" or "Q" — sell-rule check dates
      sell_on_upper        bool — sell BTC down to the upper threshold into
                           CHF cash on check dates when the weight exceeds it
      cash_reinvest_block  CHF block size for reinvesting the cash buffer into
                           property. Checked at quarter-ends: when cash >= block,
                           whole blocks are swept into prop_units, the remainder
                           stays as cash. Omit/inf = never (pure one-way buffer).
      tx_cost_bps          transaction cost on BTC trades (bps)

    Zone rule applied on each DCA date (pre-trade BTC weight w):
        w < lower_threshold   ->  boost_invest_rate
        w > upper_threshold   ->  0%  (rent stays in cash)
        otherwise             ->  base_invest_rate
    Non-invested rent and all sale proceeds accumulate as CHF cash (0%
    interest). At quarter-ends the buffer is reinvested into property in fixed
    blocks of cash_reinvest_block once it reaches that size; the remainder stays.

    Returns a daily DataFrame with columns: total_value, re_value, btc_value,
    cash, property_value, net_cf_monthly (month-end rows), btc_buys, btc_sells,
    re_reinvest (cash reinvested into property on the sweep date).
    """
    idx = btc_chf.index.intersection(prop_index_daily.index).sort_values()
    if len(idx) < 30:
        return pd.DataFrame()
    btc = btc_chf.reindex(idx).ffill()
    pidx = prop_index_daily.reindex(idx).ffill()

    cap = float(params["initial_capital"])
    tx = float(params.get("tx_cost_bps", 0.0)) / 10000.0

    btc_chf0 = cap * float(params["initial_btc_pct"])
    cash0 = cap * float(params.get("initial_cash_pct", 0.0))   # fee-reserve cushion
    btc_units = (btc_chf0 * (1 - tx)) / btc.iloc[0] if btc_chf0 > 0 else 0.0
    prop0 = cap - btc_chf0 - cash0                  # rest goes into property
    prop_units = prop0 / pidx.iloc[0]
    prop_value_0 = prop0            # initial property capital invested at t0
    prop_cost_basis = prop_value_0  # grows when cash is reinvested into property;
                                    # the static net yield applies to this basis

    cash = cash0      # CHF buffer: initial cushion + rent remainders + sale proceeds
    rent_pool = 0.0   # rent accumulated since the last DCA date
    fee_floor = cash0 # min cash protected from the block reinvest (the cushion)
    fee_debt = 0.0    # accrued mgmt fee that could not be funded (should ~never)

    ny = float(params["net_yield"])
    lo = float(params["lower_threshold"])
    up = float(params["upper_threshold"])
    base_r = float(params["base_invest_rate"])
    boost_r = float(params["boost_invest_rate"])
    f_dca = params.get("rent_to_btc_freq", "M")
    f_sell = params.get("btc_to_cash_freq", "Q")
    sell_on = bool(params.get("sell_on_upper", True))
    reinv_blk = float(params.get("cash_reinvest_block", float("inf")))
    # Management fee charged in-engine as a real cash outflow (monthly default),
    # funded by the waterfall cash -> BTC sale (property is never touched).
    mgmt_fee = float(params.get("mgmt_fee", 0.0))
    mgmt_freq = params.get("mgmt_fee_freq", "M")   # "M" or "Q"
    # Performance fee — also paid for real from liquidity (cash -> BTC) at each
    # crystallization date, on the NAV gain above the hurdle-grown HWM.
    perf_fee = float(params.get("perf_fee", 0.0))
    hwm_hurdle = float(params.get("hwm_hurdle", 0.0))
    hurdle_type = params.get("hurdle_type", "Hard Hurdle")
    cryst_freq = params.get("crystallization_freq", "Quarterly")
    if cryst_freq == "Annual":
        cryst_months = {12}
    elif cryst_freq == "Semi-Annual":
        cryst_months = {6, 12}
    else:
        cryst_months = {3, 6, 9, 12}
    hwm = cap                       # high-water mark (post-fee NAV highs)
    prev_cryst_date = idx[0]        # for pro-rata hurdle on partial periods
    total_mgmt = 0.0
    total_perf = 0.0
    # Reinvest cash floor as a fraction of CURRENT AuM (grows with the book) —
    # cash below this level is protected from the block reinvest so the fee
    # cushion is never swept into property.
    floor_pct = float(params.get("reinvest_floor_pct", 0.0))

    rows = []
    # observations per year on this (≈365) calendar, for monthly/quarterly fee
    _spanyr = max((idx[-1] - idx[0]).days / 365.25, 1e-9)
    for i, d in enumerate(idx):
        p_val = prop_units * pidx.loc[d]
        b_val = btc_units * btc.loc[d]
        net_cf = np.nan
        buys = 0.0
        sells = 0.0
        reinv = 0.0
        fee_paid = 0.0
        fee_btc_sold = 0.0
        fee_from_cash = 0.0
        fee_from_btc = 0.0

        is_me = (i == len(idx) - 1) or (idx[i + 1].to_period("M") != d.to_period("M"))
        is_qe = is_me and d.month in (3, 6, 9, 12)

        # ---- month-end: net rent accrues into the rent pool ----------------
        if is_me:
            net_cf = ny / 12.0 * prop_cost_basis
            rent_pool += net_cf

        # ---- DCA date: allocate accumulated rent per the zone rule ---------
        if (is_me and f_dca == "M") or (is_qe and f_dca == "Q"):
            nav = p_val + b_val + cash + rent_pool
            w = b_val / nav if nav > 0 else 0.0
            rate = boost_r if w < lo else (0.0 if w > up else base_r)
            invest = rate * rent_pool
            if invest > 0:
                btc_units += (invest * (1 - tx)) / btc.loc[d]
                buys = invest
            cash += rent_pool - invest
            rent_pool = 0.0
            b_val = btc_units * btc.loc[d]

        # ---- sell-rule date: trim BTC back to the upper threshold ----------
        if sell_on and ((is_me and f_sell == "M") or (is_qe and f_sell == "Q")):
            nav = p_val + b_val + cash + rent_pool
            if nav > 0 and b_val / nav > up:
                sell_chf = b_val - up * nav
                btc_units -= sell_chf / btc.loc[d]
                cash += sell_chf * (1 - tx)
                sells = sell_chf
                b_val = btc_units * btc.loc[d]

        # ---- fee waterfall helper: pay `amount` from cash, then by selling
        #      BTC (net of tx); property is never sold. Returns unpaid shortfall.
        def _pay_from_liquidity(amount):
            nonlocal cash, btc_units, b_val, fee_paid, fee_btc_sold
            nonlocal fee_from_cash, fee_from_btc
            if amount <= 1e-9:
                return 0.0
            from_cash = min(cash, amount)
            cash -= from_cash
            fee_paid += from_cash
            fee_from_cash += from_cash
            short = amount - from_cash
            if short > 1e-9 and b_val > 0:
                gross_sell = min(b_val, short / (1 - tx))
                btc_units -= gross_sell / btc.loc[d]
                fee_btc_sold += gross_sell
                proceeds = gross_sell * (1 - tx)
                fee_paid += proceeds
                fee_from_btc += proceeds
                short -= proceeds
                b_val = btc_units * btc.loc[d]
            return max(short, 0.0)

        # ---- management fee (real cash outflow): monthly or quarterly -------
        #      Charged on full AuM, funded cash-first then BTC; any residual
        #      after BTC is exhausted accrues as fee_debt (practically never).
        fee_due = ((is_me and mgmt_freq == "M") or (is_qe and mgmt_freq == "Q"))
        if fee_due and mgmt_fee > 0:
            per_year = 12.0 if mgmt_freq == "M" else 4.0
            aum = p_val + b_val + cash + rent_pool
            fee_amt = aum * (mgmt_fee / per_year) + fee_debt
            fee_debt = 0.0
            paid_before = fee_paid
            fee_debt = _pay_from_liquidity(fee_amt)
            total_mgmt += (fee_paid - paid_before)   # actually-paid this date

        # ---- performance fee (real cash outflow) at crystallization dates ---
        is_cryst = is_me and (d.month in cryst_months)
        if (is_cryst or i == len(idx) - 1) and perf_fee > 0:
            nav_now = p_val + b_val + cash + rent_pool
            _frac = max((d - prev_cryst_date).days, 0) / 365.25
            hurdle_threshold = hwm * (1.0 + hwm_hurdle * _frac)
            excess = 0.0
            if hurdle_type == "No Hurdle (HWM only)":
                if nav_now > hwm:
                    excess = nav_now - hwm
            elif hurdle_type == "Soft Hurdle":
                if nav_now > hurdle_threshold:
                    excess = nav_now - hwm
            else:  # Hard Hurdle
                if nav_now > hurdle_threshold:
                    excess = nav_now - hurdle_threshold
            if excess > 1e-9:
                perf_amt = excess * perf_fee
                _short = _pay_from_liquidity(perf_amt)
                total_perf += (perf_amt - _short)
            # update HWM to post-fee NAV high
            nav_after = p_val + b_val + cash + rent_pool
            if nav_after > hwm:
                hwm = nav_after
            prev_cryst_date = d

        # ---- cash sweep (quarter-end only): reinvest the buffer into property
        #      in fixed blocks, but only cash ABOVE a dynamic fee-reserve floor
        #      (a % of current AuM, so the cushion grows with the book) --------
        _aum_now = p_val + b_val + cash + rent_pool
        dyn_floor = max(fee_floor, floor_pct * _aum_now)
        investable = cash - dyn_floor
        if is_qe and reinv_blk > 0 and investable >= reinv_blk and pidx.loc[d] > 0:
            n_blk = int(investable // reinv_blk)
            reinv = n_blk * reinv_blk
            prop_units += reinv / pidx.loc[d]
            cash -= reinv
            prop_cost_basis += reinv      # more property -> higher net rent
            p_val = prop_units * pidx.loc[d]

        rows.append({
            "date": d,
            "total_value": p_val + b_val + cash + rent_pool,
            "re_value": p_val,
            "btc_value": b_val,
            "cash": cash + rent_pool,
            "property_value": p_val,
            "net_cf_monthly": net_cf,
            "btc_buys": buys,
            "btc_sells": sells,
            "re_reinvest": reinv,
            "mgmt_fee_paid": fee_paid,
            "fee_btc_sold": fee_btc_sold,
            "fee_from_cash": fee_from_cash,
            "fee_from_btc": fee_from_btc,
            "fee_debt": fee_debt,
            "cash_floor": max(fee_floor, floor_pct * (p_val + b_val + cash + rent_pool)),
        })

    out = pd.DataFrame(rows).set_index("date")
    out.attrs["total_mgmt"] = total_mgmt
    out.attrs["total_perf"] = total_perf
    out.attrs["fee_debt_final"] = fee_debt
    return out


def run_re_only(prop_index_daily, ref_index, params):
    """Benchmark: identical unlevered property model WITHOUT BTC — the full
    net rental income is reinvested into additional property units each
    month. Isolates the contribution of the BTC/cash rebalancing sleeve."""
    idx = ref_index.intersection(prop_index_daily.index).sort_values()
    pidx = prop_index_daily.reindex(idx).ffill()
    cap = float(params["initial_capital"])
    ny = float(params["net_yield"])
    prop_units = cap / pidx.iloc[0]
    cost_basis = cap   # grows as net rent is reinvested into property
    vals = []
    for i, d in enumerate(idx):
        p_val = prop_units * pidx.loc[d]
        is_me = (i == len(idx) - 1) or (idx[i + 1].to_period("M") != d.to_period("M"))
        if is_me:
            rent = ny / 12.0 * cost_basis
            prop_units += rent / pidx.loc[d]
            cost_basis += rent       # reinvested rent raises the basis -> compounding
            p_val = prop_units * pidx.loc[d]
        vals.append(p_val)
    return pd.Series(vals, index=idx)
# ===========================================================================
# UI
# ===========================================================================
if logo_b64:
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="Oakwood Capital"/>'
else:
    logo_html = ('<span style="color:#F5F5F1; font-family:Cormorant Garamond, serif; '
                 'font-size:28px;">Oakwood Capital</span>')

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
    f"line-height:1.1;'>OAK RE/BTC</h1>",
    unsafe_allow_html=True
)
st.markdown(
    f"<p style='color:{OAK_CREAM_DIM}; font-size:15px; margin-top:0; max-width: 820px;'>"
    "Swiss residential real estate with a structural Bitcoin allocation, rent-funded "
    "DCA and threshold-based risk management. Parametric simulation on the SNB "
    "residential price index."
    "</p>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("## Parameters")

    st.markdown("### Stress-Test Scenarios")
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
        if _sc_cols[_i % 2].button(_label, use_container_width=True, key=f"re_sc_{_i}"):
            st.session_state["re_scenario_start"] = _s
            st.session_state["re_scenario_end"] = _e
            st.session_state["re_btc_has_run"] = True  # auto-show results

    st.markdown("### Backtest Period")
    _default_start = st.session_state.get("re_scenario_start", date(2018, 1, 1))
    _default_end = st.session_state.get("re_scenario_end", date.today())
    start_date = st.date_input("Startdatum", value=_default_start,
                               min_value=date(2010, 1, 1),
                               max_value=date.today() - relativedelta(months=6))
    end_date = st.date_input("Enddatum", value=_default_end,
                             min_value=start_date + relativedelta(months=6),
                             max_value=date.today())
    initial_capital = st.number_input("Anfangskapital (CHF)", min_value=10_000,
                                      max_value=10_000_000_000, value=1_000_000, step=10_000)

    st.markdown("### Allocation")
    initial_btc_pct = st.slider("Initial BTC Allokation (%)", 0, 50, 15, 1) / 100.0
    initial_cash_pct = st.slider("Initiale Cash-Reserve (%)", 0, 25, 5, 1,
                                 help="Liquiditätspolster bei t0, aus dem die "
                                      "Management Fee bezahlt wird, damit die "
                                      "volle Nettomiete ins BTC-Band fliesst. "
                                      "Wird vom Block-Reinvest geschützt (Floor). "
                                      "Reicht es nicht, wird BTC verkauft.") / 100.0
    reinvest_floor_pct = st.slider("Reinvest Cash-Floor (% des AuM)", 0, 20, 5, 1,
                                   help="Mindest-Cash als Anteil des aktuellen "
                                        "AuM, der vom 3.0-Mio-Block-Reinvest "
                                        "ausgenommen bleibt — das Fee-Polster "
                                        "wächst so mit dem Portfolio mit.") / 100.0
    lower_threshold = st.slider("Lower BTC Threshold (%)", 0, 40, 10, 1) / 100.0
    upper_threshold = st.slider("Upper BTC Threshold (%)", 5, 75, 25, 1) / 100.0
    if initial_btc_pct + initial_cash_pct >= 1.0:
        st.error("Initial BTC + Cash muss < 100% sein (Rest geht in Immobilien).")
        st.stop()
    if lower_threshold >= upper_threshold:
        st.error("Lower Threshold muss kleiner als Upper Threshold sein.")
        st.stop()

    st.markdown("### Property Sleeve")
    net_yield = st.slider("Nettomietrendite (% p.a.)", 0.5, 6.0, 3.0, 0.1,
                          help="Extern vorberechnet — nach Leerstand, "
                               "Bewirtschaftung, Unterhalt und Finanzierung. "
                               "Bezogen auf das investierte Immobilienkapital "
                               "(Anfangsallokation + reinvestierte Blöcke), "
                               "nicht auf die SNB-Bewertung.") / 100.0
    snb_catalog, snb_error = {}, None
    try:
        snb_catalog = fetch_snb_catalog()
    except Exception as e:  # network/API issue -> manual fallback below
        snb_error = str(e)
    if snb_error:
        st.warning("SNB-API nicht erreichbar — CSV von data.snb.ch "
                   "(Cube plimoinchq) hochladen.")
        with st.expander("Fehlerdetails"):
            st.code(snb_error)
    if not snb_catalog:
        up = st.file_uploader("SNB-CSV (manueller Fallback)", type=["csv"])
        if up is not None:
            try:
                snb_catalog = snb_series_catalog(
                    _parse_snb_csv(up.getvalue().decode("utf-8-sig")))
                st.success(f"{len(snb_catalog)} Serien geladen.")
            except Exception as e:
                st.error(f"CSV konnte nicht geparst werden: {e}")
    if snb_catalog:
        _labels = sorted(snb_catalog.keys())
        _default_i = next((i for i, l in enumerate(_labels)
                           if "wohnliegenschaft" in l.lower()
                           or "mehrfamilien" in l.lower()), 0)
        series_label = st.selectbox("SNB-Indexserie (Kapitalwert)", _labels,
                                    index=_default_i)
        snb_q = snb_catalog[series_label]
        st.caption(f"{snb_q.index[0]:%Y-%m} – {snb_q.index[-1]:%Y-%m} · "
                   f"{len(snb_q)} Quartale · täglich interpoliert")
    else:
        series_label, snb_q = None, None

    st.markdown("### Rent Allocation")
    base_invest_rate = st.slider("Basis-Investitionsrate der Nettomiete (%)",
                                 0, 100, 50, 5) / 100.0
    boost_invest_rate = st.slider("Investitionsrate unter Lower Threshold (%)",
                                  0, 100, 100, 5) / 100.0
    rent_to_btc_freq = st.selectbox("Miete → BTC Allokation",
                                    ["monatlich", "quartalsweise"], index=0)
    btc_to_cash_freq = st.selectbox("BTC → Cash Allokation",
                                    ["monatlich", "quartalsweise"], index=1)
    sell_on_upper = st.checkbox(
        "Sell BTC to CHF cash on rebalancing dates whenever BTC weight is "
        "above the upper threshold.", value=True,
        help="Verkaufserlöse + Über-Cap-Miete sammeln sich als CHF-Cash; "
             "ab der unten gesetzten Schwelle wird der Puffer in Immobilien "
             "reinvestiert.")
    cash_reinvest_block = st.number_input(
        "Cash → Immobilien Reinvest-Block (CHF)",
        min_value=0, value=3_000_000, step=100_000,
        help="Quartalsweise geprüft: erreicht der Cash-Puffer am Quartalsende "
             "diese Blockgrösse, wird ein ganzer Block (oder mehrere) in "
             "Immobilien reinvestiert (kostenlos, parametrisches Modell); der "
             "Rest bleibt als Cash liegen. 0 = deaktiviert (reiner "
             "Einbahn-Cash-Puffer).")

    st.markdown("### Risk Analytics")
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 1.0, 0.25,
                               help="Annualisiert. Default ~1% entspricht "
                                    "historischem CHF/SARON-Durchschnitt.") / 100.0

    st.markdown("### Costs & Fees")
    tx_cost_bps = st.slider("Transaction Cost (bps per trade)", 0, 50, 10, 1,
                            help="Auf das gehandelte BTC-Volumen je Trade. "
                                 "10 bps = 0.10%.")
    mgmt_fee = st.slider("Management Fee (% p.a.)", 0.0, 3.0, 1.5, 0.05,
                         help="Echter Cash-Abzug auf das gesamte AuM: zuerst "
                              "aus dem Cash-Puffer, sonst durch BTC-Verkauf "
                              "(Immobilie nie). Die Nettomiete bleibt voll im "
                              "BTC-Band.") / 100.0
    mgmt_fee_freq_label = st.selectbox("Management Fee Verbuchung",
                                       ["monatlich", "quartalsweise"], index=0)
    mgmt_fee_freq = "M" if mgmt_fee_freq_label == "monatlich" else "Q"
    perf_fee = st.slider("Performance Fee (%)", 0, 30, 15, 1,
                         help="Charged on gains above the High Water Mark.") / 100.0
    hurdle_type = st.selectbox("Hurdle Type",
                               ["Hard Hurdle", "Soft Hurdle", "No Hurdle (HWM only)"],
                               index=0)
    hurdle = st.slider("Hurdle Rate Year 1 (%)", 0.0, 15.0, 5.0, 0.5,
                       help="Jahres-Hurdle vor Performance-Fee im ersten Jahr; "
                            "danach gilt die HWM.") / 100.0
    crystallization_freq = st.selectbox("Performance Fee Crystallization",
                                        ["Quarterly", "Semi-Annual", "Annual"], index=0)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Run Backtest", type="primary", use_container_width=True,
                        disabled=(lower_threshold >= upper_threshold))
    if run_btn:
        st.session_state["re_btc_has_run"] = True

    st.markdown(
        f"<div style='font-size:10px; color:{OAK_SAGE_DIM}; text-transform:uppercase; "
        f"letter-spacing:0.12em; padding-top:24px; margin-top:24px; "
        f"border-top:1px solid {OAK_BORDER};'>"
        "Data Source: SNB plimoinchq (quarterly) · Yahoo Finance<br>"
        "FX: USDCHF Spot · Parametric simulation"
        "</div>", unsafe_allow_html=True
    )

# --------------------------------------------------------------------------
# Data: SNB index (with manual-CSV fallback) + BTC in CHF
# --------------------------------------------------------------------------
if snb_q is None:
    st.info("Keine SNB-Daten verfügbar — Backtest kann nicht starten "
            "(siehe Property Sleeve in der Sidebar).")
    st.stop()

# Sticky gate (run button lives in the sidebar# Sticky gate (run button lives in the sidebar, like the SMI page): Streamlit
# buttons are only True on the rerun right after the click — the Sensitivity/
# Monte-Carlo buttons below trigger reruns, so we persist via session_state.
if not (run_btn or st.session_state.get("re_btc_has_run", False)):
    st.stop()

with st.spinner("Lade BTC/FX-Daten und simuliere…"):
    # Tickers/Konvertierung identisch zur SMI-Seite (dort nachweislich
    # funktionierend): USDCHF=X liefert CHF je USD -> BTC_CHF = BTC_USD * FX.
    btc_usd = fetch_series("BTC-USD", str(start_date), str(end_date))
    fx = fetch_series("USDCHF=X", str(start_date), str(end_date))
    problems = [name for name, s in
                [("BTC-USD", btc_usd), ("USDCHF=X", fx)] if s.empty]
    if problems:
        st.error(f"Keine Daten erhalten für: {', '.join(problems)}. "
                 "Häufigste Ursache ist ein temporäres Rate-Limit von Yahoo "
                 "Finance — nach 1–2 Minuten erneut versuchen. Besteht das "
                 "Problem, bitte prüfen, ob die SMI-Seite aktuell Daten lädt "
                 "(gleiche Quelle).")
        st.stop()
    btc_chf = (btc_usd * fx.reindex(btc_usd.index).ffill()).dropna()
    btc_chf = btc_chf[(btc_chf.index >= pd.Timestamp(start_date))
                      & (btc_chf.index <= pd.Timestamp(end_date))]

    # Market-listed real-estate fund benchmarks (real prices, incl. agio/disagio).
    # Total-return basis via Adj Close. Graceful if Yahoo rate-limits a symbol.
    fund_raw = {}
    for _flabel, _ftks, _fcol in [("UBS «Siat»", ["SIAT.SW"], "#5E83A3")]:
        for _ftk in _ftks:
            try:
                _s = fetch_series(_ftk, str(start_date), str(end_date))
                if not _s.empty:
                    fund_raw[_flabel] = (_s, _fcol)
                    break
            except Exception:
                pass

    prop_daily = interpolate_quarterly_to_daily(snb_q, btc_chf.index)

    params = dict(initial_capital=initial_capital, initial_btc_pct=initial_btc_pct,
                  net_yield=net_yield,
                  lower_threshold=lower_threshold, upper_threshold=upper_threshold,
                  base_invest_rate=base_invest_rate, boost_invest_rate=boost_invest_rate,
                  rent_to_btc_freq=("M" if rent_to_btc_freq == "monatlich" else "Q"),
                  btc_to_cash_freq=("M" if btc_to_cash_freq == "monatlich" else "Q"),
                  sell_on_upper=sell_on_upper,
                  cash_reinvest_block=cash_reinvest_block,
                  mgmt_fee=mgmt_fee, mgmt_fee_freq=mgmt_fee_freq,
                  perf_fee=perf_fee, hwm_hurdle=hurdle, hurdle_type=hurdle_type,
                  crystallization_freq=crystallization_freq,
                  initial_cash_pct=initial_cash_pct,
                  reinvest_floor_pct=reinvest_floor_pct,
                  tx_cost_bps=tx_cost_bps)

    ts = run_re_btc(prop_daily, btc_chf, params)
    if ts.empty:
        st.error("Simulation lieferte keine Daten (zu kurzer Überlappungszeitraum?).")
        st.stop()

    bench_re = run_re_only(prop_daily, ts.index, params)
    bench_index = (snb_q / snb_q.reindex(
        [ts.index[0]], method="ffill").iloc[0] * initial_capital)
    bench_index_daily = interpolate_quarterly_to_daily(bench_index, ts.index)

    # Fees are charged inside run_re_btc as real cash outflows (cash -> BTC
    # waterfall, property never sold), so ts["total_value"] is ALREADY net of
    # both management and performance fees. The net series equals gross here;
    # fee totals come from the engine. (apply_fees is retained for the SMI page.)
    net = ts["total_value"].copy()
    total_mgmt = float(ts.attrs.get("total_mgmt", 0.0))
    total_perf = float(ts.attrs.get("total_perf", 0.0))
    fee_debt_final = float(ts.attrs.get("fee_debt_final", 0.0))

    # Reconstruct a per-period fee table from the engine's paid-fee column so
    # the fee expander and PDF fee schedule keep working (cash fees per quarter).
    _fp = ts[["mgmt_fee_paid"]].copy()
    _fp = _fp[_fp["mgmt_fee_paid"] > 0]
    if not _fp.empty:
        _q = _fp["mgmt_fee_paid"].groupby(
            [_fp.index.year, _fp.index.quarter]).sum()
        fee_events = pd.DataFrame({
            "date": [pd.Timestamp(int(y), int(q) * 3, 1) for (y, q) in _q.index],
            "mgmt_fee": _q.values, "perf_fee": 0.0,
        })
    else:
        fee_events = pd.DataFrame()

# --------------------------------------------------------------------------
# KPIs & charts — section structure mirrors 1_SMI_Strategy.py
# --------------------------------------------------------------------------
st.markdown("## Performance Summary")

gross = net + (ts["mgmt_fee_paid"].cumsum()
               if "mgmt_fee_paid" in ts else 0.0)
# (gross ≈ net + cumulative fees actually paid; an illustrative pre-fee path
#  for the fee-drag KPI, since fees are now charged inside the engine.)
years = max((net.index[-1] - net.index[0]).days / 365.25, 1e-9)
net_cagr = (net.iloc[-1] / initial_capital) ** (1 / years) - 1
gross_cagr = (gross.iloc[-1] / initial_capital) ** (1 / years) - 1
re_cagr = (bench_re.iloc[-1] / initial_capital) ** (1 / years) - 1
idx_final = float(bench_index_daily.iloc[-1])
idx_cagr = (idx_final / initial_capital) ** (1 / years) - 1
fee_drag = gross_cagr - net_cagr
excess = net_cagr - re_cagr
m = compute_risk_metrics(net, risk_free_rate, base_value=initial_capital)
w_btc = ts["btc_value"].iloc[-1] / ts["total_value"].iloc[-1]
w_cash = ts["cash"].iloc[-1] / ts["total_value"].iloc[-1]

# Align fund benchmarks to the strategy window, rebased to initial_capital.
# Kept on the fund's OWN trading calendar (≈252 days/yr): force-filling onto
# the 365-day strategy grid would inject ~31% synthetic zero-return days
# (diluting vol/beta/correlation), and bfill would fabricate a flat segment
# if the fund's data starts after the backtest. Adaptive annualization in the
# metric functions handles the different calendar correctly.
fund_benches = []  # (label, scaled_series, color)
for _flabel, (_s, _fcol) in fund_raw.items():
    _w = _s[(_s.index >= net.index[0]) & (_s.index <= net.index[-1])].dropna()
    if len(_w) > 1 and _w.iloc[0] > 0:
        fund_benches.append((_flabel, _w / _w.iloc[0] * initial_capital, _fcol))
siat_series = fund_benches[0][1] if fund_benches else None
siat_m = (compute_risk_metrics(siat_series, risk_free_rate, base_value=initial_capital)
          if siat_series is not None else {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Strategy (Net of Fees)", fmt_chf(net.iloc[-1]),
          f"{(net.iloc[-1]/initial_capital - 1)*100:+.1f}%")
c2.metric("Strategy (Gross)", fmt_chf(gross.iloc[-1]),
          f"Fee drag: {fee_drag*100:.2f}% p.a.", delta_color="off")
c3.metric("RE only (same model)", fmt_chf(bench_re.iloc[-1]),
          f"{(bench_re.iloc[-1]/initial_capital - 1)*100:+.1f}%")
if siat_series is not None:
    c4.metric("UBS «Siat» (residential fund)", fmt_chf(siat_series.iloc[-1]),
              f"{(siat_series.iloc[-1]/initial_capital - 1)*100:+.1f}%")
else:
    c4.metric("UBS «Siat» (residential fund)", "n/a", "Yahoo-Daten nicht verfügbar")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Net CAGR", f"{net_cagr*100:.2f}%", f"after all fees · {years:.1f} years", delta_color="off")
c6.metric("Gross CAGR", f"{gross_cagr*100:.2f}%", "before fees", delta_color="off")
c7.metric("Excess vs RE only", f"{excess*100:+.2f}% p.a.", "net of fees")
c8.metric("BTC / Cash (today)", f"{w_btc*100:.1f}% / {w_cash*100:.1f}%",
          f"Band {lower_threshold*100:.0f}–{upper_threshold*100:.0f}%", delta_color="off")

c9, c10, c11, c12 = st.columns(4)
c9.metric("Total Mgmt Fees", fmt_chf(total_mgmt), f"{mgmt_fee*100:.2f}% p.a. on NAV", delta_color="off")
c10.metric("Total Perf Fees", fmt_chf(total_perf), f"{perf_fee*100:.0f}% × excess", delta_color="off")
c11.metric("Total Fees", fmt_chf(total_mgmt + total_perf),
           f"{(total_mgmt+total_perf)/initial_capital*100:.1f}% of initial capital", delta_color="off")
c12.metric("Net Rental Yield (input)", f"{net_yield*100:.1f}% p.a.",
           "pre-computed, on invested capital", delta_color="off")

# =====================================================================
# Portfolio Evolution vs. Benchmarks
# =====================================================================
st.markdown("## Portfolio Evolution vs. Benchmarks")
fig = go.Figure()
fig.add_trace(go.Scatter(x=net.index, y=net.values, name="Strategy (Net of Fees)",
                         line=dict(color=OAK_GOLD, width=3),
                         fill="tozeroy", fillcolor="rgba(201,169,97,0.10)"))
fig.add_trace(go.Scatter(x=gross.index, y=gross.values, name="Strategy (Gross)",
                         line=dict(color=OAK_GOLD, width=1.2, dash="dot"), opacity=0.55))
fig.add_trace(go.Scatter(x=bench_re.index, y=bench_re.values,
                         name="RE only (same model, no BTC)",
                         line=dict(color=OAK_SAGE, width=2, dash="dash")))
for _flabel, _fseries, _fcol in fund_benches:
    fig.add_trace(go.Scatter(x=_fseries.index, y=_fseries.values, name=_flabel,
                             line=dict(color=_fcol, width=1.6)))
fig = style_plotly(fig, height=480)
fig.update_yaxes(title_text="Value (CHF)", tickformat=",.0f")

# Endpoint value labels with vertical anti-overlap spreading: labels closer
# than ~4.5% of the y-range get pushed apart so they never collide.
_ep = [(net.index[-1], float(net.iloc[-1]), OAK_GOLD)]
_ep.append((bench_re.index[-1], float(bench_re.iloc[-1]), OAK_SAGE))
for _flabel, _fseries, _fcol in fund_benches:
    _ep.append((_fseries.index[-1], float(_fseries.iloc[-1]), _fcol))
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
# Sleeve Development — property, BTC, cash (point: cash/btc/EK Entwicklung)
# =====================================================================
st.markdown("## Sleeve Development")
fig_sl = go.Figure()
fig_sl.add_trace(go.Scatter(x=ts.index, y=ts["property_value"],
                            name="Immobilienwert", line=dict(color=OAK_SAGE, width=2)))
fig_sl.add_trace(go.Scatter(x=ts.index, y=ts["btc_value"],
                            name="BTC", line=dict(color=OAK_BTC, width=2)))
fig_sl.add_trace(go.Scatter(x=ts.index, y=ts["cash"],
                            name="CHF Cash", line=dict(color=OAK_CREAM_DIM, width=2)))
_reinv_days = ts.index[ts["re_reinvest"] > 0]
if len(_reinv_days):
    fig_sl.add_trace(go.Scatter(
        x=_reinv_days, y=ts.loc[_reinv_days, "cash"] + ts.loc[_reinv_days, "re_reinvest"],
        mode="markers", name="Cash → Immobilien Reinvest",
        marker=dict(symbol="triangle-down", size=11, color=OAK_GOLD,
                    line=dict(color=OAK_GREEN_2, width=1)),
        hovertemplate="%{x}<br>Reinvestiert: CHF %{customdata:,.0f}<extra></extra>",
        customdata=ts.loc[_reinv_days, "re_reinvest"]))
fig_sl = style_plotly(fig_sl, height=380)
fig_sl.update_yaxes(title_text="Value (CHF)", tickformat=",.0f")
st.plotly_chart(fig_sl, use_container_width=True)

st.markdown("### Asset Allocation Over Time (quarter-end values)")
q_end = ts[["property_value", "btc_value", "cash"]].resample("QE").last().dropna()
q_labels = [f"Q{d.quarter} {d.year}" for d in q_end.index]
fig_alloc = go.Figure()
fig_alloc.add_trace(go.Bar(x=q_labels, y=q_end["property_value"],
                           name="Immobilien", marker_color=OAK_SAGE,
                           hovertemplate="%{x}<br>Immobilien: CHF %{y:,.0f}<extra></extra>"))
fig_alloc.add_trace(go.Bar(x=q_labels, y=q_end["btc_value"],
                           name="BTC", marker_color=OAK_BTC,
                           hovertemplate="%{x}<br>BTC: CHF %{y:,.0f}<extra></extra>"))
fig_alloc.add_trace(go.Bar(x=q_labels, y=q_end["cash"],
                           name="CHF Cash", marker_color=OAK_CREAM_DIM,
                           hovertemplate="%{x}<br>Cash: CHF %{y:,.0f}<extra></extra>"))
fig_alloc.update_layout(barmode="stack", bargap=0.25)
fig_alloc = style_plotly(fig_alloc, height=400)
fig_alloc.update_yaxes(title_text="Value (CHF)", tickformat=",.0f")
fig_alloc.update_xaxes(tickangle=-45)
st.plotly_chart(fig_alloc, use_container_width=True)

st.markdown("### BTC & Cash Weight vs. Thresholds")
w_btc_series = (ts["btc_value"] / ts["total_value"]) * 100
w_cash_series = (ts["cash"] / ts["total_value"]) * 100
fig_w = go.Figure()
fig_w.add_trace(go.Scatter(x=ts.index, y=w_btc_series, name="BTC % of NAV",
                           line=dict(color=OAK_BTC, width=2.5),
                           fill="tozeroy", fillcolor="rgba(247,147,26,0.1)"))
fig_w.add_trace(go.Scatter(x=ts.index, y=w_cash_series, name="Cash % of NAV",
                           line=dict(color=OAK_CREAM_DIM, width=1.8, dash="dot")))
fig_w.add_hline(y=upper_threshold * 100, line=dict(color=OAK_RED, width=2, dash="dash"),
                annotation_text=f"Upper {upper_threshold*100:.0f}%",
                annotation_position="top right",
                annotation_font=dict(color=OAK_RED, size=11))
fig_w.add_hline(y=lower_threshold * 100, line=dict(color=OAK_SAGE, width=1.5, dash="dot"),
                annotation_text=f"Lower {lower_threshold*100:.0f}%",
                annotation_position="bottom right",
                annotation_font=dict(color=OAK_SAGE, size=11))
sell_days = ts.index[ts["btc_sells"] > 0]
if len(sell_days):
    fig_w.add_trace(go.Scatter(
        x=sell_days, y=w_btc_series.loc[sell_days] + 0.7, mode="markers",
        name="Sell → Cash",
        marker=dict(symbol="diamond", size=11, color=OAK_RED,
                    line=dict(color=OAK_CREAM, width=1.5))))
fig_w = style_plotly(fig_w, height=380)
fig_w.update_yaxes(title_text="% of NAV", ticksuffix="%")
st.plotly_chart(fig_w, use_container_width=True)

# =====================================================================
# Risk Analytics
# =====================================================================
st.markdown("## Risk Analytics")
strat_m = m
re_m = compute_risk_metrics(bench_re, risk_free_rate, base_value=initial_capital)
bm_re = compute_benchmark_metrics(net, bench_re, risk_free_rate)


def _row(label, key, fmt="pct", hint=""):
    if fmt == "pct":
        s, b1, b2 = _fmt_pct(strat_m.get(key)), _fmt_pct(re_m.get(key)), _fmt_pct(siat_m.get(key))
    else:
        s, b1, b2 = _fmt_num(strat_m.get(key)), _fmt_num(re_m.get(key)), _fmt_num(siat_m.get(key))
    hint_html = f"<span class='hint'>{hint}</span>" if hint else ""
    return (f"<tr><td class='metric-label'>{label}{hint_html}</td>"
            f"<td class='strategy-col'>{s}</td><td>{b1}</td><td>{b2}</td></tr>")


def _section(title):
    return f"<tr class='oak-section'><td colspan='4'>{title}</td></tr>"


st.markdown(f"""
<table class="oak-metrics-table">
    <thead>
        <tr><th>Metric</th><th>Strategy (Net)</th><th>RE only</th><th>UBS «Siat»</th></tr>
    </thead>
    <tbody>
        {_section("Return")}
        {_row("Total Return", "total_return")}
        {_row("Annualized Return (CAGR)", "cagr")}
        {_section("Risk · smoothed valuation index — see note")}
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
""", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{OAK_SAGE_DIM}; font-size:11px; margin-top:-8px;'>"
    f"Risk-free rate: {risk_free_rate*100:.2f}% p.a. · Property sleeve rests on a "
    f"smoothed quarterly valuation index — volatility and drawdowns of all three "
    f"columns are structurally understated and not comparable to market-priced "
    f"strategies.</p>", unsafe_allow_html=True)

bm_bench = (compute_benchmark_metrics(net, siat_series, risk_free_rate)
            if siat_series is not None else bm_re)
_bench_name = "UBS «Siat»" if siat_series is not None else "RE only"
st.markdown(f"### Strategy vs. {_bench_name}")
bc1, bc2, bc3, bc4 = st.columns(4)
bc1.metric("Alpha (Jensen, annualized)", _fmt_pct(bm_bench.get("alpha")),
           "Excess return adj. for beta")
bc2.metric("Beta", _fmt_num(bm_bench.get("beta")), f"Sensitivity to {_bench_name}")
bc3.metric("Tracking Error", _fmt_pct(bm_bench.get("tracking_error")),
           "Std. dev. of excess returns")
bc4.metric("Information Ratio", _fmt_num(bm_bench.get("information_ratio")),
           "Excess return / TE")

bc5, bc6 = st.columns([1, 3])
bc5.metric("Correlation", _fmt_num(bm_bench.get("correlation")),
           f"R² = {_fmt_num(bm_bench.get('r_squared'))}")
with bc6:
    if strat_m.get("dd_peak") and strat_m.get("dd_trough"):
        peak = pd.Timestamp(strat_m["dd_peak"]).strftime("%Y-%m-%d")
        trough = pd.Timestamp(strat_m["dd_trough"]).strftime("%Y-%m-%d")
        rec = (pd.Timestamp(strat_m["dd_recovery"]).strftime("%Y-%m-%d")
               if strat_m.get("dd_recovery") else "not yet recovered")
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
            f"</div></div>", unsafe_allow_html=True)

st.markdown("### Drawdown Analysis")
dd_strat = compute_drawdown(net) * 100
dd_re = compute_drawdown(bench_re) * 100
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=dd_strat.index, y=dd_strat.values, name="Strategy (Net)",
                            line=dict(color=OAK_GOLD, width=2),
                            fill="tozeroy", fillcolor="rgba(201,169,97,0.2)"))
fig_dd.add_trace(go.Scatter(x=dd_re.index, y=dd_re.values, name="RE only",
                            line=dict(color=OAK_SAGE, width=1.5, dash="dash")))
fig_dd = style_plotly(fig_dd, height=340)
fig_dd.update_yaxes(title_text="Drawdown", ticksuffix="%")
st.plotly_chart(fig_dd, use_container_width=True)

st.markdown("### Rolling Volatility (90-day window, annualized)")
# 90 calendar days ≈ 3 months on this 365-day series (mirrors the SMI page's
# 60 trading days); annualized with √365 to match the daily BTC/RE calendar.
roll_s = net.pct_change().dropna().rolling(90).std() * np.sqrt(365) * 100
roll_b = bench_re.pct_change().dropna().rolling(90).std() * np.sqrt(365) * 100
fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(x=roll_s.index, y=roll_s.values, name="Strategy (Net)",
                             line=dict(color=OAK_GOLD, width=2)))
fig_vol.add_trace(go.Scatter(x=roll_b.index, y=roll_b.values, name="RE only",
                             line=dict(color=OAK_SAGE, width=1.5, dash="dash")))
fig_vol = style_plotly(fig_vol, height=320)
fig_vol.update_yaxes(title_text="Annualized Volatility", ticksuffix="%")
st.plotly_chart(fig_vol, use_container_width=True)

st.markdown("### Monthly Returns · Strategy (Net)")
matrix = monthly_returns_matrix(net)
if not matrix.empty:
    z = matrix.values.astype(float) * 100
    years_idx = matrix.index.astype(str).tolist()
    cols = matrix.columns.tolist()
    colorscale = [[0.0, "#7A2A1F"], [0.25, "#B85042"], [0.5, OAK_GREEN_2],
                  [0.75, "#7A8975"], [1.0, OAK_SAGE]]
    vmax = max(abs(np.nanmin(z)), abs(np.nanmax(z)))
    text = [[f"{v:+.1f}%" if not np.isnan(v) else "" for v in row] for row in z]
    fig_hm = go.Figure(data=go.Heatmap(
        z=z, x=cols, y=years_idx, colorscale=colorscale, zmid=0, zmin=-vmax, zmax=vmax,
        text=text, texttemplate="%{text}",
        textfont=dict(size=11, color=OAK_CREAM, family="Inter"), xgap=2, ygap=2,
        colorbar=dict(title=dict(text="Return %", font=dict(color=OAK_CREAM, size=11)),
                      tickfont=dict(color=OAK_CREAM_DIM, size=10),
                      outlinecolor=OAK_BORDER, outlinewidth=1, len=0.85, thickness=12),
        hovertemplate="%{y} · %{x}: <b>%{z:+.2f}%</b><extra></extra>"))
    fig_hm = style_plotly(fig_hm, height=max(280, 38 * len(years_idx) + 80))
    fig_hm.update_xaxes(side="top", showgrid=False, ticks="")
    fig_hm.update_yaxes(showgrid=False, ticks="", autorange="reversed")
    st.plotly_chart(fig_hm, use_container_width=True)

st.markdown("### Yearly Performance & High Water Mark")
yearly_net = net.resample("YE").last()
yearly_ret = yearly_net.pct_change()
yearly_ret.iloc[0] = yearly_net.iloc[0] / initial_capital - 1
years_list = yearly_net.index.year.tolist()
rets_pct = (yearly_ret.values * 100).tolist()
fig_yr = go.Figure()
fig_yr.add_trace(go.Bar(
    x=years_list, y=rets_pct,
    marker=dict(color=[OAK_SAGE if r >= 0 else OAK_RED for r in rets_pct],
                line=dict(color=OAK_GREEN_2, width=1)),
    name="Strategy Annual Return (Net)",
    text=[f"{r:+.1f}%" for r in rets_pct], textposition="outside",
    textfont=dict(color=OAK_CREAM, size=11)))
fig_yr.add_hline(y=hurdle * 100, line=dict(color=OAK_GOLD, width=1.5, dash="dash"),
                 annotation_text=f"Year-1 Hurdle {hurdle*100:.0f}%",
                 annotation_position="top right",
                 annotation_font=dict(color=OAK_GOLD, size=11))
fig_yr.add_hline(y=0, line=dict(color=OAK_SAGE_DIM, width=1))
fig_yr = style_plotly(fig_yr, height=380)
fig_yr.update_xaxes(title_text="Year", dtick=1)
fig_yr.update_yaxes(title_text="Annual Return (Net)", ticksuffix="%")
st.plotly_chart(fig_yr, use_container_width=True)

# =====================================================================
# Parameter Sensitivity (grid backtest, like the SMI page)
# =====================================================================# =====================================================================
# Parameter Sensitivity (grid backtest, like the SMI page)
# =====================================================================
st.markdown("## Parameter Sensitivity")
st.markdown(
    f"<p style='color:{OAK_CREAM_DIM}; font-size:13px;'>"
    "Robustness check: re-runs the backtest across a grid of initial BTC "
    "allocations and upper thresholds, holding all other parameters at the "
    "current sidebar values (the lower threshold is clamped below each tested "
    "upper threshold). A single strong path means little if nearby parameters "
    "collapse.</p>", unsafe_allow_html=True)

if st.button("Run Sensitivity Analysis (grid backtest)", key="sens_btn"):
    btc_grid = [0.05, 0.10, 0.15, 0.20, 0.25]
    thr_grid = [0.20, 0.25, 0.30, 0.35]
    cagr_matrix, dd_matrix = [], []
    prog = st.progress(0.0, text="Running grid backtests ...")
    done, total_cells = 0, len(btc_grid) * len(thr_grid)
    for b in btc_grid:
        cagr_row, dd_row = [], []
        for thr in thr_grid:
            lo_g = min(lower_threshold, round(thr * 0.6, 2))
            try:
                ts_g = run_re_btc(prop_daily, btc_chf,
                                  dict(params, initial_btc_pct=b,
                                       upper_threshold=thr, lower_threshold=lo_g))
                if not ts_g.empty:
                    net_g, _, _, _ = apply_fees(
                        ts_g["total_value"], initial_capital,
                        mgmt_fee_annual=mgmt_fee, perf_fee_rate=perf_fee,
                        hwm_hurdle=hurdle, crystallization_freq=crystallization_freq,
                        hurdle_type=hurdle_type)
                    m_g = compute_risk_metrics(net_g, risk_free_rate,
                                               base_value=initial_capital)
                    cagr_row.append(m_g.get("cagr", float("nan")) * 100)
                    dd_row.append(m_g.get("max_drawdown", float("nan")) * 100)
                else:
                    cagr_row.append(float("nan")); dd_row.append(float("nan"))
            except Exception:
                cagr_row.append(float("nan")); dd_row.append(float("nan"))
            done += 1
            prog.progress(done / total_cells,
                          text=f"Running grid backtests ... {done}/{total_cells}")
        cagr_matrix.append(cagr_row); dd_matrix.append(dd_row)
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
                                yshift=-17,
                                font=dict(size=9, color=OAK_CREAM))
        return figh

    sc1, sc2 = st.columns(2)
    with sc1:
        fig_cagr = go.Figure(data=go.Heatmap(
            z=cagr_matrix, x=x_labels, y=y_labels,
            colorscale=[[0, OAK_RED], [0.5, OAK_GREEN_3], [1, OAK_GOLD]],
            text=[[f"{v:.1f}%" for v in row] for row in cagr_matrix],
            texttemplate="%{text}", textfont=dict(size=11, color=OAK_CREAM),
            colorbar=dict(title="CAGR %", tickfont=dict(color=OAK_CREAM)),
            hovertemplate="BTC init %{y} · Upper %{x}<br>Net CAGR %{z:.2f}%<extra></extra>"))
        fig_cagr.update_layout(title="Net CAGR (%)")
        fig_cagr = style_plotly(_mark_current(fig_cagr), height=380)
        fig_cagr.update_xaxes(title_text="Upper Threshold")
        fig_cagr.update_yaxes(title_text="Initial BTC %")
        st.plotly_chart(fig_cagr, use_container_width=True)
    with sc2:
        fig_ddh = go.Figure(data=go.Heatmap(
            z=dd_matrix, x=x_labels, y=y_labels,
            colorscale=[[0, OAK_RED], [1, OAK_GREEN_3]],
            text=[[f"{v:.1f}%" for v in row] for row in dd_matrix],
            texttemplate="%{text}", textfont=dict(size=11, color=OAK_CREAM),
            colorbar=dict(title="Max DD %", tickfont=dict(color=OAK_CREAM)),
            hovertemplate="BTC init %{y} · Upper %{x}<br>Max Drawdown %{z:.2f}%<extra></extra>"))
        fig_ddh.update_layout(title="Maximum Drawdown (%)")
        fig_ddh = style_plotly(_mark_current(fig_ddh), height=380)
        fig_ddh.update_xaxes(title_text="Upper Threshold")
        fig_ddh.update_yaxes(title_text="Initial BTC %")
        st.plotly_chart(fig_ddh, use_container_width=True)

# =====================================================================
# Monte-Carlo Forward Projection (like the SMI page)
# =====================================================================
st.markdown("## Monte-Carlo Projection")
st.markdown(
    f"<p style='color:{OAK_CREAM_DIM}; font-size:13px;'>"
    "Forward-looking simulation: bootstraps the strategy's historical daily net "
    "returns to generate thousands of possible future paths, shown as percentile "
    "bands. This is a statistical illustration based on past behaviour — "
    "<strong>not a forecast</strong>. Because the property sleeve rests on a "
    "smoothed valuation index, the projected bands understate real-world "
    "dispersion.</p>", unsafe_allow_html=True)

mc1, mc2, mc3 = st.columns(3)
with mc1:
    mc_years = st.slider("Projection Horizon (years)", 1, 10, 5, key="mc_years")
with mc2:
    mc_paths = st.select_slider("Number of Paths", options=[500, 1000, 2000, 5000],
                                value=1000, key="mc_paths")
with mc3:
    mc_method = st.selectbox("Method", ["Bootstrap (historical)", "Normal (parametric)"],
                             key="mc_method",
                             help="Bootstrap resamples actual historical daily returns "
                                  "(keeps fat tails). Normal assumes Gaussian returns "
                                  "with the same mean/volatility.")

if st.button("Run Monte-Carlo Simulation", key="mc_btn"):
    daily_ret = net.pct_change().dropna().values
    if len(daily_ret) < 30:
        st.warning("Not enough history for a meaningful projection.")
    else:
        start_value = float(net.iloc[-1])
        horizon_days = int(mc_years * 365)  # series runs on a 365-day calendar
        n_paths = int(mc_paths)
        rng = np.random.default_rng(42)
        if mc_method.startswith("Bootstrap"):
            sampled = rng.choice(daily_ret, size=(n_paths, horizon_days), replace=True)
        else:
            sampled = rng.normal(float(np.mean(daily_ret)), float(np.std(daily_ret)),
                                 size=(n_paths, horizon_days))
        cum = start_value * np.cumprod(1.0 + sampled, axis=1)
        bands = {p: np.percentile(cum, p, axis=0) for p in [5, 25, 50, 75, 95]}
        future_idx = pd.date_range(net.index[-1], periods=horizon_days + 1, freq="D")[1:]

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(x=future_idx, y=bands[95], mode="lines",
                                    line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_mc.add_trace(go.Scatter(x=future_idx, y=bands[5], mode="lines", fill="tonexty",
                                    fillcolor="rgba(153,167,150,0.15)", line=dict(width=0),
                                    name="5th–95th percentile"))
        fig_mc.add_trace(go.Scatter(x=future_idx, y=bands[75], mode="lines",
                                    line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_mc.add_trace(go.Scatter(x=future_idx, y=bands[25], mode="lines", fill="tonexty",
                                    fillcolor="rgba(153,167,150,0.30)", line=dict(width=0),
                                    name="25th–75th percentile"))
        fig_mc.add_trace(go.Scatter(x=future_idx, y=bands[50], mode="lines",
                                    line=dict(color=OAK_GOLD, width=2.5), name="Median path"))
        fig_mc = style_plotly(fig_mc, height=420)
        fig_mc.update_xaxes(title_text="Projected Date")
        fig_mc.update_yaxes(title_text="Projected Value (CHF)", tickformat=",.0f")
        st.plotly_chart(fig_mc, use_container_width=True)

        terminal = cum[:, -1]
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("5th percentile", fmt_chf(np.percentile(terminal,5)))
        t2.metric("25th percentile", fmt_chf(np.percentile(terminal,25)))
        t3.metric("Median", fmt_chf(np.percentile(terminal,50)))
        t4.metric("75th percentile", fmt_chf(np.percentile(terminal,75)))
        t5.metric("95th percentile", fmt_chf(np.percentile(terminal,95)))
        prob_loss = float(np.mean(terminal < start_value)) * 100
        st.markdown(
            f"<p style='color:{OAK_SAGE_DIM}; font-size:12px;'>"
            f"Starting from the current net value of CHF {start_value:,.0f}, over a "
            f"{mc_years}-year horizon across {n_paths:,} simulated paths: "
            f"<strong>{prob_loss:.1f}%</strong> of paths end below today's value. "
            "Results assume the future resembles the backtest period — which it may "
            "not, and the smoothed property index narrows the bands.</p>",
            unsafe_allow_html=True)

# ==========================================================================
# Fee Funding & Liquidity Stress — makes the pro-cyclical risk visible:
# in a flat/bear BTC regime the cash buffer dries up (no sell-downs, boost
# pulls rent into BTC) and fees increasingly force BTC liquidation.
# ==========================================================================
st.markdown("## Fee Funding & Liquidity Stress")
st.markdown(
    "<p style='color:#A9B5A4;margin-top:-6px'>Woher die Gebühren real bezahlt "
    "wurden — und wie nah das Konstrukt an Liquiditätsdruck kam. Im Bull-Case "
    "speisen Sell-downs den Cash; bei fallendem oder seitwärts laufendem BTC "
    "bleiben Sell-downs aus und Gebühren müssen zunehmend über BTC-Verkäufe "
    "gedeckt werden (prozyklisch).</p>", unsafe_allow_html=True)

total_fees_paid = float(ts["mgmt_fee_paid"].sum())
fee_from_btc_total = float(ts["fee_from_btc"].sum())
fee_from_cash_total = float(ts["fee_from_cash"].sum())
btc_share = (fee_from_btc_total / total_fees_paid * 100) if total_fees_paid > 0 else 0.0
forced = ts.index[ts["fee_btc_sold"] > 0]
first_forced = f"{forced[0]:%b %Y}" if len(forced) else "nie"
below_floor = int((ts["cash"] < ts["cash_floor"] - 1e-6).sum())
below_floor_pct = below_floor / len(ts) * 100
max_debt = float(ts["fee_debt"].max())
# Cash runway: how many months the current cash covers the current monthly fee
_last_aum = float(ts["total_value"].iloc[-1])
_monthly_fee_now = _last_aum * (mgmt_fee / 12.0)
runway = (float(ts["cash"].iloc[-1]) / _monthly_fee_now) if _monthly_fee_now > 0 else float("inf")

s1, s2, s3, s4 = st.columns(4)
s1.metric("Fees via BTC-Verkauf", fmt_chf(fee_from_btc_total),
          f"{btc_share:.0f}% aller Gebühren", delta_color="off")
s2.metric("Erster Zwangsverkauf", first_forced,
          "BTC zur Fee-Deckung" if len(forced) else "Cushion hat getragen",
          delta_color="off")
s3.metric("Tage Cash < Floor", f"{below_floor:,}",
          f"{below_floor_pct:.0f}% der Laufzeit", delta_color="off")
s4.metric("Cash-Runway (heute)",
          ("∞" if runway == float("inf") else f"{runway:.1f} Mte"),
          "aktuelles Polster ÷ Monatsfee", delta_color="off")

# Fee funding source over time (quarterly stacked: cash vs forced BTC sales)
_q_cash = ts["fee_from_cash"].groupby([ts.index.year, ts.index.quarter]).sum()
_q_btc = ts["fee_from_btc"].groupby([ts.index.year, ts.index.quarter]).sum()
_q_labels = [f"Q{q} {y}" for (y, q) in _q_cash.index]
fig_fund = go.Figure()
fig_fund.add_trace(go.Bar(x=_q_labels, y=_q_cash.values, name="aus Cash",
                          marker_color=OAK_SAGE))
fig_fund.add_trace(go.Bar(x=_q_labels, y=_q_btc.values, name="aus BTC-Verkauf",
                          marker_color=OAK_BTC))
fig_fund.update_layout(barmode="stack", title="Gebühren-Finanzierungsquelle je Quartal")
fig_fund = style_plotly(fig_fund, height=340)
st.plotly_chart(fig_fund, use_container_width=True)

# Cash vs dynamic floor, with stress shading where cash sits below the floor
fig_liq = go.Figure()
fig_liq.add_trace(go.Scatter(x=ts.index, y=ts["cash"], name="CHF Cash",
                             line=dict(color=OAK_CREAM_DIM, width=2)))
fig_liq.add_trace(go.Scatter(x=ts.index, y=ts["cash_floor"], name="Reserve-Floor",
                             line=dict(color=OAK_GOLD, width=1.5, dash="dot")))
if len(forced):
    fig_liq.add_trace(go.Scatter(
        x=forced, y=ts.loc[forced, "cash"], mode="markers",
        name="BTC-Zwangsverkauf",
        marker=dict(symbol="x", size=8, color=OAK_BTC)))
fig_liq.update_layout(title="Cash vs. Reserve-Floor (Liquiditätsdruck)")
fig_liq = style_plotly(fig_liq, height=340)
fig_liq.update_yaxes(tickformat=",.0f")
st.plotly_chart(fig_liq, use_container_width=True)

if btc_share >= 25:
    st.warning(
        f"⚠️ In diesem Szenario wurden **{btc_share:.0f}%** der Gebühren durch "
        f"BTC-Verkäufe gedeckt ({fmt_chf(fee_from_btc_total)}). Das ist der "
        "prozyklische Effekt: bei schwachem BTC trocknet der Cash-Puffer aus "
        "und Gebühren zwingen zu Verkäufen — genau dann, wenn die Strategie "
        "eigentlich akkumulieren möchte. Hebel zur Abfederung: höhere "
        "Initial-Cash-Reserve, Fee nur auf den Immobilien-NAV, oder Boost-Rate "
        "bei knappem Cash drosseln.")

# ---- detail expanders (unchanged) ----
with st.expander("Monatliche Netto-Cashflows (Mieterträge → BTC-DCA)"):
    cf = ts["net_cf_monthly"].dropna()
    st.bar_chart(cf)

with st.expander("Gebühren-Aufstellung je Periode"):
    if fee_events is not None and not fee_events.empty:
        st.dataframe(fee_events)
    st.write(f"Mgmt: {fmt_chf(total_mgmt)} · Perf: {fmt_chf(total_perf)} · "
             f"Total: CHF {total_mgmt + total_perf:,.0f}")

# --------------------------------------------------------------------------
# PDF tearsheet (bilingual, reuses pdf_report with RE-specific overrides)
# --------------------------------------------------------------------------
st.markdown("---")
if st.button("PDF-Tearsheet generieren (DE+EN)"):
    with st.spinner("Erzeuge PDF…"):
        gross = ts["total_value"]
        fee_drag = ((gross.iloc[-1] / initial_capital) ** (1 / years) - 1) - net_cagr
        excess = net_cagr - re_cagr

        line_series = [
            ("OAK RE/BTC (Net of Fees)", net, "#B8954A", {"lw": 2.2}),
            ("RE only (same model, no BTC)", bench_re, "#7C8978", {"ls": "--", "lw": 1.6}),
        ]
        for _flabel, _fseries, _fcol in fund_benches:
            line_series.append((_flabel, _fseries, _fcol, {"lw": 1.3}))
        line = render_line_chart(line_series, ylabel="Value (CHF)",
                                 annotate_end=True, fill_first=True)

        dd = compute_drawdown(net)
        dd_b = compute_drawdown(bench_re)
        dd_chart = render_line_chart([
            ("Strategy (Net)", dd, "#B8954A", {"lw": 1.8}),
            ("RE only", dd_b, "#7C8978", {"ls": "--", "lw": 1.4}),
        ], ylabel="Drawdown", percent=True)

        yearly = net.resample("YE").last()
        yearly_ret = yearly.pct_change()
        first_year_ret = yearly.iloc[0] / initial_capital - 1
        yearly_ret.iloc[0] = first_year_ret
        ylabels = [str(y.year) + ("*" if i == len(yearly_ret) - 1 else "")
                   for i, y in enumerate(yearly_ret.index)]
        bar = render_bar_chart(ylabels, [v * 100 for v in yearly_ret.values],
                               hurdle=hurdle * 100)

        mb = compute_risk_metrics(bench_re, base_value=initial_capital)

        def _sm(key, pct=True):
            v = siat_m.get(key)
            if v is None:
                return "n/a"
            return f"{v*100:.2f}%" if pct else f"{v:.2f}"
        scatter = render_scatter_chart([
            ("Strategy", m["vol_ann"] * 100, net_cagr * 100, "#B8954A", "o"),
            ("RE only", mb["vol_ann"] * 100, re_cagr * 100, "#7C8978", "s"),
        ])

        q_end_pdf = ts[["property_value", "btc_value", "cash"]].resample("QE").last().dropna()
        alloc_png = render_stacked_bar_chart(
            [f"Q{d.quarter} {d.year}" for d in q_end_pdf.index],
            [("Residential RE", q_end_pdf["property_value"].tolist(), "#7C8978"),
             ("Bitcoin", q_end_pdf["btc_value"].tolist(), "#F7931A"),
             ("CHF Cash", q_end_pdf["cash"].tolist(), "#C9C9C0")],
            ylabel="Value (CHF)")

        mm = monthly_returns_matrix(net)
        monthly_dict = {int(y): [None if pd.isna(v) else round(v * 100, 1)
                                 for v in mm.loc[y, mm.columns[:12]]]
                        for y in mm.index} if not mm.empty else None

        fee_rows = []
        if fee_events is not None and not fee_events.empty:
            for _, ev in fee_events.iterrows():
                per = pd.Timestamp(ev["date"])
                fee_rows.append([f"Q{per.quarter} {per.year}",
                                 f"CHF {ev.get('mgmt_fee', 0):,.0f}",
                                 f"CHF {ev.get('perf_fee', 0):,.0f}",
                                 f"CHF {ev.get('mgmt_fee', 0) + ev.get('perf_fee', 0):,.0f}"])

        period_str = f"{net.index[0]:%Y-%m-%d} to {net.index[-1]:%Y-%m-%d}"
        freq_de = {"M": "monatlich", "Q": "quartalsweise"}
        sell_de = ("oberhalb wird auf Rebalancing-Terminen auf die "
                   "Obergrenze zurückgeführt; die Erlöse verbleiben als "
                   "Cash-Puffer" if sell_on_upper else
                   "oberhalb wird die Mietallokation ausgesetzt")
        sell_en = ("above it, positions are trimmed back to the cap on "
                   "rebalancing dates, with proceeds held as a cash buffer"
                   if sell_on_upper else
                   "above it, rent allocation is suspended")
        exec_de = (f"Die Strategie kombiniert ein Schweizer Wohnimmobilien-Portfolio "
                   f"mit einer strukturellen Bitcoin-Allokation und einem "
                   f"CHF-Cash-Puffer. Die Wertentwicklung der Liegenschaften folgt "
                   f"dem Wohnimmobilienpreisindex der Schweizerischen Nationalbank; "
                   f"die Nettomietrendite von {net_yield*100:.1f}% p.a. — extern "
                   f"vorberechnet, nach Leerstand, Bewirtschaftung und Finanzierung "
                   f"— wird nach festen Bandregeln alloziert: unterhalb einer "
                   f"BTC-Quote von {lower_threshold*100:.0f}% fliessen "
                   f"{boost_invest_rate*100:.0f}% der Nettomiete in Bitcoin, "
                   f"innerhalb des Bandes {base_invest_rate*100:.0f}%, oberhalb von "
                   f"{upper_threshold*100:.0f}% wird nicht investiert — {sell_de}. "
                   f"Im Simulationszeitraum erzielte die Strategie einen Netto-CAGR "
                   f"von {net_cagr*100:.1f}%, gegenüber {re_cagr*100:.1f}% für das "
                   f"identische Immobilienmodell ohne Bitcoin. Der Immobilienteil "
                   f"beruht auf einem geglätteten Bewertungsindex — die Ergebnisse "
                   f"sind als parametrische Simulation zu verstehen, nicht als "
                   f"marktdatenbasierter Backtest.")
        exec_en = (f"The strategy combines a Swiss residential property portfolio "
                   f"with a structural Bitcoin allocation and a CHF cash buffer. "
                   f"Property values track the Swiss National Bank's residential "
                   f"property price index, while the net rental yield of "
                   f"{net_yield*100:.1f}% p.a. — pre-computed externally, after "
                   f"vacancy, operating costs and financing — is allocated by fixed "
                   f"band rules: below a Bitcoin weight of "
                   f"{lower_threshold*100:.0f}%, {boost_invest_rate*100:.0f}% of "
                   f"net rent flows into Bitcoin; within the band, "
                   f"{base_invest_rate*100:.0f}%; above {upper_threshold*100:.0f}%, "
                   f"no new investments are made — {sell_en}. Over the simulation "
                   f"period the strategy delivered a net CAGR of "
                   f"{net_cagr*100:.1f}%, versus {re_cagr*100:.1f}% for the "
                   f"identical property model without Bitcoin. As the property "
                   f"sleeve rests on a smoothed valuation index, results should be "
                   f"read as a parametric simulation rather than a market-data "
                   f"backtest.")
        kt_de = [
            f"Netto-CAGR von {net_cagr*100:.1f}% gegenüber {re_cagr*100:.1f}% für das identische Immobilienmodell ohne Bitcoin — ein BTC-Beitrag von {excess*100:+.1f}% p.a.",
            f"Bandregeln steuern die Mietallokation: {boost_invest_rate*100:.0f}% unter {lower_threshold*100:.0f}% BTC-Quote, {base_invest_rate*100:.0f}% im Band, Stopp über {upper_threshold*100:.0f}% — aktuelle BTC/Cash-Quote {w_btc*100:.0f}%/{w_cash*100:.0f}%.",
            "Geglätteter Bewertungsindex: Volatilität und Drawdowns des Immobilienteils sind strukturell untererfasst — siehe Hinweise.",
        ]
        kt_en = [
            f"Net CAGR of {net_cagr*100:.1f}% versus {re_cagr*100:.1f}% for the identical property model without Bitcoin — a BTC contribution of {excess*100:+.1f}% p.a.",
            f"Band rules govern rent allocation: {boost_invest_rate*100:.0f}% below a {lower_threshold*100:.0f}% BTC weight, {base_invest_rate*100:.0f}% within the band, none above {upper_threshold*100:.0f}% — current BTC/cash weights {w_btc*100:.0f}%/{w_cash*100:.0f}%.",
            "Smoothed valuation index: volatility and drawdowns of the property sleeve are structurally understated — see Disclosures.",
        ]
        snapshot = [("sn_inception", f"{net.index[0]:%d %b %Y}"),
                    ("sn_currency", "CHF"),
                    ("sn_benchmark", "RE only (same model)"),
                    ("sn_style", "Real Assets (Residential + BTC)"),
                    ("sn_domicile", "Switzerland"),
                    ("sn_frequency", "Daily")]

        params_summary = [
            ("Initial Capital", f"CHF {initial_capital:,.0f}"),
            ("Initial Allocation", f"{(1-initial_btc_pct)*100:.0f}% Residential RE / {initial_btc_pct*100:.0f}% BTC"),
            ("Capital-Value Source", f"SNB index '{series_label[:48]}' (quarterly, interpolated)"),
            ("Net Rental Yield", f"{net_yield*100:.1f}% p.a. (pre-computed, on invested capital)"),
            ("Lower BTC Threshold", f"{lower_threshold*100:.0f}%"),
            ("Upper BTC Threshold", f"{upper_threshold*100:.0f}%"),
            ("Base Investment Rate (net rent)", f"{base_invest_rate*100:.0f}%"),
            ("Boosted Rate below Lower Threshold", f"{boost_invest_rate*100:.0f}%"),
            ("Rent → BTC Allocation", "Monthly" if rent_to_btc_freq == "monatlich" else "Quarterly"),
            ("BTC → Cash Allocation", "Monthly" if btc_to_cash_freq == "monatlich" else "Quarterly"),
            ("Sell Rule (above Upper Threshold)", "Sell down to upper threshold into CHF cash" if sell_on_upper else "Disabled"),
            ("Cash Treatment", f"Uninvested CHF, 0% interest; reinvested into property quarterly in CHF {cash_reinvest_block:,.0f} blocks"),
            ("Transaction Cost (BTC)", f"{tx_cost_bps} bps per trade"),
            ("Management Fee", f"{mgmt_fee*100:.2f}% p.a."),
            ("Performance Fee", f"{perf_fee*100:.0f}% ({crystallization_freq}, {hurdle_type} {hurdle*100:.1f}% Yr 1)"),
        ]
        universe_rows = [
            ["Bitcoin", "BTC", "Digital Assets",
             f"Band {lower_threshold*100:.0f}–{upper_threshold*100:.0f}%"],
            ["CH Wohnliegenschaften (parametrisch)", "SNB plimoinchq", "Residential Real Estate",
             f"{(1-initial_btc_pct)*100:.0f}% initial"],
            ["CHF Cash", "—", "Cash",
             "Sweep-Puffer · 0%"],
        ]
        disc_de = [
            "Dieses Dokument wurde von Oakwood Capital ausschliesslich zu illustrativen und informativen Zwecken erstellt. Es stellt weder eine Anlageberatung, eine Empfehlung, ein Angebot noch eine Aufforderung zum Kauf oder Verkauf eines Finanzinstruments dar.",
            "OAK RE/BTC ist eine parametrische Simulation und kein marktdatenbasierter Backtest. Die Wertentwicklung des Immobilienteils folgt einem quartalsweise erhobenen Bewertungsindex der SNB-Datenplattform, der für die Simulation linear auf Tagesbasis interpoliert wird. Bewertungsindizes sind geglättet und unterzeichnen die tatsächliche Volatilität und die Drawdowns von Immobilienanlagen erheblich; Volatilität, Sharpe Ratio und Drawdown-Kennzahlen sind deshalb nicht mit marktbasierten Strategien vergleichbar. Die Nettomietrendite ist eine extern vorberechnete Annahme — nach Leerstand, Bewirtschaftung, Unterhalt und Finanzierung — und kein realisierter Wert; eine allfällige Fremdfinanzierung der Liegenschaften ist im Modell nicht abgebildet.",
            "Der Bitcoin-Anteil basiert auf historischen Marktpreisen (BTC/USD, in Schweizer Franken umgerechnet). Digitale Vermögenswerte sind hochvolatil und können zum Totalverlust des eingesetzten Kapitals führen. Der CHF-Cash-Puffer wird unverzinst gehalten und quartalsweise in festgelegten Blöcken in Immobilien reinvestiert, sobald die Blockgrösse erreicht ist; der verbleibende Cash dämpft die Volatilität zulasten der erwarteten Rendite.",
            "Die simulierte Performance ist hypothetisch, unterliegt dem Vorteil der Rückschau und ist kein verlässlicher Indikator für zukünftige Ergebnisse. Die ausgewiesenen Zahlen verstehen sich nach Abzug der angegebenen Management- und Performance-Gebühren; Steuern — insbesondere Grundstückgewinn-, Liegenschafts- und Einkommenssteuern — sind nicht modelliert.",
            "Dieses Material ist streng vertraulich und ausschliesslich für den Empfänger bestimmt. Es darf ohne vorherige schriftliche Zustimmung von Oakwood Capital weder reproduziert noch verbreitet werden.",
        ]
        disc_en = [
            "This document has been prepared by Oakwood Capital for illustrative and informational purposes only. It does not constitute investment advice, a recommendation, an offer, or a solicitation to buy or sell any financial instrument.",
            "OAK RE/BTC is a parametric simulation, not a market-data backtest. Capital values of the property sleeve follow a quarterly valuation index from the SNB data portal, linearly interpolated to daily frequency for the simulation. Valuation indices are smoothed and materially understate the true volatility and drawdowns of real estate investments; volatility, Sharpe ratio and drawdown figures are therefore not comparable to market-priced strategies. The net rental yield is an externally pre-computed assumption — after vacancy, operating costs, maintenance and financing — not a realized figure; any debt financing of the properties is not modelled.",
            "The Bitcoin sleeve is based on historical market prices (BTC/USD converted to CHF). Digital assets are highly volatile and may result in total loss. The CHF cash buffer is held uninvested and reinvested into property quarterly in fixed blocks once the buffer reaches the block size; the remaining cash dampens volatility at the cost of expected return.",
            "Simulated performance is hypothetical, benefits from hindsight, and is not a reliable indicator of future results. Figures are shown net of the stated management and performance fees; taxes — in particular property-gains, property and income taxes — are not modelled.",
            "This material is strictly confidential and intended solely for the recipient. It may not be reproduced or distributed without the prior written consent of Oakwood Capital.",
        ]
        pdf_bytes = build_bilingual_tearsheet(
            strategy_name="OAK RE/BTC",
            strategy_subtitle_de="Schweizer Wohnimmobilien mit struktureller Bitcoin-Allokation, mietertragsfinanziertem DCA und schwellenwertbasiertem Risikomanagement.",
            strategy_subtitle_en="Swiss residential real estate with a structural Bitcoin allocation, rent-funded DCA and threshold-based risk management.",
            period_str=period_str,
            kpis_performance=[("Strategy (Net)", f"CHF {net.iloc[-1]:,.0f}"),
                              ("Net CAGR", f"{net_cagr*100:.2f}%"),
                              ("RE only", f"CHF {bench_re.iloc[-1]:,.0f}"),
                              ("BTC Contribution", f"{excess*100:+.2f}% p.a.")],
            kpis_risk=[("Sharpe Ratio*", f"{m['sharpe']:.2f}"),
                       ("Sortino Ratio*", f"{m['sortino']:.2f}"),
                       ("Max Drawdown*", f"{m['max_drawdown']*100:.2f}%"),
                       ("Volatility*", f"{m['vol_ann']*100:.2f}%")],
            fee_summary=[("Mgmt Fees", f"CHF {total_mgmt:,.0f}"),
                         ("Perf Fees", f"CHF {total_perf:,.0f}"),
                         ("Total Fees", f"CHF {total_mgmt+total_perf:,.0f}"),
                         ("Fee Drag", f"{fee_drag*100:.2f}% p.a.")],
            risk_table_headers=["Metric", "Strategy (Net)", "RE only", "UBS «Siat»"],
            risk_table_rows=[
                ["Total Return", f"{(net.iloc[-1]/initial_capital-1)*100:.2f}%",
                 f"{(bench_re.iloc[-1]/initial_capital-1)*100:.2f}%", _sm("total_return")],
                ["CAGR", f"{net_cagr*100:.2f}%", f"{re_cagr*100:.2f}%", _sm("cagr")],
                ["Volatility*", f"{m['vol_ann']*100:.2f}%", f"{mb['vol_ann']*100:.2f}%", _sm("vol_ann")],
                ["Max Drawdown*", f"{m['max_drawdown']*100:.2f}%", f"{mb['max_drawdown']*100:.2f}%", _sm("max_drawdown")],
                ["Sharpe Ratio*", f"{m['sharpe']:.2f}", f"{mb['sharpe']:.2f}", _sm("sharpe", pct=False)],
                ["Sortino Ratio*", f"{m['sortino']:.2f}", f"{mb['sortino']:.2f}", _sm("sortino", pct=False)],
            ],
            fee_table_headers=["Period", "Mgmt Fee", "Perf Fee", "Total Cost"],
            fee_table_rows=fee_rows,
            figures=[("Portfolio Evolution vs. RE-only & UBS «Siat»", line),
                     ("Asset Allocation Over Time (quarter-end)", alloc_png),
                     ("Drawdown Analysis*", dd_chart),
                     ("Yearly Net Performance", bar)],
            params_summary=params_summary,
            universe_rows=universe_rows,
            monthly_returns=monthly_dict,
            exec_summary_de=exec_de, exec_summary_en=exec_en,
            key_takeaways_de=kt_de, key_takeaways_en=kt_en,
            scatter_png=scatter,
            snapshot_data=snapshot,
            period_returns=compute_period_returns(net, bench_re),
            top_drawdowns=identify_top_drawdowns(net),
            perf_summary_sub_de="Nach Gebühren und Transaktionskosten · *Kennzahlen auf geglättetem Bewertungsindex — siehe Hinweise",
            perf_summary_sub_en="Net of fees and transaction costs · *Metrics on a smoothed valuation index — see Disclosures",
            benchmark_label_de="RE only (gleiches Modell)",
            benchmark_label_en="RE only (same model)",
            universe_sub_de=("Drei Sleeves: Schweizer Wohnimmobilien (parametrisch, "
                             "SNB-Wohnimmobilienpreisindex), eine strukturelle "
                             "Bitcoin-Allokation (bandgesteuert) und ein unverzinster "
                             "CHF-Cash-Puffer. Siehe Methodik und Hinweise."),
            universe_sub_en=("Three sleeves: Swiss residential real estate (parametric, "
                             "SNB residential price index), a structural band-managed "
                             "Bitcoin allocation, and an uninvested CHF cash buffer. "
                             "See Methodology and Disclosures."),
            disclaimer_paragraphs_de=disc_de,
            disclaimer_paragraphs_en=disc_en,
        )

    fname = f"OAK_RE_BTC_{date.today():%Y%m%d}.pdf"
    st.download_button("PDF herunterladen", data=pdf_bytes, file_name=fname,
                       mime="application/pdf")
    try:
        status = get_font_status()
        if status.get("crimson_pro") and status.get("work_sans"):
            st.success("✓ Brand fonts embedded: Crimson Pro + Work Sans")
        else:
            st.warning(f"Fallback-Fonts aktiv (Times/Helvetica) — Status: {status}")
    except Exception:
        pass
