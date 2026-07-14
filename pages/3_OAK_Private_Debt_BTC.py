"""
OAK Swiss Private Debt / Bitcoin — AMC Backtesting

Konzept (OAK Yield Bridge, dritte Anwendung):
  * Kern: diversifiziertes, immobilienbesichertes Schweizer Kreditbuch
    (Referenz: LEND Hypovest, ISIN CH1357099691 — nachrangige Hypotheken,
    Zielrendite 5.5-6.5% brutto). Parametrisch modelliert: die Nettorendite
    ist ein regelbarer Input, es gibt KEINE Marktdaten-Historie (Emission
    Juli 2024) und KEINE Kapitalwertschwankung (at par, held to maturity).

  * BESONDERHEIT: Das Underlying ist THESAURIEREND — es schüttet nichts aus.
    Der Ertrag wird über eine monatliche ERTRAGS-ERNTE realisiert: es werden
    exakt so viele Anteile redimiert, wie dem aufgelaufenen NAV-Zuwachs über
    der Kostenbasis entsprechen. Nie das Kapital. Ökonomisch identisch mit
    einem Coupon; die Kostenbasis bleibt nachweislich konstant.

  * REDEMPTION-RISIKO: Rücknahmen sind best-effort (kein Sekundärmarkt).
    Klemmen sie, bleibt der Ertrag im Zertifikat und verzinst sich weiter —
    der DCA pausiert und holt auf. Der Ausfallmodus ist damit gutartig, aber
    er MUSS ausgewiesen werden (Parameter "Redemption-Erfolgsquote").

  * Wachstum ausschliesslich über NEUE ZEICHNUNGEN (kein organischer
    Kapitalwertzuwachs im Kreditbuch).

  * BTC-Sleeve: identische Bandmechanik wie in den Produkten 1 und 2.
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

st.set_page_config(page_title="Oakwood Capital — Swiss Private Debt / Bitcoin",
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
        # An empty Series/DataFrame defaults to a RangeIndex; returning it as-is
        # makes later `index <= timestamp` comparisons raise TypeError under
        # pandas 3.x. Give it an empty DatetimeIndex so comparisons stay safe.
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




def run_pd_btc(btc_chf, idx, params):
    """OAK Swiss Private Debt / Bitcoin — Engine.

    KERNUNTERSCHIED zu RE/BTC: Das Underlying (LEND Hypovest, CH1357099691) ist
    ein THESAURIERENDES Zertifikat — es schüttet nichts aus. Der Ertrag wächst in
    den NAV hinein. Um die Yield Bridge zu speisen, wird er geerntet:

      ERTRAGS-ERNTE: Monatlich wird exakt der aufgelaufene NAV-Zuwachs über der
      Kostenbasis redimiert — nie das eingesetzte Kapital. Ökonomisch identisch
      mit einem Coupon; die Kostenbasis bleibt konstant, das Kernkapital bleibt
      vollständig investiert («core capital untouched» ist hier buchstäblich wahr).

      REDEMPTION-RISIKO: Rücknahmen erfolgen best-effort (kein Sekundärmarkt).
      redemption_rate < 100% modelliert klemmende Rücknahmen. Der NICHT geerntete
      Teil geht NICHT verloren — er bleibt im Zertifikat und verzinst sich weiter.
      Der DCA pausiert also nur und holt automatisch auf. Genau dieses Verhalten
      ist der Grund, warum das Produkt in Stressphasen nicht bricht.

    Der Debt-Sleeve hat KEINE Kapitalwertschwankung (at par, held to maturity) —
    das ist ehrlich für Private Debt, muss aber in den Disclosures stehen: die
    Volatilität des Sleeves ist strukturell null und die Risikokennzahlen des
    Gesamtprodukts sind dadurch nach unten verzerrt.

    Params:
      initial_capital, initial_btc_pct, initial_cash_pct
      net_yield          Nettorendite p.a. des Kreditbuchs (nach Investor Fee)
      credit_loss_rate   Kreditausfälle p.a. (Stressparameter; subordinated!)
      redemption_rate    Anteil der Ernte, der tatsächlich redimiert wird (0..1)
      subscription_amount / subscription_freq   neue Zeichnungen ("M","Q","A","N")
      lower_threshold, upper_threshold, base_invest_rate, boost_invest_rate
      harvest_to_btc_freq, btc_to_cash_freq, sell_on_upper
      mgmt_fee, mgmt_fee_freq, perf_fee, hwm_hurdle, hurdle_type,
      crystallization_freq, tx_cost_bps, cash_rate, risk_free_rate
    """
    idx = pd.DatetimeIndex(idx).normalize().unique().sort_values()
    if len(idx) < 30:
        return pd.DataFrame()
    btc = btc_chf.reindex(idx).ffill().bfill()

    cap = float(params["initial_capital"])
    tx = float(params.get("tx_cost_bps", 0.0)) / 10000.0
    btc_chf0 = cap * float(params["initial_btc_pct"])
    cash0 = cap * float(params.get("initial_cash_pct", 0.0))

    # ---- ATTRIBUTION: zwei getrennte BTC-Lots, Verkäufe PRO RATA ------------
    btc_u_init = (btc_chf0 * (1 - tx)) / btc.iloc[0] if btc_chf0 > 0 else 0.0
    btc_u_dca = 0.0
    btc_init_invested = btc_chf0
    btc_dca_invested = 0.0
    btc_init_realized = 0.0
    btc_dca_realized = 0.0

    # ---- Debt-Sleeve (rein parametrisch, kein Marktdaten-Index) ------------
    debt_cost_basis = cap - btc_chf0 - cash0   # tatsächlich eingesetztes Kapital
    debt_accrued = 0.0                         # aufgelaufener, noch nicht geernteter Ertrag
    harvest_total = 0.0                        # tatsächlich geerntet (= der "Coupon")
    accrued_total = 0.0                        # gesamter Ertragsanfall (auch ungeerntet)
    loss_total = 0.0                           # Kreditausfälle
    subs_total = 0.0                           # neue Zeichnungen

    cash = cash0
    harvest_pool = 0.0     # geerntet, aber noch nicht in BTC/Cash alloziert
    fee_floor = cash0
    fee_debt = 0.0
    cash_interest_total = 0.0
    cash_drag = 0.0

    ny = float(params["net_yield"])
    loss_r = float(params.get("credit_loss_rate", 0.0))
    redeem_r = float(params.get("redemption_rate", 1.0))
    sub_amt = float(params.get("subscription_amount", 0.0))
    sub_freq = params.get("subscription_freq", "N")

    lo = float(params["lower_threshold"])
    up = float(params["upper_threshold"])
    base_rate = float(params["base_invest_rate"])
    boost_rate = float(params["boost_invest_rate"])
    h_freq = params.get("harvest_to_btc_freq", "M")
    c_freq = params.get("btc_to_cash_freq", "Q")
    sell_on = bool(params.get("sell_on_upper", True))

    mgmt_fee = float(params.get("mgmt_fee", 0.0))
    mgmt_freq = params.get("mgmt_fee_freq", "M")
    perf_fee = float(params.get("perf_fee", 0.0))
    hwm_hurdle = float(params.get("hwm_hurdle", 0.0))
    hurdle_type = params.get("hurdle_type", "Hard Hurdle")
    cryst = params.get("crystallization_freq", "Quarterly")
    cryst_months = ({12} if cryst == "Annual"
                    else {6, 12} if cryst == "Semi-Annual" else {3, 6, 9, 12})
    hwm = cap
    prev_cryst_date = idx[0]
    total_mgmt = 0.0
    total_perf = 0.0

    rf_cash = float(params.get("cash_rate", 0.0))
    rf_opp = float(params.get("risk_free_rate", 0.0))

    _btc_v = btc.to_numpy(dtype=float)
    rows = []

    for i, d in enumerate(idx):
        px = _btc_v[i]
        debt_val = debt_cost_basis + debt_accrued
        b_val = (btc_u_init + btc_u_dca) * px

        harvest = 0.0
        buys = 0.0
        sells = 0.0
        subs = 0.0
        fee_paid = 0.0
        fee_btc_sold = 0.0
        fee_from_cash = 0.0
        fee_from_btc = 0.0
        mgmt_paid_row = 0.0
        perf_paid_row = 0.0

        is_me = (i == len(idx) - 1) or (idx[i + 1].to_period("M") != d.to_period("M"))
        is_qe = is_me and d.month in (3, 6, 9, 12)
        is_ye = is_me and d.month == 12

        # ---- Ertragsanfall im Zertifikat (thesaurierend) --------------------
        # Netto-Ertrag minus Kreditausfälle, täglich auf den vollen Sleeve-Wert.
        _accr = debt_val * (ny / 365.0)
        _loss = debt_val * (loss_r / 365.0)
        debt_accrued += _accr - _loss
        accrued_total += _accr
        loss_total += _loss

        # ---- Cash-Verzinsung (real) + Drag-Memo ----------------------------
        if rf_cash > 0 and cash > 0:
            _int = cash * (rf_cash / 365.0)
            cash += _int
            cash_interest_total += _int
        cash_drag += (cash + harvest_pool) * (max(rf_opp - rf_cash, 0.0) / 365.0)

        # ---- Neue Zeichnungen: wachsen den KERN, nie den Satelliten ---------
        if sub_amt > 0 and (
                (sub_freq == "M" and is_me) or (sub_freq == "Q" and is_qe)
                or (sub_freq == "A" and is_ye)):
            subs = sub_amt
            debt_cost_basis += subs      # frisches Kapital -> Kreditbuch
            subs_total += subs
            debt_val = debt_cost_basis + debt_accrued

        # ---- ERTRAable-ERNTE (monatlich): nur der Zuwachs, nie das Kapital --
        if is_me and debt_accrued > 0:
            target = debt_accrued
            harvest = target * redeem_r         # best-effort: Rest bleibt drin
            debt_accrued -= harvest             # und verzinst sich weiter
            harvest_pool += harvest
            harvest_total += harvest
            debt_val = debt_cost_basis + debt_accrued

        # ---- Bandlogik: geernteter Ertrag -> BTC ---------------------------
        is_dca = (is_me if h_freq == "M" else is_qe)
        if is_dca and harvest_pool > 0:
            tot = debt_val + b_val + cash + harvest_pool
            w = (b_val / tot) if tot > 0 else 0.0
            rate = boost_rate if w < lo else (0.0 if w > up else base_rate)
            invest = harvest_pool * rate
            if invest > 0 and px > 0:
                btc_u_dca += (invest * (1 - tx)) / px
                btc_dca_invested += invest
                buys = invest
            cash += harvest_pool - invest
            harvest_pool = 0.0
            b_val = (btc_u_init + btc_u_dca) * px

        # ---- Sell-Regel: BTC über Cap -> zurück auf exakt Cap --------------
        is_sell = (is_me if c_freq == "M" else is_qe)
        if sell_on and is_sell and px > 0:
            tot = debt_val + b_val + cash
            if tot > 0 and (b_val / tot) > up:
                target_b = up * (debt_val + cash) / (1 - up) if up < 1 else b_val
                excess = max(b_val - target_b, 0.0)
                if excess > 0:
                    _u = btc_u_init + btc_u_dca
                    _f = (btc_u_init / _u) if _u > 0 else 0.0
                    _sold_u = excess / px
                    btc_u_init -= _sold_u * _f
                    btc_u_dca -= _sold_u * (1 - _f)
                    _net = excess * (1 - tx)
                    btc_init_realized += _net * _f
                    btc_dca_realized += _net * (1 - _f)
                    cash += _net
                    sells = excess
                    b_val = (btc_u_init + btc_u_dca) * px

        # ---- Gebühren-Wasserfall: Cash -> BTC. Das KREDITBUCH NIE. ---------
        def _pay(amount):
            nonlocal cash, btc_u_init, btc_u_dca, b_val, fee_paid, fee_btc_sold
            nonlocal fee_from_cash, fee_from_btc, btc_init_realized, btc_dca_realized
            if amount <= 1e-9:
                return 0.0
            from_cash = min(cash, amount)
            cash -= from_cash
            fee_paid += from_cash
            fee_from_cash += from_cash
            short = amount - from_cash
            if short > 1e-9 and b_val > 0 and px > 0:
                gross = min(b_val, short / (1 - tx))
                _u = btc_u_init + btc_u_dca
                _f = (btc_u_init / _u) if _u > 0 else 0.0
                _su = gross / px
                btc_u_init -= _su * _f
                btc_u_dca -= _su * (1 - _f)
                proceeds = gross * (1 - tx)
                btc_init_realized += proceeds * _f
                btc_dca_realized += proceeds * (1 - _f)
                fee_btc_sold += gross
                fee_paid += proceeds
                fee_from_btc += proceeds
                short -= proceeds
                b_val = (btc_u_init + btc_u_dca) * px
            return max(short, 0.0)

        fee_due = ((is_me and mgmt_freq == "M") or (is_qe and mgmt_freq == "Q"))
        if fee_due and mgmt_fee > 0:
            per_year = 12.0 if mgmt_freq == "M" else 4.0
            aum = debt_val + b_val + cash + harvest_pool
            amt = aum * (mgmt_fee / per_year) + fee_debt
            fee_debt = 0.0
            _before = fee_paid
            fee_debt = _pay(amt)
            mgmt_paid_row = fee_paid - _before
            total_mgmt += mgmt_paid_row

        if (is_me and d.month in cryst_months) or i == len(idx) - 1:
            if perf_fee > 0:
                nav_now = debt_val + b_val + cash + harvest_pool
                _frac = max((d - prev_cryst_date).days, 0) / 365.25
                thr = hwm * (1.0 + hwm_hurdle * _frac)
                excess_p = 0.0
                if hurdle_type == "No Hurdle (HWM only)":
                    if nav_now > hwm:
                        excess_p = nav_now - hwm
                elif hurdle_type == "Soft Hurdle":
                    if nav_now > thr:
                        excess_p = nav_now - hwm
                else:
                    if nav_now > thr:
                        excess_p = nav_now - thr
                if excess_p > 1e-9:
                    _amt = excess_p * perf_fee
                    _short = _pay(_amt)
                    perf_paid_row = _amt - _short
                    total_perf += perf_paid_row
                _after = debt_val + b_val + cash + harvest_pool
                if _after > hwm:
                    hwm = _after
                prev_cryst_date = d

        rows.append({
            "date": d,
            "debt_value": debt_val,
            "btc_value": b_val,
            "cash": cash + harvest_pool,
            "total_value": debt_val + b_val + cash + harvest_pool,
            "harvest": harvest if is_me else np.nan,
            "debt_cost_basis": debt_cost_basis,
            "debt_accrued": debt_accrued,
            "subscriptions": subs,
            "btc_buys": buys,
            "btc_sells": sells,
            "mgmt_fee_paid": fee_paid,
            "mgmt_fee_only": mgmt_paid_row,
            "perf_fee_only": perf_paid_row,
            "fee_btc_sold": fee_btc_sold,
            "fee_from_cash": fee_from_cash,
            "fee_from_btc": fee_from_btc,
            "fee_debt": fee_debt,
            "cash_floor": max(fee_floor, 0.0),
        })

    out = pd.DataFrame(rows).set_index("date")

    # ================== RENDITEZERLEGUNG (ATTRIBUTION) ====================
    # Identität (auf den Rappen):
    #   NAV_end − Startkapital − Zeichnungen
    #     = Ertragsanfall − Kreditausfälle + BTC(Start) + BTC(DCA) + Cash-Zins − Gebühren
    _px_end = _btc_v[-1]
    _bi_end = btc_u_init * _px_end
    _bd_end = btc_u_dca * _px_end
    btc_init_gain = (_bi_end + btc_init_realized) - btc_init_invested
    btc_dca_gain = (_bd_end + btc_dca_realized) - btc_dca_invested
    _btot = btc_init_gain + btc_dca_gain
    dca_share = (btc_dca_gain / _btot) if abs(_btot) > 1e-9 else float("nan")

    nav_end = float(out["total_value"].iloc[-1])
    pnl = nav_end - cap - subs_total          # Zeichnungen sind kein Gewinn!
    recon = (accrued_total - loss_total + btc_init_gain + btc_dca_gain
             + cash_interest_total - (total_mgmt + total_perf))

    out.attrs["total_mgmt"] = total_mgmt
    out.attrs["total_perf"] = total_perf
    out.attrs["fee_debt_final"] = fee_debt
    out.attrs["attribution"] = {
        "debt_income": accrued_total,          # Ertragsanfall im Kreditbuch (brutto)
        "credit_losses": -loss_total,          # Kreditausfälle (negativ)
        "btc_initial_gain": btc_init_gain,
        "btc_dca_gain": btc_dca_gain,
        "cash_interest": cash_interest_total,
        "fees": -(total_mgmt + total_perf),
        "total_pnl": pnl,
        "reconciliation_error": recon - pnl,
        "dca_share": dca_share,
        "btc_initial_invested": btc_init_invested,
        "btc_dca_invested": btc_dca_invested,
        "harvest_total": harvest_total,        # tatsächlich realisierter "Coupon"
        "accrued_total": accrued_total,
        "unharvested": debt_accrued,           # klemmt im Zertifikat
        "harvest_ratio": (harvest_total / accrued_total) if accrued_total > 0 else float("nan"),
        "subscriptions": subs_total,
        "cash_drag": -cash_drag,
        "years": max((idx[-1] - idx[0]).days / 365.25, 1e-9),
    }
    return out


def run_debt_only(idx, params):
    """Benchmark: identisches Kreditbuch OHNE Bitcoin — der Ertrag wird
    thesauriert (nichts wird geerntet, nichts verkauft). Isoliert exakt den
    Beitrag der Bitcoin-Allokation."""
    idx = pd.DatetimeIndex(idx).normalize().unique().sort_values()
    cap = float(params["initial_capital"])
    ny = float(params["net_yield"])
    loss_r = float(params.get("credit_loss_rate", 0.0))
    sub_amt = float(params.get("subscription_amount", 0.0))
    sub_freq = params.get("subscription_freq", "N")
    v = cap
    vals = []
    for i, d in enumerate(idx):
        v += v * ((ny - loss_r) / 365.0)
        is_me = (i == len(idx) - 1) or (idx[i + 1].to_period("M") != d.to_period("M"))
        is_qe = is_me and d.month in (3, 6, 9, 12)
        is_ye = is_me and d.month == 12
        if sub_amt > 0 and ((sub_freq == "M" and is_me) or (sub_freq == "Q" and is_qe)
                            or (sub_freq == "A" and is_ye)):
            v += sub_amt
        vals.append(v)
    return pd.Series(vals, index=idx)

# ===========================================================================
# UI
# ===========================================================================
st.markdown(
    f"""<div class='oak-topbar'><span>Oakwood Capital</span>
    <span class='stamp'>Intern · Vertraulich</span></div>""",
    unsafe_allow_html=True)
st.markdown(
    f"<h1 style='font-family:\"Cormorant Garamond\",serif; color:{OAK_CREAM}; "
    f"font-size:46px; font-weight:500; margin:18px 0 4px 0; "
    f"line-height:1.1;'>OAK Swiss Private Debt / Bitcoin</h1>",
    unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{OAK_SAGE}; font-size:16px; max-width:900px; "
    f"margin-bottom:26px;'>Immobilienbesichertes Schweizer Kreditbuch mit "
    f"struktureller Bitcoin-Allokation. Der thesaurierende Ertrag wird monatlich "
    f"geerntet und über Bandregeln in Bitcoin investiert — das eingesetzte "
    f"Kapital bleibt unangetastet. Parametrische Simulation.</p>",
    unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Parameter")

    st.markdown("### Stress-Test-Szenarien")
    st.caption("Historische Krisenfenster — setzt den Backtest-Zeitraum.")
    _sc = [("COVID Crash (2020)", "2020-01-01", "2020-12-31"),
           ("BTC Bear Market (2022)", "2022-01-01", "2022-12-31"),
           ("Banking Crisis / CS (2023)", "2023-01-01", "2023-12-31"),
           ("Full History (2018–heute)", "2018-01-01", None)]
    _c = st.columns(2)
    for _i, (_lab, _s, _e) in enumerate(_sc):
        if _c[_i % 2].button(_lab, use_container_width=True, key=f"pd_sc_{_i}"):
            st.session_state["pd_scen_start"] = pd.Timestamp(_s).date()
            st.session_state["pd_scen_end"] = (pd.Timestamp(_e).date() if _e
                                               else date.today())

    st.markdown("### Backtest-Zeitraum")
    start_date = st.date_input("Startdatum",
                               st.session_state.get("pd_scen_start", date(2019, 1, 1)),
                               key="pd_start")
    end_date = st.date_input("Enddatum",
                             st.session_state.get("pd_scen_end", date.today()),
                             key="pd_end")
    initial_capital = st.number_input("Anfangskapital (CHF)", min_value=100_000,
                                      max_value=10_000_000_000, value=10_000_000,
                                      step=100_000)

    st.markdown("### Allokation")
    initial_btc_pct = st.slider("Initial BTC Allokation (%)", 0, 30, 10, 1,
                                help="0% = reine Yield Bridge (BTC wird "
                                     "ausschliesslich aus dem geernteten Ertrag "
                                     "aufgebaut).") / 100.0
    initial_cash_pct = st.slider("Initiale Cash-Reserve (%)", 0, 25, 5, 1,
                                 help="Polster zur Gebührendeckung. Reicht es "
                                      "nicht, wird BTC verkauft — nie das "
                                      "Kreditbuch.") / 100.0
    cash_rate = st.slider("Cash-Verzinsung (% p.a., SARON-nah)", 0.0, 3.0, 1.0,
                          0.05) / 100.0
    lower_threshold = st.slider("Untere BTC-Schwelle (%)", 0, 40, 10, 1) / 100.0
    upper_threshold = st.slider("Obere BTC-Schwelle (%)", 5, 60, 25, 1) / 100.0
    if lower_threshold >= upper_threshold:
        st.error("Untere Schwelle muss kleiner als die obere sein.")
        st.stop()

    st.markdown("### Kreditbuch (Debt-Sleeve)")
    net_yield = st.slider("Nettorendite (% p.a.)", 0.0, 10.0, 5.0, 0.1,
                          help="Nach Investor Fee des Underlyings (LEND "
                               "Hypovest: 0.95% p.a. auf 5.5–6.5% brutto). "
                               "Parametrisch — es gibt keine verwertbare "
                               "Kurshistorie.") / 100.0
    credit_loss_rate = st.slider("Kreditausfälle (% p.a.)", 0.0, 5.0, 0.5, 0.1,
                                 help="Nachrangige Hypotheken tragen das "
                                      "First-Loss-Risiko. Stressparameter.") / 100.0
    redemption_rate = st.slider("Redemption-Erfolgsquote (%)", 0, 100, 100, 5,
                                help="Rücknahmen erfolgen best-effort (kein "
                                     "Sekundärmarkt). Was nicht geerntet werden "
                                     "kann, bleibt im Zertifikat und verzinst "
                                     "sich weiter — es geht NICHT verloren.") / 100.0

    st.markdown("### Zeichnungen (Wachstum)")
    subscription_freq_label = st.selectbox(
        "Rhythmus", ["keine", "monatlich", "quartalsweise", "jährlich"], index=0)
    subscription_freq = {"keine": "N", "monatlich": "M",
                         "quartalsweise": "Q", "jährlich": "A"}[subscription_freq_label]
    subscription_amount = st.number_input(
        "Betrag je Zeichnung (CHF)", min_value=0, value=0, step=100_000,
        help="Neue Zeichnungen fliessen vollständig ins Kreditbuch (Kern) — "
             "nie direkt in Bitcoin. Sie erhöhen dadurch die Ertragsbasis und "
             "damit die Ernte, die den DCA speist.")

    st.markdown("### Ertragsallokation")
    base_invest_rate = st.slider("Basis-Investitionsrate (%)", 0, 100, 50, 5) / 100.0
    boost_invest_rate = st.slider("Boost-Rate unter unterer Schwelle (%)",
                                  0, 100, 100, 5) / 100.0
    harvest_to_btc_freq = "M" if st.selectbox(
        "Ernte → BTC", ["monatlich", "quartalsweise"], index=0) == "monatlich" else "Q"
    btc_to_cash_freq = "M" if st.selectbox(
        "BTC → Cash (Sell-Prüfung)", ["monatlich", "quartalsweise"],
        index=1) == "monatlich" else "Q"
    sell_on_upper = st.checkbox(
        "BTC bei Überschreiten der oberen Schwelle auf den Cap zurückführen",
        value=True)

    st.markdown("### Benchmark")
    bench_ticker = st.text_input(
        "Anleihen-Benchmark (Yahoo-Ticker)", value="AGG",
        help="Default AGG = iShares Core U.S. Aggregate Bond (grösster "
             "Anleihen-ETF). Wird in CHF umgerechnet — die FX-Bewegung ist "
             "dadurch im Benchmark enthalten.")

    st.markdown("### Risikoanalyse")
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 1.0, 0.25) / 100.0

    st.markdown("### Kosten & Gebühren")
    tx_cost_bps = st.slider("Transaction Cost (bps per trade)", 0, 50, 10, 1)
    mgmt_fee = st.slider("Management Fee (% p.a.)", 0.0, 3.0, 1.5, 0.05) / 100.0
    mgmt_fee_freq = "M" if st.selectbox("Management Fee Verbuchung",
                                        ["monatlich", "quartalsweise"],
                                        index=0) == "monatlich" else "Q"
    perf_fee = st.slider("Performance Fee (%)", 0, 30, 15, 1) / 100.0
    hurdle_type = st.selectbox("Hurdle Type",
                               ["Hard Hurdle", "Soft Hurdle", "No Hurdle (HWM only)"])
    hurdle = st.slider("Hurdle Rate Year 1 (%)", 0.0, 15.0, 5.0, 0.5) / 100.0
    crystallization_freq = st.selectbox("Performance Fee Crystallization",
                                        ["Quarterly", "Semi-Annual", "Annual"])

    run_bt = st.button("Backtest starten", type="primary", use_container_width=True)

if run_bt:
    st.session_state["pd_has_run"] = True
if not st.session_state.get("pd_has_run"):
    st.info("Parameter links einstellen und **Backtest starten**.")
    st.stop()

# ---- Daten ---------------------------------------------------------------
with st.spinner("Lade Bitcoin- und FX-Daten…"):
    _btc_usd = _clean_index(fetch_series("BTC-USD", start_date, end_date))
    _fx = _clean_index(fetch_series("USDCHF=X", start_date, end_date))
    _bench_raw = _clean_index(fetch_series(bench_ticker, start_date, end_date))

if _btc_usd.empty or _fx.empty:
    st.error("⚠️ Keine BTC- oder FX-Daten erhalten (temporäre Yahoo-Störung oder "
             "Rate-Limit). Bitte in ein paar Minuten erneut versuchen.")
    st.stop()

_common = _btc_usd.index.intersection(_fx.index)
btc_chf = (_btc_usd.loc[_common] * _fx.loc[_common]).dropna()
idx = pd.date_range(btc_chf.index[0], btc_chf.index[-1], freq="D")
btc_chf = btc_chf.reindex(idx).ffill().bfill()

params = dict(
    initial_capital=float(initial_capital), initial_btc_pct=initial_btc_pct,
    initial_cash_pct=initial_cash_pct, net_yield=net_yield,
    credit_loss_rate=credit_loss_rate, redemption_rate=redemption_rate,
    subscription_amount=float(subscription_amount),
    subscription_freq=subscription_freq,
    lower_threshold=lower_threshold, upper_threshold=upper_threshold,
    base_invest_rate=base_invest_rate, boost_invest_rate=boost_invest_rate,
    harvest_to_btc_freq=harvest_to_btc_freq, btc_to_cash_freq=btc_to_cash_freq,
    sell_on_upper=sell_on_upper, mgmt_fee=mgmt_fee, mgmt_fee_freq=mgmt_fee_freq,
    perf_fee=perf_fee, hwm_hurdle=hurdle, hurdle_type=hurdle_type,
    crystallization_freq=crystallization_freq, tx_cost_bps=float(tx_cost_bps),
    cash_rate=cash_rate, risk_free_rate=risk_free_rate)

ts = run_pd_btc(btc_chf, idx, params)
if ts.empty:
    st.error("Simulation lieferte keine Daten (zu kurzer Zeitraum?).")
    st.stop()

net = ts["total_value"]
bench_debt = run_debt_only(idx, params)
_att = ts.attrs["attribution"]
total_mgmt = ts.attrs["total_mgmt"]
total_perf = ts.attrs["total_perf"]

# Brutto-Rekonstruktion (Fees laufen in-engine als echter Cash-Abfluss)
gross = net + ts["mgmt_fee_paid"].cumsum()
years = _att["years"]
# WICHTIG: Zeichnungen sind kein Gewinn — die CAGR-Basis muss sie enthalten.
_cap_base = initial_capital + _att["subscriptions"]
net_cagr = (net.iloc[-1] / _cap_base) ** (1 / years) - 1 if _cap_base > 0 else 0.0
gross_cagr = (gross.iloc[-1] / _cap_base) ** (1 / years) - 1 if _cap_base > 0 else 0.0
debt_cagr = (bench_debt.iloc[-1] / _cap_base) ** (1 / years) - 1 if _cap_base > 0 else 0.0
fee_drag = gross_cagr - net_cagr
btc_contrib = net_cagr - debt_cagr

# Anleihen-Benchmark auf eigenem Handelskalender, rebasiert
bench_bond = None
if not _bench_raw.empty:
    _w = _bench_raw[(_bench_raw.index >= net.index[0])
                    & (_bench_raw.index <= net.index[-1])].dropna()
    _fxw = _fx.reindex(_w.index).ffill().bfill()
    if len(_w) > 1 and _w.iloc[0] > 0:
        _wc = _w * _fxw          # in CHF
        bench_bond = _wc / _wc.iloc[0] * initial_capital

m = compute_risk_metrics(net, risk_free_rate, base_value=_cap_base)

# ==========================================================================
st.markdown("## Performance-Übersicht")
if _att["subscriptions"] > 0:
    st.caption(f"CAGR-Basis: Anfangskapital + Zeichnungen = "
               f"{fmt_chf(_cap_base)} (Zeichnungen sind kein Gewinn).")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Strategie (netto)", fmt_chf(net.iloc[-1]),
          f"{(net.iloc[-1]/_cap_base - 1)*100:+.1f}%", delta_color="off")
c2.metric("Netto-CAGR", f"{net_cagr*100:.2f}%",
          f"nach allen Gebühren · {years:.1f} Jahre", delta_color="off")
c3.metric("Nur Kreditbuch (ohne BTC)", fmt_chf(bench_debt.iloc[-1]),
          f"{debt_cagr*100:.2f}% p.a.", delta_color="off")
c4.metric("BTC-Beitrag", f"{btc_contrib*100:+.2f}% p.a.",
          "vs. identisches Modell ohne BTC", delta_color="off")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Sharpe Ratio*", f"{m['sharpe']:.2f}")
c6.metric("Max Drawdown*", f"{m['max_dd']*100:.2f}%")
c7.metric("Volatilität*", f"{m['vol_ann']*100:.2f}%")
_wb = ts["btc_value"].iloc[-1] / net.iloc[-1] if net.iloc[-1] > 0 else 0
c8.metric("BTC-Quote (aktuell)", f"{_wb*100:.1f}%",
          f"Band {lower_threshold*100:.0f}–{upper_threshold*100:.0f}%",
          delta_color="off")
st.caption("*Der Debt-Sleeve hat konstruktionsbedingt KEINE Kapitalwert­"
           "schwankung (at par). Volatilität, Sharpe und Drawdown der "
           "Gesamtstrategie sind dadurch nach unten verzerrt und nicht mit "
           "marktbewerteten Strategien vergleichbar.")

c9, c10, c11, c12 = st.columns(4)
c9.metric("Management-Gebühren", fmt_chf(total_mgmt),
          f"{mgmt_fee*100:.2f}% p.a. auf NAV", delta_color="off")
c10.metric("Performance-Gebühren", fmt_chf(total_perf),
           f"{perf_fee*100:.0f}% × Excess", delta_color="off")
c11.metric("Gebühren total", fmt_chf(total_mgmt + total_perf),
           f"Gebührenlast: {fee_drag*100:.2f}% p.a.", delta_color="off")
c12.metric("Nettorendite (Input)", f"{net_yield*100:.1f}% p.a.",
           "parametrisch, auf Kostenbasis", delta_color="off")

# ==========================================================================
# ERTRAGS-ERNTE & REDEMPTION-STRESS — die produktspezifische Kernsektion
# ==========================================================================
st.markdown("## Ertrags-Ernte & Redemption-Stress")
st.markdown(
    "<p style='color:#A9B5A4;margin-top:-6px'>Das Underlying ist "
    "<strong>thesaurierend</strong> — es schüttet nichts aus. Der Ertrag wird "
    "monatlich geerntet, indem exakt der aufgelaufene NAV-Zuwachs über der "
    "Kostenbasis redimiert wird. Das eingesetzte Kapital wird nie angetastet. "
    "Klemmen die Rücknahmen, bleibt der Ertrag im Zertifikat und verzinst sich "
    "weiter — er geht nicht verloren.</p>", unsafe_allow_html=True)

h1, h2, h3, h4 = st.columns(4)
with h1:
    st.metric("Ertragsanfall (brutto)", fmt_chf(_att["accrued_total"]))
    st.caption("im Kreditbuch aufgelaufen")
with h2:
    st.metric("Davon geerntet", fmt_chf(_att["harvest_total"]))
    _hr = _att["harvest_ratio"]
    st.caption("n/a" if _hr != _hr else f"{_hr*100:.0f}% des Ertragsanfalls")
with h3:
    st.metric("Ungeerntet (klemmt)", fmt_chf(_att["unharvested"]))
    st.caption("bleibt im Zertifikat, verzinst weiter")
with h4:
    _cb = ts["debt_cost_basis"]
    _drift = _cb.iloc[-1] - _cb.iloc[0] - _att["subscriptions"]
    st.metric("Kernkapital angetastet?",
              "Nein" if abs(_drift) < 1.0 else f"{_drift:+,.0f}")
    st.caption("Kostenbasis-Drift (ohne Zeichnungen)")

if redemption_rate < 1.0:
    st.warning(
        f"⚠️ **Redemption-Erfolgsquote {redemption_rate*100:.0f}%** — es werden "
        f"nur {redemption_rate*100:.0f}% des monatlichen Ertragszuwachses "
        f"tatsächlich redimiert. Aktuell klemmen "
        f"{fmt_chf(_att['unharvested'])} im Zertifikat. Wichtig zur Einordnung: "
        "dieser Betrag ist **nicht verloren**, er verzinst sich weiter und wird "
        "geerntet, sobald Rücknahmen wieder gefüllt werden. Der DCA pausiert "
        "lediglich — und zwar typischerweise genau in Stressphasen, in denen "
        "aggressive Zukäufe ohnehin fragwürdig wären.")

fig_h = go.Figure()
fig_h.add_trace(go.Scatter(x=ts.index, y=ts["debt_cost_basis"],
                           name="Kostenbasis (eingesetztes Kapital)",
                           line=dict(color=OAK_SAGE, width=2)))
fig_h.add_trace(go.Scatter(x=ts.index, y=ts["debt_value"],
                           name="Kreditbuch-NAV (inkl. aufgelaufener Ertrag)",
                           line=dict(color=OAK_GOLD, width=2)))
fig_h = style_plotly(fig_h, height=340)
fig_h.update_layout(margin=dict(l=70, r=30, t=10, b=60),
                    legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))
st.markdown("##### Kostenbasis vs. NAV — der Abstand ist der ungeerntete Ertrag")
st.plotly_chart(fig_h, use_container_width=True)

# ==========================================================================
# RENDITEZERLEGUNG
# ==========================================================================
st.markdown("## Renditezerlegung")
st.markdown(
    "<p style='color:#A9B5A4;margin-top:-6px'>Zerlegung des Ergebnisses in seine "
    "Quellen. Zeichnungen sind dabei <strong>kein Gewinn</strong> und werden "
    "herausgerechnet. Die Positionen summieren sich exakt auf die Gesamt-P&amp;L.</p>",
    unsafe_allow_html=True)


def _pp(v):
    return (v / _cap_base) / years * 100


_rows = [
    ("Ertragsanfall Kreditbuch (brutto)", _att["debt_income"]),
    ("Kreditausfälle", _att["credit_losses"]),
    ("Bitcoin — Startallokation (Tag 1)", _att["btc_initial_gain"]),
    ("Bitcoin — ertragsfinanzierter DCA (Yield Bridge)", _att["btc_dca_gain"]),
    ("Cash-Zins (Puffer)", _att["cash_interest"]),
    ("Gebühren", _att["fees"]),
]
_h = ["<table class='oak-metrics-table'><thead><tr><th>Beitrag</th><th>CHF</th>"
      "<th>%-Punkte p.a.</th></tr></thead><tbody>"]
for _lab, _v in _rows:
    _c = OAK_GOLD if "DCA" in _lab else OAK_CREAM
    _h.append(f"<tr><td class='metric-label'>{_lab}</td>"
              f"<td class='strategy-col' style='color:{_c}'>{_v:+,.0f}</td>"
              f"<td style='color:{_c}'>{_pp(_v):+.2f}</td></tr>")
_h.append(f"<tr class='oak-section'><td>Total (= NAV − Startkapital − Zeichnungen)</td>"
          f"<td>{_att['total_pnl']:+,.0f}</td><td>{_pp(_att['total_pnl']):+.2f}</td></tr>")
_h.append("</tbody></table>")
st.markdown("".join(_h), unsafe_allow_html=True)

d1, d2, d3 = st.columns(3)
_ds = _att["dca_share"]
with d1:
    st.metric("DCA-Anteil am BTC-Gewinn",
              "n/a" if _ds != _ds else f"{_ds*100:.1f}%")
    st.caption("DCA / (DCA + Startallokation)")
with d2:
    st.metric("BTC Startallokation", fmt_chf(_att["btc_initial_invested"]))
    st.caption("am Tag 1 investiert")
with d3:
    st.metric("BTC via Ernte investiert", fmt_chf(_att["btc_dca_invested"]))
    st.caption("über die gesamte Laufzeit")

if _ds == _ds:
    if _ds < 0.30:
        st.warning(
            f"⚠️ **Der DCA-Anteil liegt bei {_ds*100:.1f}%.** Der überwiegende "
            "Teil des Bitcoin-Gewinns stammt aus der Startallokation vom ersten "
            "Tag, nicht aus dem ertragsfinanzierten DCA. Wichtig zur Einordnung: "
            "der DCA-Anteil ist **invers zum Einstiegsglück** — je schlechter der "
            "Einstiegszeitpunkt, desto grösser der Beitrag des DCA. Ein tiefer "
            "Wert zeigt primär, dass der Zeitraum für die Startallokation günstig "
            "lag. Hebel: tiefere Startallokation oder höhere Zeichnungen (mehr "
            "Ertragsbasis → mehr Ernte → mehr DCA).")
    else:
        st.success(f"✅ Der DCA-Anteil liegt bei {_ds*100:.1f}% — der "
                   "Ertragsmechanismus trägt den Bitcoin-Beitrag substanziell.")
st.caption(f"Abstimmdifferenz der Zerlegung: "
           f"{_att['reconciliation_error']:+.2f} CHF · Cash-Drag (Memo): "
           f"{fmt_chf(_att['cash_drag'])}")

# ==========================================================================
st.markdown("## Portfolioentwicklung vs. Benchmarks")
fig = go.Figure()
fig.add_trace(go.Scatter(x=net.index, y=net.values, name="Strategie (netto)",
                         line=dict(color=OAK_GOLD, width=2.4)))
fig.add_trace(go.Scatter(x=bench_debt.index, y=bench_debt.values,
                         name="Nur Kreditbuch (gleiches Modell, ohne BTC)",
                         line=dict(color=OAK_SAGE, width=1.6, dash="dash")))
if bench_bond is not None:
    fig.add_trace(go.Scatter(x=bench_bond.index, y=bench_bond.values,
                             name=f"{bench_ticker} (Anleihen, in CHF)",
                             line=dict(color="#7FA7C4", width=1.4)))
fig = style_plotly(fig, height=460)
fig.update_yaxes(title_text="Wert (CHF)", tickformat=",.0f")
st.plotly_chart(fig, use_container_width=True)

st.markdown("## Sleeve-Entwicklung")
fig_s = go.Figure()
fig_s.add_trace(go.Scatter(x=ts.index, y=ts["debt_value"], name="Kreditbuch",
                           line=dict(color=OAK_SAGE, width=2)))
fig_s.add_trace(go.Scatter(x=ts.index, y=ts["btc_value"], name="Bitcoin",
                           line=dict(color=OAK_BTC, width=2)))
fig_s.add_trace(go.Scatter(x=ts.index, y=ts["cash"], name="CHF Cash",
                           line=dict(color=OAK_CREAM_DIM, width=2)))
fig_s = style_plotly(fig_s, height=380)
fig_s.update_yaxes(title_text="Wert (CHF)", tickformat=",.0f")
st.plotly_chart(fig_s, use_container_width=True)

st.markdown("### BTC-Quote vs. Schwellenwerte")
_wser = ts["btc_value"] / ts["total_value"] * 100
fig_w = go.Figure()
fig_w.add_trace(go.Scatter(x=ts.index, y=_wser, name="BTC in % des NAV",
                           line=dict(color=OAK_BTC, width=2)))
fig_w.add_hline(y=lower_threshold * 100, line=dict(color=OAK_SAGE, dash="dot"))
fig_w.add_hline(y=upper_threshold * 100, line=dict(color=OAK_GOLD, dash="dot"))
_sd = ts.index[ts["btc_sells"] > 0]
if len(_sd):
    fig_w.add_trace(go.Scatter(x=_sd, y=_wser.loc[_sd], mode="markers",
                               name="Verkauf → Cash",
                               marker=dict(symbol="x", size=7, color=OAK_BTC)))
fig_w = style_plotly(fig_w, height=320)
fig_w.update_yaxes(title_text="BTC-Quote (%)")
st.plotly_chart(fig_w, use_container_width=True)

# ==========================================================================
st.markdown("## Risikoanalyse")
st.caption("Die Kennzahlen der Gesamtstrategie sind durch den nicht "
           "marktbewerteten Debt-Sleeve nach unten verzerrt. Für den "
           "Debt-Sleeve selbst werden Volatilität, Sharpe, Sortino und "
           "Drawdown bewusst NICHT ausgewiesen — sie wären konstruktionsbedingt "
           "sinnlos (Volatilität exakt null).")
_dd = compute_drawdown(net)
fig_d = go.Figure()
fig_d.add_trace(go.Scatter(x=_dd.index, y=_dd.values * 100, name="Strategie (netto)",
                           line=dict(color=OAK_GOLD, width=1.6), fill="tozeroy",
                           fillcolor="rgba(201,169,97,0.15)"))
fig_d = style_plotly(fig_d, height=300)
fig_d.update_yaxes(title_text="Drawdown (%)")
st.markdown("### Drawdown-Analyse")
st.plotly_chart(fig_d, use_container_width=True)

_rm = [("Gesamtrendite", f"{(net.iloc[-1]/_cap_base - 1)*100:.2f}%"),
       ("CAGR (netto)", f"{net_cagr*100:.2f}%"),
       ("Volatilität*", f"{m['vol_ann']*100:.2f}%"),
       ("Max Drawdown*", f"{m['max_dd']*100:.2f}%"),
       ("Sharpe Ratio*", f"{m['sharpe']:.2f}"),
       ("Sortino Ratio*", f"{m['sortino']:.2f}")]
_ht = ["<table class='oak-metrics-table'><thead><tr><th>Kennzahl</th>"
       "<th>Strategie (netto)</th><th>Nur Kreditbuch</th></tr></thead><tbody>"]
for _lab, _v in _rm:
    _bv = (f"{(bench_debt.iloc[-1]/_cap_base - 1)*100:.2f}%" if "Gesamt" in _lab
           else f"{debt_cagr*100:.2f}%" if "CAGR" in _lab else "—")
    _ht.append(f"<tr><td class='metric-label'>{_lab}</td>"
               f"<td class='strategy-col'>{_v}</td><td>{_bv}</td></tr>")
_ht.append("</tbody></table>")
st.markdown("".join(_ht), unsafe_allow_html=True)

with st.expander("Monatliche Ernte (Ertragsrealisierung)"):
    _hv = ts["harvest"].dropna()
    st.bar_chart(_hv)
    st.caption(f"Summe geerntet: {fmt_chf(_hv.sum())} · "
               f"Durchschnitt/Monat: {fmt_chf(_hv.mean())}")

st.markdown(
    f"""<div class='oak-footer'>
    Zu illustrativen Zwecken · Keine Anlageberatung · Parametrische Simulation ·
    Vergangene Wertentwicklung ist kein Indikator für zukünftige Ergebnisse
    <span class='oak-mark'>Oakwood Capital · Quantitatives Research</span>
    </div>""", unsafe_allow_html=True)
