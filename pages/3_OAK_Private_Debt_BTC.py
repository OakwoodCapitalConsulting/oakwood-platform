"""
OAK Swiss Private Debt / Bitcoin — AMC Backtesting

Konzept (OAK Yield Bridge, dritte Anwendung):
  * Kern: diversifiziertes, immobilienbesichertes Schweizer Kreditbuch
    (Referenz: LEND Hypovest, ISIN CH1357099691 — nachrangige Hypotheken,
    Zielrendite 5.5-6.5% brutto). Parametrisch modelliert: die Nettorendite
    ist ein regelbarer Input, es gibt KEINE Marktdaten-Historie (Emission
    Juli 2024) und KEINE Kapitalwertschwankung (at par, held to maturity).

  * BESONDERHEIT: Das Underlying ist THESAURIEREND — es schüttet nichts aus.
    Der Ertrag wird über eine JÄHRLICHE ERTRAGS-ERNTE (Default: Januar)
    realisiert: es werden exakt so viele Anteile redimiert, wie dem seit der
    letzten Ernte aufgelaufenen NAV-Zuwachs über der Kostenbasis entsprechen.
    Nie das Kapital. Der geerntete Betrag wird anschliessend rollierend über
    dca_months (Default 12) in Bitcoin investiert. Ökonomisch identisch mit
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
    btc_u_subs = 0.0                # DRITTES LOT: zeichnungsfinanziertes BTC
    btc_init_invested = btc_chf0
    btc_dca_invested = 0.0
    btc_subs_invested = 0.0
    btc_init_realized = 0.0
    btc_dca_realized = 0.0
    btc_subs_realized = 0.0
    # ZEICHNUNGS-ALLOKATION (Open-End!): neue Gelder MÜSSEN pro-rata zu den
    # AKTUELLEN Portfoliogewichten investiert werden. Alles andere verwässert
    # die Bestandsinvestoren: flösse alles ins Kreditbuch, sänke die BTC-Quote
    # mit jeder Zeichnung — die Strategie würde von Mittelzuflüssen gesteuert
    # statt von ihren Regeln.
    sub_mode = params.get("subscription_mode", "prorata")   # prorata|debt|target
    cash_floor_pct = float(params.get("cash_floor_pct", 0.05))  # Zielquote Puffer
    cash_topup_total = 0.0
    # INVESTOREN-RÜCKNAHMEN (Open-End). Liquiditäts-Wasserfall:
    #   Cash -> BTC (NUR bis zur unteren Bandgrenze!) -> Kreditbuch (best effort)
    #   -> Rest wird GEGATED (anteilig ausgesetzt).
    # Bitcoin ist der Liquiditäts-Sleeve, wird aber nicht geplündert: sonst
    # entsteht ein First-Mover-Vorteil und die Bleibenden halten den illiquiden
    # Rest. Genau daran sind offene Immobilienfonds gescheitert.
    principal_redeem_r = float(params.get("principal_redeem_rate", 1.0))
    out_pct = float(params.get("outflow_pct", 0.0))       # je Termin, % des NAV
    out_freq = params.get("outflow_freq", "N")            # M|Q|A|N
    outflow_paid = 0.0
    outflow_gated = 0.0
    n_gates = 0

    # ---- Debt-Sleeve (rein parametrisch, kein Marktdaten-Index) ------------
    debt_cost_basis = cap - btc_chf0 - cash0   # tatsächlich eingesetztes Kapital
    debt_accrued = 0.0                         # aufgelaufener, noch nicht geernteter Ertrag
    harvest_total = 0.0                        # tatsächlich geerntet (= der "Coupon")
    accrued_total = 0.0                        # gesamter Ertragsanfall (auch ungeerntet)
    loss_total = 0.0                           # Kreditausfälle
    subs_total = 0.0                           # neue Zeichnungen

    cash = cash0
    pending_dca = 0.0      # geerntet, noch nicht investiert — TEIL DES NAV
    fee_floor = cash0
    fee_debt = 0.0
    cash_interest_total = 0.0
    cash_drag = 0.0

    ny = float(params["net_yield"])
    loss_r = float(params.get("credit_loss_rate", 0.0))
    redeem_r = float(params.get("redemption_rate", 1.0))
    # ERNTE-RHYTHMUS: Das Underlying schüttet nicht aus. Monatliche Rücknahmen
    # wären operativ unsinnig und teuer. Stattdessen wird EINMAL PRO PERIODE
    # (Default: jährlich im Januar) der über das Vorjahr aufgelaufene Ertrag
    # redimiert und anschliessend ROLLIEREND über dca_months in BTC investiert.
    # KREDITSCHOCK: einmaliger Wertverlust des Kreditbuchs (Immobilienabschwung).
    # Anders als der laufende Ausfall-Tropfen frisst er in die KAPITALSUBSTANZ:
    # debt_accrued wird negativ -> der Buchwert fällt UNTER die Kostenbasis.
    # Folge (gewollt): solange debt_accrued < 0, gibt es NICHTS zu ernten — der
    # DCA pausiert, bis der Ertrag den Verlust wieder aufgeholt hat.
    shock_pct = float(params.get("credit_shock_pct", 0.0))
    shock_date = params.get("credit_shock_date", None)
    shock_date = pd.Timestamp(shock_date).normalize() if shock_date else None
    shock_amount = 0.0
    harvest_freq = params.get("harvest_freq", "A")     # "A" | "S" | "Q"
    harvest_month = int(params.get("harvest_month", 1))  # Erntemonat bei "A"
    dca_months = int(params.get("dca_months", 12))     # Streckung der Ernte
    dca_queue = []          # [{"remaining": n, "monthly": chf}]
    n_redemptions = 0       # Anzahl Rücknahme-Vorgänge (Kostentransparenz)
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
        b_val = (btc_u_init + btc_u_dca + btc_u_subs) * px

        harvest = 0.0
        buys = 0.0
        sells = 0.0
        subs = 0.0
        outflow = 0.0
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
        # WICHTIG: debt_val nach der Zuschreibung neu berechnen — sonst wird der
        # Ertragsanfall des laufenden Tages nicht im NAV geführt (Off-by-one, der
        # die Abstimmung der Zerlegung um genau einen Tagesertrag verfehlt).
        debt_val = debt_cost_basis + debt_accrued

        # ---- Kreditschock (einmalig) ---------------------------------------
        if shock_pct > 0 and shock_date is not None and d == shock_date and debt_val > 0:
            _sh = debt_val * shock_pct
            debt_accrued -= _sh            # frisst Ertrag, dann Substanz
            loss_total += _sh
            shock_amount = _sh
            debt_val = debt_cost_basis + debt_accrued

        # ---- Cash-Verzinsung (real) + Drag-Memo ----------------------------
        if rf_cash > 0 and cash > 0:
            _int = cash * (rf_cash / 365.0)
            cash += _int
            cash_interest_total += _int
        cash_drag += (cash + pending_dca) * (max(rf_opp - rf_cash, 0.0) / 365.0)

        # ---- Neue Zeichnungen: wachsen den KERN, nie den Satelliten ---------
        if sub_amt > 0 and (
                (sub_freq == "M" and is_me) or (sub_freq == "Q" and is_qe)
                or (sub_freq == "A" and is_ye)):
            subs = sub_amt
            subs_total += subs
            _nav = debt_val + b_val + cash + pending_dca
            if sub_mode == "debt" or _nav <= 0:
                # Reine Yield-Bridge-Lesart: alles ins Kreditbuch. ACHTUNG:
                # verwässert die BTC-Quote der Bestandsinvestoren.
                debt_cost_basis += subs
            else:
                if sub_mode == "target":
                    w_btc = float(params["initial_btc_pct"])
                    w_cash = float(params.get("initial_cash_pct", 0.0))
                else:   # prorata — der faire Fondsstandard
                    w_btc = b_val / _nav
                    w_cash = (cash + pending_dca) / _nav
                w_btc = min(max(w_btc, 0.0), 1.0)
                w_cash = min(max(w_cash, 0.0), 1.0 - w_btc)
                _to_btc = subs * w_btc
                _to_cash = subs * w_cash
                _to_debt = subs - _to_btc - _to_cash
                debt_cost_basis += _to_debt
                cash += _to_cash
                if _to_btc > 0 and px > 0:
                    btc_u_subs += (_to_btc * (1 - tx)) / px
                    btc_subs_invested += _to_btc
                    b_val = (btc_u_init + btc_u_dca + btc_u_subs) * px
            debt_val = debt_cost_basis + debt_accrued

        # ---- ERTRAGS-ERNTE (periodisch, Default jährlich) -------------------
        #      EIN Rücknahme-Vorgang realisiert den seit der letzten Ernte
        #      aufgelaufenen Zuwachs — nie das Kapital. Das Geld verlässt das
        #      Zertifikat und liegt ab sofort als `pending_dca` im Produkt
        #      (Teil des NAV!). Von dort wird es gestreckt investiert.
        is_harvest = (
            (harvest_freq == "A" and is_me and d.month == harvest_month)
            or (harvest_freq == "S" and is_me
                and d.month in (harvest_month, (harvest_month + 5) % 12 + 1))
            or (harvest_freq == "Q" and is_qe))
        if is_harvest and debt_accrued > 0:
            target = debt_accrued
            harvest = target * redeem_r      # best-effort: Rest bleibt drin
            debt_accrued -= harvest          # und verzinst sich weiter
            harvest_total += harvest
            n_redemptions += 1
            pending_dca += harvest           # ab jetzt Cash im Produkt
            if harvest > 0 and dca_months > 0:
                dca_queue.append({"remaining": dca_months,
                                  "monthly": harvest / dca_months})
            debt_val = debt_cost_basis + debt_accrued

        # ---- Rollierender DCA: monatlich eine Tranche durch die Bandlogik ---
        if is_me and dca_queue and pending_dca > 0:
            tranche = min(sum(e["monthly"] for e in dca_queue if e["remaining"] > 0),
                          pending_dca)
            for e in dca_queue:
                if e["remaining"] > 0:
                    e["remaining"] -= 1
            dca_queue = [e for e in dca_queue if e["remaining"] > 0]
            if tranche > 0:
                _tranche0 = tranche          # WICHTIG: Originalgrösse merken —
                                             # pending_dca muss um DIESE sinken.
                tot = debt_val + b_val + cash + pending_dca
                # CASH-AUFFÜLLUNG ZUERST: Der Puffer zahlt die Gebühren. Er wird
                # auf die Zielquote gebracht, BEVOR alloziert wird — sonst muss die
                # nächste Management Fee gestundet werden (fee_debt). Der Topup
                # minimiert die Stundung; ein BTC-Verkauf für die laufende Fee
                # findet nicht statt.
                _cash_target = cash_floor_pct * tot
                _topup = min(max(_cash_target - cash, 0.0), tranche)
                cash += _topup
                tranche -= _topup
                cash_topup_total += _topup
                w = (b_val / tot) if tot > 0 else 0.0
                rate = boost_rate if w < lo else (0.0 if w > up else base_rate)
                invest = tranche * rate
                if invest > 0 and px > 0:
                    btc_u_dca += (invest * (1 - tx)) / px
                    btc_dca_invested += invest
                    buys = invest
                cash += tranche - invest     # Rest der Tranche bleibt Cash
                pending_dca -= _tranche0     # Auffüllung + Rest + Invest
                b_val = (btc_u_init + btc_u_dca + btc_u_subs) * px

        # ---- Sell-Regel: BTC über Cap -> zurück auf exakt Cap --------------
        is_sell = (is_me if c_freq == "M" else is_qe)
        if sell_on and is_sell and px > 0:
            tot = debt_val + b_val + cash
            if tot > 0 and (b_val / tot) > up:
                target_b = up * (debt_val + cash) / (1 - up) if up < 1 else b_val
                excess = max(b_val - target_b, 0.0)
                if excess > 0:
                    _u = btc_u_init + btc_u_dca + btc_u_subs
                    _fi = (btc_u_init / _u) if _u > 0 else 0.0
                    _fd = (btc_u_dca / _u) if _u > 0 else 0.0
                    _fs = 1.0 - _fi - _fd
                    _sold_u = excess / px
                    btc_u_init -= _sold_u * _fi
                    btc_u_dca -= _sold_u * _fd
                    btc_u_subs -= _sold_u * _fs
                    _net = excess * (1 - tx)
                    btc_init_realized += _net * _fi
                    btc_dca_realized += _net * _fd
                    btc_subs_realized += _net * _fs
                    cash += _net
                    sells = excess
                    b_val = (btc_u_init + btc_u_dca + btc_u_subs) * px

        # ---- INVESTOREN-RÜCKNAHMEN (Liquiditäts-Wasserfall mit Gate) --------
        is_out = ((out_freq == "M" and is_me) or (out_freq == "Q" and is_qe)
                  or (out_freq == "A" and is_ye))
        if is_out and out_pct > 0:
            _nav = debt_val + b_val + cash + pending_dca
            _want = _nav * out_pct
            _paid = 0.0

            # (1) Cash — aber der Gebührenpuffer bleibt geschützt
            _cash_keep = cash_floor_pct * max(_nav - _want, 0.0)
            _from_cash = min(max(cash - _cash_keep, 0.0), _want)
            cash -= _from_cash
            _paid += _from_cash

            # (2) BTC — NUR bis zur unteren Bandgrenze. Nicht darüber hinaus:
            #     sonst plündert der Aussteiger den liquiden Sleeve.
            _rest = _want - _paid
            if _rest > 1e-9 and b_val > 0 and px > 0:
                _nav_after = _nav - _want
                _btc_min = lo * max(_nav_after, 0.0)   # Bandgrenze halten
                _sellable = max(b_val - _btc_min, 0.0)
                _gross = min(_sellable, _rest / (1 - tx))
                if _gross > 0:
                    _u = btc_u_init + btc_u_dca + btc_u_subs
                    _fi = (btc_u_init / _u) if _u > 0 else 0.0
                    _fd = (btc_u_dca / _u) if _u > 0 else 0.0
                    _fs = 1.0 - _fi - _fd
                    _su = _gross / px
                    btc_u_init -= _su * _fi
                    btc_u_dca -= _su * _fd
                    btc_u_subs -= _su * _fs
                    _pr = _gross * (1 - tx)
                    btc_init_realized += _pr * _fi
                    btc_dca_realized += _pr * _fd
                    btc_subs_realized += _pr * _fs
                    _paid += _pr
                    b_val = (btc_u_init + btc_u_dca + btc_u_subs) * px

            # (3) Kreditbuch — best effort. EIGENER Parameter: eine Kapital-
            #     rücknahme ist ein grosses Ticket und deutlich schwerer zu
            #     platzieren als die kleine jährliche Ertrags-Ernte.
            _rest = _want - _paid
            if _rest > 1e-9 and debt_val > 0:
                _fill = min(_rest * principal_redeem_r, _rest)
                if _fill > 0:
                    _fb = _fill * (debt_cost_basis / debt_val) if debt_val > 0 else 0.0
                    _fa = _fill - _fb
                    debt_cost_basis -= _fb
                    debt_accrued -= _fa
                    debt_val = debt_cost_basis + debt_accrued
                    _paid += _fill

            # (4) GATE: was nicht liquidiert werden konnte, wird ausgesetzt
            _gap = _want - _paid
            if _gap > 1e-6:
                outflow_gated += _gap
                n_gates += 1
            outflow_paid += _paid
            outflow = _paid

        # ---- Gebühren-Wasserfall ------------------------------------------
        #   Management Fee: Cash -> STUNDEN (fee_debt). Nie BTC, nie Kreditbuch.
        #   Performance Fee: Cash -> BTC (feuert nur über HWM, also in guten
        #     Phasen mit reichlich Cash). Kreditbuch NIE.
        def _pay(amount, allow_btc=True):
            nonlocal cash, btc_u_init, btc_u_dca, btc_u_subs, b_val, fee_paid
            nonlocal fee_btc_sold, fee_from_cash, fee_from_btc
            nonlocal btc_init_realized, btc_dca_realized, btc_subs_realized
            if amount <= 1e-9:
                return 0.0
            from_cash = min(cash, amount)
            cash -= from_cash
            fee_paid += from_cash
            fee_from_cash += from_cash
            short = amount - from_cash
            if allow_btc and short > 1e-9 and b_val > 0 and px > 0:
                gross = min(b_val, short / (1 - tx))
                _u = btc_u_init + btc_u_dca + btc_u_subs
                _fi = (btc_u_init / _u) if _u > 0 else 0.0
                _fd = (btc_u_dca / _u) if _u > 0 else 0.0
                _fs = 1.0 - _fi - _fd
                _su = gross / px
                btc_u_init -= _su * _fi
                btc_u_dca -= _su * _fd
                btc_u_subs -= _su * _fs
                proceeds = gross * (1 - tx)
                btc_init_realized += proceeds * _fi
                btc_dca_realized += proceeds * _fd
                btc_subs_realized += proceeds * _fs
                fee_btc_sold += gross
                fee_paid += proceeds
                fee_from_btc += proceeds
                short -= proceeds
                b_val = (btc_u_init + btc_u_dca + btc_u_subs) * px
            return max(short, 0.0)

        fee_due = ((is_me and mgmt_freq == "M") or (is_qe and mgmt_freq == "Q"))
        if fee_due and mgmt_fee > 0:
            per_year = 12.0 if mgmt_freq == "M" else 4.0
            aum = debt_val + b_val + cash + pending_dca
            amt = aum * (mgmt_fee / per_year) + fee_debt
            fee_debt = 0.0
            _before = fee_paid
            # Management Fee: Cash zuerst, Rest wird GESTUNDET (fee_debt) — NIE
            # BTC verkaufen. Die Fee ist beim AMC eine Level-Deduktion; sie grenzt
            # sich gegen die NAV ab (siehe total_value) und wird in Cash beglichen,
            # sobald die Ernte wieder Liquidität liefert. Marktstandard (Fee-Deferral),
            # verhindert prozyklischen BTC-Verkauf im Stressfall.
            fee_debt = _pay(amt, allow_btc=False)
            mgmt_paid_row = fee_paid - _before
            total_mgmt += mgmt_paid_row

        if (is_me and d.month in cryst_months) or i == len(idx) - 1:
            if perf_fee > 0:
                nav_now = debt_val + b_val + cash + pending_dca - fee_debt
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
                _after = debt_val + b_val + cash + pending_dca - fee_debt
                if _after > hwm:
                    hwm = _after
                prev_cryst_date = d

        rows.append({
            "date": d,
            "debt_value": debt_val,
            "btc_value": b_val,
            "cash": cash + pending_dca,
            "total_value": debt_val + b_val + cash + pending_dca - fee_debt,
            "harvest": harvest if harvest > 0 else np.nan,
            "debt_cost_basis": debt_cost_basis,
            "debt_accrued": debt_accrued,
            "subscriptions": subs,
            "outflow": outflow,
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
    _bs_end = btc_u_subs * _px_end
    btc_init_gain = (_bi_end + btc_init_realized) - btc_init_invested
    btc_dca_gain = (_bd_end + btc_dca_realized) - btc_dca_invested
    btc_subs_gain = (_bs_end + btc_subs_realized) - btc_subs_invested
    # DCA-Anteil misst NUR den ertragsfinanzierten Beitrag gegen die
    # kapitalfinanzierten Lots (Startallokation + Zeichnungen). Sonst würden
    # Mittelzuflüsse die Kennzahl verfälschen.
    _btot = btc_init_gain + btc_dca_gain + btc_subs_gain
    dca_share = (btc_dca_gain / _btot) if abs(_btot) > 1e-9 else float("nan")
    # ZWEITE KENNZAHL: der Mechanismus-Anteil OHNE Zeichnungseffekt. Ein schnell
    # wachsender Fonds kauft pro-rata viel BTC aus Zeichnungen — das drückt den
    # rohen DCA-Anteil, sagt aber NICHTS über die Kraft des Ertragsmechanismus.
    _bmech = btc_init_gain + btc_dca_gain
    dca_share_ex_subs = ((btc_dca_gain / _bmech) if abs(_bmech) > 1e-9
                         else float("nan"))

    nav_end = float(out["total_value"].iloc[-1])
    # Zeichnungen sind kein Gewinn, Rücknahmen kein Verlust.
    pnl = nav_end - cap - subs_total + outflow_paid
    # Die gestundete Management Fee (fee_debt) ist eine ökonomisch bereits
    # getragene Gebühr: sie mindert die NAV (siehe total_value), auch wenn sie
    # noch nicht in Cash beglichen ist. Sie gehört daher in den Fee-Term der
    # Abstimmung, sonst bricht die Identität um genau fee_debt.
    recon = (accrued_total - loss_total + btc_init_gain + btc_dca_gain
             + btc_subs_gain + cash_interest_total
             - (total_mgmt + total_perf + fee_debt))

    out.attrs["total_mgmt"] = total_mgmt
    out.attrs["total_perf"] = total_perf
    out.attrs["fee_debt_final"] = fee_debt
    out.attrs["attribution"] = {
        "debt_income": accrued_total,          # Ertragsanfall im Kreditbuch (brutto)
        "credit_losses": -loss_total,          # Kreditausfälle (negativ)
        "btc_initial_gain": btc_init_gain,
        "btc_dca_gain": btc_dca_gain,
        "btc_subs_gain": btc_subs_gain,
        "btc_subs_invested": btc_subs_invested,
        "cash_interest": cash_interest_total,
        "fees": -(total_mgmt + total_perf + fee_debt),
        "total_pnl": pnl,
        "reconciliation_error": recon - pnl,
        "dca_share": dca_share,
        "dca_share_ex_subs": dca_share_ex_subs,
        "btc_initial_invested": btc_init_invested,
        "btc_dca_invested": btc_dca_invested,
        "harvest_total": harvest_total,        # tatsächlich realisierter "Coupon"
        "n_redemptions": n_redemptions,        # Anzahl Rücknahme-Vorgänge
        "credit_shock": shock_amount,          # einmaliger Schock (CHF)
        # KAPITALERHALTUNGS-NACHWEIS: NAV, wenn Bitcoin auf NULL ginge
        "nav_if_btc_zero": float(out["debt_value"].iloc[-1] + out["cash"].iloc[-1]) - fee_debt,
        "dca_pending": pending_dca,
        "accrued_total": accrued_total,
        "unharvested": debt_accrued,           # klemmt im Zertifikat
        "harvest_ratio": (harvest_total / accrued_total) if accrued_total > 0 else float("nan"),
        "subscriptions": subs_total,
        "outflow_paid": outflow_paid,
        "outflow_gated": outflow_gated,
        "n_gates": n_gates,
        "cash_topup": cash_topup_total,
        "cash_drag": -cash_drag,
        "years": max((idx[-1] - idx[0]).days / 365.25, 1e-9),
    }
    return out


def run_debt_only(idx, params):
    """Benchmark: identisches Kreditbuch OHNE Bitcoin.

    KORREKTUR (Systemaudit): Der Benchmark muss DIESELBEN Belastungen tragen wie
    die Strategie — sonst isoliert der Vergleich nicht den Bitcoin-Beitrag,
    sondern vermischt ihn mit Gebühren und Kreditschock. Enthalten sind daher:
      * Nettorendite (thesaurierend, keine Ernte, kein Verkauf)
      * laufende Kreditausfälle
      * EINMALIGER KREDITSCHOCK (neu)
      * MANAGEMENT-GEBÜHR (neu) — sie wird dem Kreditbuch belastet
      * Zeichnungen
    NICHT enthalten: Performance Fee (fällt ohne Bitcoin-Überrendite faktisch
    nicht an) und der Cash-Puffer (der existiert nur wegen des BTC-Mechanismus —
    seine Renditebremse ist zu Recht ein Nachteil der Strategie).
    """
    idx = pd.DatetimeIndex(idx).normalize().unique().sort_values()
    cap = float(params["initial_capital"])
    ny = float(params["net_yield"])
    loss_r = float(params.get("credit_loss_rate", 0.0))
    mgmt = float(params.get("mgmt_fee", 0.0))
    mgmt_freq = params.get("mgmt_fee_freq", "M")
    sub_amt = float(params.get("subscription_amount", 0.0))
    sub_freq = params.get("subscription_freq", "N")
    shock_pct = float(params.get("credit_shock_pct", 0.0))
    shock_date = params.get("credit_shock_date", None)
    shock_date = pd.Timestamp(shock_date).normalize() if shock_date else None

    v = cap
    vals = []
    for i, d in enumerate(idx):
        v += v * ((ny - loss_r) / 365.0)
        is_me = (i == len(idx) - 1) or (idx[i + 1].to_period("M") != d.to_period("M"))
        is_qe = is_me and d.month in (3, 6, 9, 12)
        is_ye = is_me and d.month == 12

        if shock_pct > 0 and shock_date is not None and d == shock_date:
            v -= v * shock_pct                      # gleicher Schock wie die Strategie

        if mgmt > 0 and ((is_me and mgmt_freq == "M") or (is_qe and mgmt_freq == "Q")):
            v -= v * (mgmt / (12.0 if mgmt_freq == "M" else 4.0))

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
    f"struktureller Bitcoin-Allokation. Der thesaurierende Ertrag wird einmal "
    f"jährlich (Default Januar) geerntet und dann rollierend über Bandregeln in "
    f"Bitcoin investiert — das eingesetzte "
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
                                 help="Polster zur Gebührendeckung, gefüllt aus der "
                                      "Ernte. Reicht es nicht, wird die Management "
                                      "Fee GESTUNDET (gegen die NAV abgegrenzt, "
                                      "später in Cash beglichen) — kein BTC- oder "
                                      "Kreditbuch-Verkauf für die laufende Gebühr.") / 100.0
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

    st.markdown("### Ertrags-Ernte")
    st.caption("Das Underlying schüttet NICHT aus. Der Ertrag wird periodisch "
               "durch Rücknahme realisiert und danach gestreckt investiert.")
    _hf_label = st.selectbox(
        "Ernte-Rhythmus", ["jährlich", "halbjährlich", "quartalsweise"], index=0,
        help="Jährlich = ein einziger Rücknahme-Vorgang pro Jahr. Häufigere "
             "Rücknahmen erhöhen den operativen Aufwand, ohne dass mehr Ertrag "
             "entsteht.")
    harvest_freq = {"jährlich": "A", "halbjährlich": "S",
                    "quartalsweise": "Q"}[_hf_label]
    harvest_month = st.selectbox(
        "Erntemonat (bei jährlich)",
        list(range(1, 13)), index=0,
        format_func=lambda mth: ["Januar", "Februar", "März", "April", "Mai",
                                 "Juni", "Juli", "August", "September", "Oktober",
                                 "November", "Dezember"][mth - 1],
        help="Es wird der seit der letzten Ernte aufgelaufene Ertrag realisiert "
             "— also im Januar der Ertrag des Vorjahres.")
    dca_months = st.slider(
        "DCA-Streckung der Ernte (Monate)", 1, 24, 12, 1,
        help="Der geerntete Jahresertrag wird über diese Anzahl Monate "
             "gleichmässig in Bitcoin investiert (rollierender DCA) — nicht als "
             "Klumpen.")

    st.markdown("### Rücknahmen (Investoren)")
    st.caption("Open-End: Investoren können aussteigen. Wasserfall: Cash → BTC "
               "(nur bis zur unteren Bandgrenze) → Kreditbuch → Gate.")
    _of_label = st.selectbox("Rücknahme-Rhythmus",
                             ["keine", "quartalsweise", "jährlich"], index=0)
    outflow_freq = {"keine": "N", "quartalsweise": "Q", "jährlich": "A"}[_of_label]
    outflow_pct = st.slider("Rücknahme je Termin (% des NAV)", 0, 30, 0, 1,
                            help="Wie viel Prozent des Fondsvermögens je Termin "
                                 "zurückgenommen wird.") / 100.0
    principal_redeem_rate = st.slider(
        "Kapitalrücknahme beim Emittenten (% erfüllbar)", 0, 100, 50, 5,
        help="Anteil einer KAPITAL-Rücknahme, der beim Underlying tatsächlich "
             "platziert werden kann. Grosse Tickets sind deutlich schwerer als "
             "die kleine jährliche Ertrags-Ernte. Was nicht liquidiert werden "
             "kann, wird GEGATED — das schützt die bleibenden Investoren.") / 100.0
    cash_floor_pct = st.slider(
        "Cash-Zielquote (% des NAV)", 0, 20, 5, 1,
        help="Der Puffer zahlt die Gebühren. Er wird aus der Ernte "
             "aufgefüllt, BEVOR in Bitcoin alloziert wird.") / 100.0

    st.markdown("### Kreditschock (Stress)")
    st.caption("Einmaliger Wertverlust des Kreditbuchs — z.B. Schweizer "
               "Immobilienabschwung. Nachrangige Hypotheken tragen den Erstverlust.")
    credit_shock_pct = st.select_slider(
        "Schockgrösse", options=[0, 10, 20, 30, 40], value=0,
        format_func=lambda v: "kein Schock" if v == 0 else f"−{v}%") / 100.0
    credit_shock_date = None
    if credit_shock_pct > 0:
        credit_shock_date = st.date_input(
            "Zeitpunkt des Schocks", value=date(2022, 6, 1), key="pd_shock_dt")

    st.markdown("### Zeichnungen (Wachstum)")
    subscription_freq_label = st.selectbox(
        "Rhythmus", ["keine", "monatlich", "quartalsweise", "jährlich"], index=0)
    subscription_freq = {"keine": "N", "monatlich": "M",
                         "quartalsweise": "Q", "jährlich": "A"}[subscription_freq_label]
    subscription_amount = st.number_input(
        "Betrag je Zeichnung (CHF)", min_value=0, value=0, step=100_000)
    _sm_label = st.selectbox(
        "Zeichnungs-Allokation",
        ["Pro-rata (aktuelle Gewichte)", "100% ins Kreditbuch",
         "Zielgewichte (Startallokation)"], index=0,
        help="OPEN-END: Neue Gelder zeichnen zum NAV und kaufen einen Anteil am "
             "BESTEHENDEN Portfolio. Pro-rata hält die Gewichte konstant und ist "
             "der faire Fondsstandard. «100% ins Kreditbuch» verwässert die "
             "BTC-Quote der Bestandsinvestoren mit jeder Zeichnung.")
    subscription_mode = {"Pro-rata (aktuelle Gewichte)": "prorata",
                         "100% ins Kreditbuch": "debt",
                         "Zielgewichte (Startallokation)": "target"}[_sm_label]

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
    harvest_freq=harvest_freq, harvest_month=int(harvest_month),
    dca_months=int(dca_months),
    credit_shock_pct=credit_shock_pct, credit_shock_date=credit_shock_date,
    subscription_mode=subscription_mode, cash_floor_pct=cash_floor_pct,
    outflow_pct=outflow_pct, outflow_freq=outflow_freq,
    principal_redeem_rate=principal_redeem_rate,
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
c6.metric("Max Drawdown*", f"{m['max_drawdown']*100:.2f}%")
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

# Fee-Funding-Transparenz: die Management Fee wird aus dem Cash-Puffer (Ernte)
# bezahlt; reicht er nicht, wird sie GESTUNDET (fee_debt) statt BTC zu verkaufen.
_fee_deferred = float(ts.attrs.get("fee_debt_final", 0.0))
_fee_from_btc_total = float(ts["fee_from_btc"].sum()) if "fee_from_btc" in ts else 0.0
if _fee_deferred > 1e-6 or _fee_from_btc_total > 1e-6:
    st.caption(
        f"**Fee-Funding:** aktuell **{fmt_chf(_fee_deferred)} Management Fee gestundet** "
        "— aufgelaufen, mindert die NAV bereits, wird aber erst in Cash beglichen, "
        "sobald die Ernte wieder Liquidität liefert (kein BTC-Verkauf für die laufende "
        f"Gebühr). Über BTC-Verkauf finanziert: {fmt_chf(_fee_from_btc_total)} "
        "(nur Performance Fee bzw. Investoren-Rücknahmen).")
else:
    st.caption(
        "**Fee-Funding:** Gebühren vollständig aus dem Cash-Puffer (aus der Ernte) "
        "bezahlt — kein BTC-Verkauf, keine Stundung. Die Management Fee würde bei "
        "Liquiditätsknappheit gestundet, nicht durch BTC-Verkauf gedeckt.")

# ==========================================================================
# ERTRAGS-ERNTE & REDEMPTION-STRESS — die produktspezifische Kernsektion
# ==========================================================================
st.markdown("## Ertrags-Ernte & Redemption-Stress")
st.markdown(
    "<p style='color:#A9B5A4;margin-top:-6px'>Das Underlying ist "
    "<strong>thesaurierend</strong> — es schüttet nichts aus. Der Ertrag wird "
    "einmal jährlich (Default Januar) geerntet, indem exakt der seit der letzten "
    "Ernte aufgelaufene NAV-Zuwachs über der Kostenbasis redimiert und dann "
    "rollierend über mehrere Monate in Bitcoin gestreckt wird. Das eingesetzte "
    "Kapital wird nie angetastet. "
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

h5, h6, h7, h8 = st.columns(4)
with h5:
    st.metric("Rücknahme-Vorgänge", f"{_att['n_redemptions']}")
    st.caption(f"{_hf_label} · über {years:.1f} Jahre")
with h6:
    st.metric("Ø Ernte je Vorgang",
              fmt_chf(_att["harvest_total"] / max(_att["n_redemptions"], 1)))
    st.caption("Ticketgrösse der Rücknahme")
with h7:
    st.metric("DCA noch ausstehend", fmt_chf(_att["dca_pending"]))
    st.caption(f"geerntet, wird über {dca_months} Mte investiert")
with h8:
    st.metric("DCA-Streckung", f"{dca_months} Monate")
    st.caption("rollierend, kein Klumpen")

if redemption_rate < 1.0:
    st.warning(
        f"⚠️ **Redemption-Erfolgsquote {redemption_rate*100:.0f}%** — es werden "
        f"nur {redemption_rate*100:.0f}% des aufgelaufenen Ertragszuwachses "
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
# KAPITALERHALTUNG — die Kennzahl, die dieses Produkt verkauft
# ==========================================================================
st.markdown("## Kapitalerhaltung — was bleibt, wenn Bitcoin auf null geht?")
st.markdown(
    "<p style='color:#A9B5A4;margin-top:-6px'>Das eingesetzte Kapital liegt im "
    "besicherten Kreditbuch. Bitcoin wird <strong>ausschliesslich aus dem "
    "geernteten Ertrag</strong> gekauft. Der Investor kann sein Kapital daher "
    "strukturell nicht an Bitcoin verlieren — nur den Ertrag, den es bereits "
    "erwirtschaftet hat.</p>", unsafe_allow_html=True)

_nav_zero = ts["debt_value"] + ts["cash"]          # NAV, wenn BTC = 0
_zero_end = float(_nav_zero.iloc[-1])
_zero_ratio = _zero_end / _cap_base if _cap_base > 0 else 0.0
# Der ungünstigste Zeitpunkt: wann war die Deckung am tiefsten?
_cov = _nav_zero / (initial_capital + ts["subscriptions"].cumsum())
_cov_min = float(_cov.min())
_cov_min_at = _cov.idxmin()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("NAV bei BTC = 0 (heute)", fmt_chf(_zero_end))
    st.caption("Kreditbuch + Cash, ohne jeden Bitcoin")
with k2:
    st.metric("Kapitaldeckung", f"{_zero_ratio*100:.1f}%")
    st.caption("in % des eingesetzten Kapitals")
with k3:
    st.metric("Tiefste Deckung je", f"{_cov_min*100:.1f}%")
    st.caption(f"am {_cov_min_at:%b %Y}")
with k4:
    st.metric("Kreditschock", "keiner" if credit_shock_pct <= 0
              else f"−{credit_shock_pct*100:.0f}%")
    st.caption(fmt_chf(_att["credit_shock"]) if credit_shock_pct > 0
               else "kein Stress modelliert")

if _zero_ratio >= 1.0:
    st.success(
        f"✅ **Selbst bei einem Totalverlust von Bitcoin läge der NAV bei "
        f"{fmt_chf(_zero_end)} — das sind {_zero_ratio*100:.1f}% des "
        f"eingesetzten Kapitals.** Das Kapital ist strukturell nicht dem "
        "Bitcoin-Risiko ausgesetzt: gekauft wird nur aus dem Ertrag, den das "
        "Kreditbuch bereits abgeworfen hat.")
else:
    st.warning(
        f"⚠️ Bei einem Totalverlust von Bitcoin läge der NAV bei "
        f"{_zero_ratio*100:.1f}% des eingesetzten Kapitals. Die Unterdeckung "
        "stammt NICHT aus Bitcoin, sondern aus Kreditausfällen bzw. dem "
        "Kreditschock sowie aus Gebühren, die den Ertrag übersteigen.")

fig_k = go.Figure()
fig_k.add_trace(go.Scatter(x=ts.index, y=net.values, name="NAV (mit Bitcoin)",
                           line=dict(color=OAK_GOLD, width=2.2)))
fig_k.add_trace(go.Scatter(x=ts.index, y=_nav_zero.values,
                           name="NAV, wenn Bitcoin auf null ginge",
                           line=dict(color=OAK_SAGE, width=2, dash="dash")))
fig_k.add_trace(go.Scatter(
    x=ts.index, y=(initial_capital + ts["subscriptions"].cumsum()).values,
    name="Eingesetztes Kapital", line=dict(color=OAK_CREAM_DIM, width=1.4,
                                           dash="dot")))
fig_k = style_plotly(fig_k, height=380)
fig_k.update_yaxes(title_text="Wert (CHF)", tickformat=",.0f")
fig_k.update_layout(margin=dict(l=70, r=30, t=10, b=60),
                    legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))
st.markdown("##### Der Sicherheitsboden: die gestrichelte Linie darf das "
            "eingesetzte Kapital nie unterschreiten")
st.plotly_chart(fig_k, use_container_width=True)

# ==========================================================================
# LIQUIDITÄT & RÜCKNAHMEN
# ==========================================================================
if outflow_pct > 0 or _att["outflow_paid"] > 0:
    st.markdown("## Liquidität & Rücknahmen")
    st.markdown(
        "<p style='color:#A9B5A4;margin-top:-6px'>Bitcoin ist der liquide "
        "Sleeve — aber er wird <strong>nicht geplündert</strong>. Verkauft wird "
        "nur bis zur unteren Bandgrenze; darüber hinaus muss das Kreditbuch "
        "liefern. Kann es das nicht, wird die Auszahlung anteilig "
        "<strong>gegated</strong>. Das verhindert, dass Aussteiger die "
        "Liquidität abräumen und die Bleibenden mit dem illiquiden Rest "
        "zurücklassen.</p>", unsafe_allow_html=True)

    l1, l2, l3, l4 = st.columns(4)
    with l1:
        st.metric("Ausgezahlt", fmt_chf(_att["outflow_paid"]))
        st.caption("tatsächlich bedient")
    with l2:
        st.metric("Gegated", fmt_chf(_att["outflow_gated"]))
        st.caption(f"{_att['n_gates']} Gate-Ereignisse")
    with l3:
        _req = _att["outflow_paid"] + _att["outflow_gated"]
        _fill = (_att["outflow_paid"] / _req * 100) if _req > 0 else 100.0
        st.metric("Erfüllungsquote", f"{_fill:.0f}%")
        st.caption("bedient / beantragt")
    with l4:
        st.metric("Cash-Auffüllung", fmt_chf(_att["cash_topup"]))
        st.caption("aus der Ernte, vor der Allokation")

    if _att["outflow_gated"] > 0:
        st.warning(
            f"⚠️ **{fmt_chf(_att['outflow_gated'])} konnten nicht ausgezahlt "
            f"werden** ({_att['n_gates']} Gate-Ereignisse). Das ist kein Defekt, "
            "sondern die Schutzfunktion: Ohne Gate hätten diese Auszahlungen den "
            "Bitcoin-Sleeve unter die Bandgrenze gedrückt und den Gebührenpuffer "
            "geleert — die bleibenden Investoren hätten einen übergewichteten, "
            "illiquiden Kreditbuch-Rest gehalten, den nur die (unzuverlässige) "
            "Ernte hätte reparieren können.")
    else:
        st.success("✅ Alle Rücknahmen konnten bedient werden, ohne die "
                   "Bandstruktur oder den Gebührenpuffer zu verletzen.")

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
    ("Bitcoin — zeichnungsfinanziert (pro-rata)", _att["btc_subs_gain"]),
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
if _att["btc_subs_invested"] > 0:
    _dse = _att["dca_share_ex_subs"]
    e1, e2 = st.columns(2)
    with e1:
        st.metric("Mechanismus-Anteil (ohne Zeichnungseffekt)",
                  "n/a" if _dse != _dse else f"{_dse*100:.1f}%")
        st.caption("DCA vs. Startallokation — misst die Kraft der Yield Bridge")
    with e2:
        st.metric("BTC aus Zeichnungen", fmt_chf(_att["btc_subs_invested"]))
        st.caption("pro-rata, kein Kernkapital umgeschichtet")
    st.caption(
        f"Zusätzlich {fmt_chf(_att['btc_subs_invested'])} Bitcoin aus "
        f"Zeichnungen (pro-rata zum bestehenden Portfolio). Das ist KEINE "
        "Umschichtung von Kernkapital: neue Investoren kaufen zum NAV einen "
        "Anteil am bestehenden Portfolio. Der DCA-Anteil misst deshalb nur den "
        "ertragsfinanzierten Beitrag gegen die kapitalfinanzierten Lots.")

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
       ("Max Drawdown*", f"{m['max_drawdown']*100:.2f}%"),
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

with st.expander("Ertrags-Ernte (jährliche Realisierung)"):
    _hv = ts["harvest"].dropna()
    st.bar_chart(_hv)
    st.caption(f"Summe geerntet: {fmt_chf(_hv.sum())} · "
               f"Ø je Ernte: {fmt_chf(_hv.mean())}")

# ==========================================================================
# KALIBRIERUNG — Grundallokation, Cash-Sleeve, Bandbreite (vor Emission).
# Multi-Objective-Grid-Search mit harten Constraints statt Wert-Picking:
#   1. Mechanismus-Constraint: Median DCA-Anteil ex-subs über einer Mindest-
#      schwelle (Default 60%) — sichert, dass der Mechanismus tatsächlich das
#      tut, was der Produktname verspricht.
#   2. Downside-Constraint: im unteren Viertel der Fenster (P25, gepaart pro
#      Fenster) darf die Strategie nicht schlechter sein als „nur Kreditbuch"
#      (Downside-Toleranz einstellbar) — die Versicherung darf im schlechten
#      Fall nicht schlechter sein als gar keine Versicherung zu haben.
# Unter den so gefundenen zulässigen Kombinationen wird die Median-Netto-CAGR
# maximiert. Bewusst KEIN gewichteter Score (versteckt den Trade-off) —
# stattdessen Constraints zuerst, dann Optimierung im zulässigen Raum.
# ==========================================================================
st.markdown("---")
st.markdown("## Kalibrierung — Grundallokation, Cash-Sleeve, Bandbreite")
st.markdown(
    "<p style='color:#A9B5A4;margin-top:-6px'>Rastersuche über Startallokation × "
    "Cash-Sleeve × Bandbreite, ausgewertet über alle rollierenden Fenster. Zwei "
    "harte Nebenbedingungen (kein Kompromiss-Score): Mechanismus-Reinheit "
    "(DCA-Anteil ex-subs) und Downside-Robustheit (Strategie ≥ nur Kreditbuch im "
    "unteren Viertel der Fenster). Unter den zulässigen Kombinationen wird die "
    "Median-Rendite maximiert.</p>", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_pd_calibration_grid(_btc, base, allocs, cash_pcts, widths, win_years, step_months):
    """Grid über (Startallokation × Cash-Sleeve × Bandbreite) × rollierende
    Fenster. Paart Strategie- und Benchmark-CAGR (identischer Fee-Schedule,
    run_debt_only trägt die Fee bereits intern) pro Fenster. Kreditschock wird
    neutralisiert (Einmal-Event gehört nicht in eine Fenster-Verteilung)."""
    full = pd.DatetimeIndex(_btc.index)
    if len(full) < 400:
        return pd.DataFrame()
    starts = pd.date_range(full[0], full[-1] - pd.DateOffset(years=win_years),
                           freq=f"{step_months}MS")
    rows = []
    for alloc in allocs:
        lo = max(alloc * 2.0, 0.05)
        for width in widths:
            up = lo + width
            for cash_pct in cash_pcts:
                p = dict(base)
                p.update(initial_btc_pct=alloc, initial_cash_pct=cash_pct,
                         lower_threshold=lo, upper_threshold=up,
                         credit_shock_pct=0.0, credit_shock_date=None)
                for s in starts:
                    e = s + pd.DateOffset(years=win_years)
                    w = full[(full >= s) & (full <= e)]
                    if len(w) < 300:
                        continue
                    t = run_pd_btc(_btc, w, p)
                    if t.empty:
                        continue
                    yrs = max((t.index[-1] - t.index[0]).days / 365.25, 1e-9)
                    cagr = (t["total_value"].iloc[-1] / p["initial_capital"]) ** (1 / yrs) - 1
                    a = t.attrs.get("attribution", {})
                    tb = run_debt_only(w, p)
                    bcagr = (tb.iloc[-1] / p["initial_capital"]) ** (1 / yrs) - 1
                    rows.append({"alloc": alloc, "cash_pct": cash_pct, "width": width,
                                 "start": s, "net_cagr": cagr, "bench_cagr": bcagr,
                                 "dca_share": a.get("dca_share_ex_subs", np.nan)})
    return pd.DataFrame(rows)


def score_calibration(grid, dca_floor, downside_tol):
    """Gruppiert nach Parameterkombination, wendet die zwei harten Constraints
    an, sortiert nach Median-CAGR. Kein gewichteter Score."""
    g = grid.copy()
    g["excess"] = g["net_cagr"] - g["bench_cagr"]
    gp = g.groupby(["alloc", "cash_pct", "width"])
    s = gp.agg(median_cagr=("net_cagr", "median"),
               p05_cagr=("net_cagr", lambda x: x.quantile(.05)),
               median_dca=("dca_share", "median"),
               p25_excess=("excess", lambda x: x.quantile(.25)),
               spread=("net_cagr", lambda x: x.max() - x.min()),
               n=("net_cagr", "size")).reset_index()
    s["feasible"] = (s["median_dca"] >= dca_floor) & (s["p25_excess"] >= downside_tol)
    return s.sort_values("median_cagr", ascending=False)


cc1, cc2, cc3 = st.columns(3)
with cc1:
    _pd_dca_floor = st.slider("Mindest-DCA-Anteil (Mechanismus-Constraint)",
                              30, 90, 60, 5, key="pd_calib_floor",
                              help="Median DCA-Anteil ex-subs muss über allen "
                                   "Fenstern mindestens diesen Wert erreichen.") / 100.0
with cc2:
    _pd_downside_tol = st.slider("Downside-Toleranz (pp, P25 Excess vs. Benchmark)",
                                 -3.0, 1.0, -1.0, 0.25, key="pd_calib_tol",
                                 help="Wie weit darf die Strategie im unteren "
                                      "Viertel der Fenster hinter «nur Kreditbuch» "
                                      "zurückbleiben? 0 = nie schlechter.") / 100.0
with cc3:
    st.caption("")
    _pd_run_calib = st.button("Kalibrierung starten", key="pd_calib_go")

if _pd_run_calib:
    st.session_state["pd_calib_has_run"] = True

if st.session_state.get("pd_calib_has_run"):
    _pd_c_allocs = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
    _pd_c_cash = [0.025, 0.05, 0.075]
    _pd_c_widths = [0.10, 0.15, 0.20]
    _pd_c_base = dict(params)
    with st.spinner("Rastersuche über alle Kombinationen × rollierende Fenster… "
                     "(kann 1–2 Minuten dauern)"):
        pd_calib_grid = compute_pd_calibration_grid(
            btc_chf, _pd_c_base, _pd_c_allocs, _pd_c_cash, _pd_c_widths,
            win_years=3, step_months=6)

    if pd_calib_grid.empty:
        st.warning("Zu wenig überlappende Daten für die Kalibrierung.")
    else:
        pd_calib_scored = score_calibration(pd_calib_grid, _pd_dca_floor, _pd_downside_tol)
        _n_feas = int(pd_calib_scored["feasible"].sum())
        st.caption(f"{len(pd_calib_grid):,} Engine-Läufe × 2 (Strategie + Benchmark) · "
                   f"{pd_calib_grid['start'].nunique()} rollierende 3-Jahres-Fenster · "
                   f"{pd_calib_scored['n'].iloc[0]} Fenster je Kombination · "
                   f"**{_n_feas} von {len(pd_calib_scored)}** Kombinationen zulässig")

        if _n_feas == 0:
            st.error("⚠️ Keine Kombination erfüllt beide Constraints bei dieser "
                     "Einstellung. Lockere den DCA-Floor oder die Downside-Toleranz.")
        else:
            _rec = pd_calib_scored[pd_calib_scored["feasible"]].iloc[0]
            r1, r2, r3, r4, r5 = st.columns(5)
            with r1:
                st.metric("Empfohlene Startallokation", f"{_rec['alloc']*100:.1f}%")
            with r2:
                st.metric("Empfohlener Cash-Sleeve", f"{_rec['cash_pct']*100:.1f}%")
            with r3:
                st.metric("Bandbreite", f"{_rec['width']*100:.0f}pp")
                st.caption(f"Band {max(_rec['alloc']*2,0.05)*100:.1f}–"
                           f"{(max(_rec['alloc']*2,0.05)+_rec['width'])*100:.1f}%")
            with r4:
                st.metric("Median-Netto-CAGR", f"{_rec['median_cagr']*100:.1f}%")
            with r5:
                st.metric("Median DCA-Anteil", f"{_rec['median_dca']*100:.0f}%")
            st.caption(
                "Empfehlung = höchste Median-CAGR unter allen Kombinationen, die "
                "beide Constraints erfüllen. Kein Bestwert-Cherrypicking: basiert "
                "auf dem Median über alle rollierenden Fenster, nicht auf einem "
                "einzelnen Startdatum.")

            st.markdown("##### Zulässige Kombinationen (nach Median-CAGR)")
            _disp = pd_calib_scored[pd_calib_scored["feasible"]].head(10).copy()
            _disp["Startallokation"] = (_disp["alloc"] * 100).round(1).astype(str) + "%"
            _disp["Cash-Sleeve"] = (_disp["cash_pct"] * 100).round(1).astype(str) + "%"
            _disp["Bandbreite"] = (_disp["width"] * 100).round(0).astype(int).astype(str) + "pp"
            _disp["Median-CAGR"] = (_disp["median_cagr"] * 100).round(2).astype(str) + "%"
            _disp["P5-CAGR"] = (_disp["p05_cagr"] * 100).round(2).astype(str) + "%"
            _disp["DCA-Anteil"] = (_disp["median_dca"] * 100).round(0).astype(str) + "%"
            _disp["Streuung"] = (_disp["spread"] * 100).round(1).astype(str) + "pp"
            st.dataframe(_disp[["Startallokation", "Cash-Sleeve", "Bandbreite",
                                "Median-CAGR", "P5-CAGR", "DCA-Anteil", "Streuung"]],
                        use_container_width=True, hide_index=True)

            st.markdown("##### Effizienzlinie — Downside-Schutz vs. Median-Rendite")
            _fig_ef = go.Figure()
            _infeas = pd_calib_scored[~pd_calib_scored["feasible"]]
            _feas = pd_calib_scored[pd_calib_scored["feasible"]]
            _fig_ef.add_trace(go.Scatter(
                x=_infeas["p05_cagr"] * 100, y=_infeas["median_cagr"] * 100,
                mode="markers", name="nicht zulässig",
                marker=dict(size=7, color=OAK_GREEN_3, opacity=0.5)))
            _fig_ef.add_trace(go.Scatter(
                x=_feas["p05_cagr"] * 100, y=_feas["median_cagr"] * 100,
                mode="markers", name="zulässig",
                marker=dict(size=9 + _feas["median_dca"] * 10, color=OAK_GOLD,
                           line=dict(width=1, color=OAK_CREAM))))
            _fig_ef.add_trace(go.Scatter(
                x=[_rec["p05_cagr"] * 100], y=[_rec["median_cagr"] * 100],
                mode="markers", name="Empfehlung",
                marker=dict(size=16, symbol="star", color=OAK_BTC,
                           line=dict(width=1.5, color=OAK_CREAM))))
            _fig_ef.update_xaxes(title_text="P5 Netto-CAGR — Downside (%)")
            _fig_ef.update_yaxes(title_text="Median Netto-CAGR (%)")
            _fig_ef = style_plotly(_fig_ef, height=420)
            st.plotly_chart(_fig_ef, use_container_width=True)
            st.caption("Punktgrösse = DCA-Anteil ex-subs. Grüne Punkte verfehlen "
                       "mindestens einen Constraint — meist hohe Startallokation "
                       "(gute Median-Rendite, aber der Mechanismus wird kannibalisiert).")

        st.warning(
            "⚠️ **Provisorisch, solange mit synthetischen Testpfaden gerechnet "
            "wird.** Diese Rastersuche läuft auf denselben echten Kursen "
            "(`btc_chf`, yfinance) wie der Rest der Seite — im Streamlit-"
            "Deployment liefert derselbe Button die launch-tauglichen Zahlen.")

# ==========================================================================
# ROBUSTHEIT — Grid über Startallokation × Fee, rollierende Startzeitpunkte,
# Break-even-Fee. Beantwortet die drei Positionierungs-Fragen:
#   1. Welche Startallokation?  2. Welche Management Fee?
#   3. Trägt der Ertragsmechanismus (DCA-Anteil) über verschiedene Einstiege?
#
# Bewusst KEIN Bestwert-Cherrypicking: über rollierende Fenster wird die
# VERTEILUNG der Netto-CAGR gezeigt, nicht der Bestwert. Der Debt-Sleeve ist
# parametrisch (at par) → die Engine ist sehr schnell, das Grid billig.
# Der (datierte) Kreditschock wird im Grid neutralisiert: ein Einmal-Event auf
# festem Datum gehört nicht in eine Verteilung über rollierende Startzeitpunkte.
# ==========================================================================
st.markdown("---")
st.markdown("## Robustheit — Grid, Startzeitpunkt & Break-even")
st.markdown(
    "<p style='color:#A9B5A4;margin-top:-6px'>Ein einzelnes Startdatum ist keine "
    "Evidenz. Die Engine läuft hier über ein Parameter-Gitter (Startallokation × "
    "Management Fee) und über viele rollierende Startzeitpunkte — ausgewiesen wird "
    "die Verteilung. Der DCA-Anteil ist der <strong>Mechanismus-Anteil ohne "
    "Zeichnungseffekt</strong> (Zeichnungen kaufen pro-rata BTC mit und würden den "
    "rohen Wert sonst verzerren).</p>", unsafe_allow_html=True)


def _derive_band_pd(alloc):
    """Schwellen skalieren mit der Startallokation — fixe 10/25% wären bei
    kleiner Startallokation sinnlos (die Quote läge dauerhaft unter der unteren
    Schwelle, der Boost-Modus liefe permanent).
      untere Schwelle = Startallokation × 2  (Minimum 5%)
      obere Schwelle  = untere + 10pp
    """
    lo = max(alloc * 2.0, 0.05)
    return lo, lo + 0.10


@st.cache_data(ttl=3600, show_spinner=False)
def compute_pd_robustness_grid(_btc, base, allocs, fees, win_years, step_months):
    """Grid × rollierende Fenster → lange Tabelle (alloc, fee, start) ->
    net_cagr, dca_share (mechanismus-rein). Der Kreditschock wird neutralisiert."""
    full = pd.DatetimeIndex(_btc.index)
    if len(full) < 400:
        return pd.DataFrame()
    starts = pd.date_range(full[0], full[-1] - pd.DateOffset(years=win_years),
                           freq=f"{step_months}MS")
    rows = []
    for alloc in allocs:
        lo, up = _derive_band_pd(alloc)
        for fee in fees:
            for s in starts:
                e = s + pd.DateOffset(years=win_years)
                w = full[(full >= s) & (full <= e)]
                if len(w) < 300:
                    continue
                p = dict(base)
                p.update(initial_btc_pct=alloc, lower_threshold=lo,
                         upper_threshold=up, mgmt_fee=fee,
                         credit_shock_pct=0.0, credit_shock_date=None)
                t = run_pd_btc(_btc, w, p)
                if t.empty:
                    continue
                yrs = max((t.index[-1] - t.index[0]).days / 365.25, 1e-9)
                cagr = (t["total_value"].iloc[-1] / p["initial_capital"]) ** (1 / yrs) - 1
                a = t.attrs.get("attribution", {})
                rows.append({"alloc": alloc, "fee": fee, "start": s,
                             "net_cagr": cagr,
                             "dca_share": a.get("dca_share_ex_subs", np.nan)})
    return pd.DataFrame(rows)


_prb1, _prb2, _prb3 = st.columns([1, 1, 1])
with _prb1:
    _pd_win_years = st.selectbox("Fensterlänge (Jahre)", [3, 5], index=0,
                                 key="pd_rb_win")
with _prb2:
    _pd_step_label = st.selectbox("Fenster-Schritt", ["quartalsweise", "monatlich"],
                                  index=0, key="pd_rb_step",
                                  help="Monatlich ist gründlicher, dauert aber "
                                       "länger (mehr Läufe).")
with _prb3:
    st.caption("")
    _pd_run_rb = st.button("Robustheitsanalyse starten", key="pd_rb_go")

if _pd_run_rb:
    st.session_state["pd_rb_has_run"] = True

if st.session_state.get("pd_rb_has_run"):
    _pd_allocs = [0.0, 0.05, 0.075, 0.10, 0.20]
    _pd_fees = [0.0150, 0.0100, 0.0075, 0.0050]   # eigene Fee ≤1% p.a. (Gebühren-Stapel)
    _pd_step = 3 if _pd_step_label == "quartalsweise" else 1

    _pd_base = dict(params)   # aktuelle Sidebar-Parameter als Basis
    with st.spinner("Rechne Grid über alle rollierenden Fenster…"):
        pd_grid = compute_pd_robustness_grid(btc_chf, _pd_base, _pd_allocs,
                                             _pd_fees, _pd_win_years, _pd_step)

    if pd_grid.empty:
        st.warning("Zu wenig überlappende Daten für die Fensteranalyse.")
    else:
        _pd_n_win = pd_grid["start"].nunique()
        st.caption(f"{len(pd_grid):,} Engine-Läufe · {_pd_n_win} rollierende "
                   f"{_pd_win_years}-Jahres-Fenster · Schwellen skalieren mit der "
                   f"Startallokation (untere = Start × 2, min. 5%; obere = untere + 10pp) "
                   f"· Kreditschock im Grid neutralisiert")

        # ---- 1) Grid-Matrix: Startallokation × Fee -> Median Netto-CAGR -----
        st.markdown("##### Netto-CAGR (Median über alle Fenster) — Startallokation × Management Fee")
        _ppiv = (pd_grid.groupby(["alloc", "fee"])["net_cagr"].median().unstack() * 100)
        _pfig_g = go.Figure(data=go.Heatmap(
            z=_ppiv.values,
            x=[f"{f*100:.2f}%" for f in _ppiv.columns],
            y=[f"{a*100:.1f}%" for a in _ppiv.index],
            colorscale=[[0, OAK_GREEN_2], [0.5, OAK_SAGE], [1, OAK_GOLD]],
            text=[[f"{v:.1f}%" for v in r] for r in _ppiv.values],
            texttemplate="%{text}", showscale=False))
        _pfig_g.update_xaxes(title_text="Management Fee (p.a.)", type="category")
        _pfig_g.update_yaxes(title_text="Startallokation BTC", type="category")
        _pfig_g = style_plotly(_pfig_g, height=340)
        st.plotly_chart(_pfig_g, use_container_width=True)

        # ---- 2) DCA-Anteil-Matrix (mechanismus-rein) — hält der Name? -------
        st.markdown("##### DCA-Anteil am BTC-Gewinn (Median, ohne Zeichnungseffekt) — hält der Name, was er verspricht?")
        _ppivd = (pd_grid.groupby(["alloc", "fee"])["dca_share"].median().unstack() * 100)
        _pfig_d = go.Figure(data=go.Heatmap(
            z=_ppivd.values,
            x=[f"{f*100:.2f}%" for f in _ppivd.columns],
            y=[f"{a*100:.1f}%" for a in _ppivd.index],
            colorscale=[[0, "#8C3A2B"], [0.3, OAK_GREEN_3], [1, OAK_GOLD]],
            text=[[("n/a" if v != v else f"{v:.0f}%") for v in r] for r in _ppivd.values],
            texttemplate="%{text}", showscale=False, zmin=0, zmax=100))
        _pfig_d.update_xaxes(title_text="Management Fee (p.a.)", type="category")
        _pfig_d.update_yaxes(title_text="Startallokation BTC", type="category")
        _pfig_d = style_plotly(_pfig_d, height=340)
        st.plotly_chart(_pfig_d, use_container_width=True)
        st.caption("Ein tiefer Wert heisst NICHT «Mechanismus schwach», sondern «der "
                   "Einstieg lag günstig». Bei 0% Startallokation ist der Anteil 100% "
                   "(jeder BTC-Gewinn stammt dann aus dem geernteten Ertrag).")

        # ---- 3) Verteilung der Netto-CAGR je Startallokation ----------------
        st.markdown("##### Verteilung der Netto-CAGR je Startallokation (alle Fenster, alle Fees)")
        _pfig_b = go.Figure()
        for a in _pd_allocs:
            v = pd_grid.loc[pd_grid["alloc"] == a, "net_cagr"] * 100
            _pfig_b.add_trace(go.Box(y=v, name=f"{a*100:.1f}%",
                                     marker_color=OAK_GOLD, line_color=OAK_SAGE,
                                     boxmean=True))
        _pfig_b.update_yaxes(title_text="Netto-CAGR (%)")
        _pfig_b.update_xaxes(title_text="Startallokation BTC", type="category")
        _pfig_b = style_plotly(_pfig_b, height=380)
        _pfig_b.update_layout(showlegend=False)
        st.plotly_chart(_pfig_b, use_container_width=True)

        _pd_dist = (pd_grid.groupby("alloc")["net_cagr"]
                    .agg(Minimum="min", P25=lambda s: s.quantile(.25), Median="median",
                         P75=lambda s: s.quantile(.75), Maximum="max") * 100).round(2)
        _pd_dist["Streuung"] = (_pd_dist["Maximum"] - _pd_dist["Minimum"]).round(2)
        _pd_dist.index = [f"{a*100:.1f}%" for a in _pd_dist.index]
        _pd_dist.index.name = "Startallokation"
        st.dataframe(_pd_dist.style.format("{:.2f}%"), use_container_width=True)

        st.caption(
            "⚠️ **Das Minimum ist kein Risikomass.** Es steigt mit der "
            "Startallokation, weil die Datenreihe kein einziges Mehrjahres-Fenster "
            "mit einem Bitcoin-Kollaps ohne Erholung enthält — die Stichprobe "
            "enthält das Risiko schlicht nicht. Das belastbare Signal ist die "
            "**Streuung**: wie stark das Ergebnis vom Einstiegszeitpunkt abhängt.")

        # ---- 4) Worst-Entry — die Zahl für den Investor --------------------
        st.markdown("##### Worst-Entry — was passiert dem Investor mit dem schlechtesten Einstieg?")
        _pd_cur = min(_pd_allocs, key=lambda a: abs(a - initial_btc_pct))
        _pd_g = pd_grid[(pd_grid["alloc"] == _pd_cur)
                        & (np.isclose(pd_grid["fee"], mgmt_fee))]
        if _pd_g.empty:
            _pd_g = pd_grid[pd_grid["alloc"] == _pd_cur]
        if not _pd_g.empty:
            _pd_worst = _pd_g.loc[_pd_g["net_cagr"].idxmin()]
            _pw1, _pw2, _pw3, _pw4 = st.columns(4)
            with _pw1:
                st.metric("Schlechtestes Fenster", f"{_pd_worst['net_cagr']*100:.1f}% p.a.")
                st.caption(f"Einstieg {_pd_worst['start']:%b %Y}")
            with _pw2:
                st.metric("DCA-Anteil dort",
                          "n/a" if _pd_worst["dca_share"] != _pd_worst["dca_share"]
                          else f"{_pd_worst['dca_share']*100:.0f}%")
                st.caption("Yield Bridge im Stressfall")
            with _pw3:
                st.metric("Median", f"{_pd_g['net_cagr'].median()*100:.1f}% p.a.")
                st.caption("mittleres Fenster")
            with _pw4:
                _pd_spr = (_pd_g["net_cagr"].max() - _pd_g["net_cagr"].min()) * 100
                st.metric("Streuung", f"{_pd_spr:.0f} pp")
                st.caption("Max − Min über alle Fenster")
            st.caption(
                f"Bei {_pd_cur*100:.1f}% Startallokation und {mgmt_fee*100:.2f}% Fee. "
                "Je schlechter der Einstieg, desto WICHTIGER wird der "
                "ertragsfinanzierte DCA — er kauft antizyklisch nach, während das "
                "Kreditbuch unangetastet weiter Ertrag abwirft.")

        st.session_state["pd_rb_dist"] = _pd_dist

        # ---- 5) Feste Startzeitpunkte — Zyklushoch vs. Zyklustief ----------
        st.markdown("##### Feste Startzeitpunkte — Zyklushoch vs. Zyklustief")
        _pd_fixed = {"2018-01-01 (BTC nahe Zyklushoch)": "2018-01-01",
                     "2019-01-01 (nahe Bärenmarkt-Tief)": "2019-01-01",
                     "2021-11-01 (Zyklushoch)": "2021-11-01",
                     "2022-06-01 (Zyklustief)": "2022-06-01"}
        _pd_frows = []
        _pd_full = pd.DatetimeIndex(btc_chf.index)
        for lab, s in _pd_fixed.items():
            s = pd.Timestamp(s)
            w = _pd_full[_pd_full >= s]
            if len(w) < 300:
                _pd_frows.append({"Startzeitpunkt": lab, "Netto-CAGR": np.nan,
                                  "DCA-Anteil": np.nan})
                continue
            p = dict(params)
            _lo, _up = _derive_band_pd(initial_btc_pct)
            p.update(lower_threshold=_lo, upper_threshold=_up,
                     credit_shock_pct=0.0, credit_shock_date=None)
            t = run_pd_btc(btc_chf, w, p)
            yrs = max((t.index[-1] - t.index[0]).days / 365.25, 1e-9)
            cg = (t["total_value"].iloc[-1] / float(initial_capital)) ** (1 / yrs) - 1
            ds = t.attrs.get("attribution", {}).get("dca_share_ex_subs", np.nan)
            _pd_frows.append({"Startzeitpunkt": lab, "Netto-CAGR": cg * 100,
                              "DCA-Anteil": ds * 100})
        _pd_fdf = pd.DataFrame(_pd_frows).set_index("Startzeitpunkt")
        st.dataframe(_pd_fdf.style.format("{:.1f}%", na_rep="n/a"),
                     use_container_width=True)
        st.session_state["pd_rb_fixed"] = _pd_fdf
        st.caption(f"Mit der aktuellen Startallokation ({initial_btc_pct*100:.0f}%) und "
                   "daraus abgeleiteten Schwellen. Die Spreizung zwischen den "
                   "Startzeitpunkten zeigt, wie stark das Ergebnis vom Einstieg abhängt.")

# ---- 6) Break-even-Fee: ab wann frisst die Gebühr den ganzen Ertrag? -------
st.markdown("##### Break-even Management Fee — ab wann bleibt für den DCA nichts mehr?")
_pd_debt_share = max(1.0 - initial_btc_pct - initial_cash_pct, 1e-9)
_pd_be_fee = net_yield * _pd_debt_share
_pbc1, _pbc2, _pbc3 = st.columns(3)
with _pbc1:
    st.metric("Break-even Fee", f"{_pd_be_fee*100:.2f}% p.a.")
    st.caption("Nettorendite × Debt-Anteil am AuM")
with _pbc2:
    st.metric("Aktuelle Fee", f"{mgmt_fee*100:.2f}% p.a.")
    st.caption("Management Fee laut Sidebar")
with _pbc3:
    _pd_head = _pd_be_fee - mgmt_fee
    st.metric("Puffer bis Break-even", f"{_pd_head*100:+.2f} pp",
              delta_color="normal" if _pd_head > 0 else "inverse")
    st.caption("positiv = Ertrag trägt die Fee")
if mgmt_fee >= _pd_be_fee:
    st.warning(
        f"⚠️ **Die Management Fee ({mgmt_fee*100:.2f}%) erreicht oder übersteigt "
        f"den Break-even ({_pd_be_fee*100:.2f}%)** — die Gebühr frisst den gesamten "
        "geernteten Ertrag, für den BTC-DCA bleibt nichts. Die Yield Bridge steht "
        "still, bis Fee < Ertrag.")
else:
    st.caption(
        "Hinweis: Die Fee wird aktuell auf das **Gesamt-AuM** verbucht (inkl. BTC "
        "und Cash), nicht nur auf den Debt-Sleeve — deshalb Break-even = Nettorendite "
        "× Debt-Anteil. Würde die Fee auf den Debt-Sleeve beschränkt, läge der "
        f"Break-even bei der vollen Nettorendite ({net_yield*100:.2f}%).")

# --------------------------------------------------------------------------
# PDF-Tearsheet (bilingual DE/EN)
# --------------------------------------------------------------------------
st.markdown("---")
if st.button("PDF-Tearsheet generieren (DE+EN)"):
    with st.spinner("Erzeuge PDF…"):
        try:
            # ---- Charts -------------------------------------------------
            _lines = [("OAK Swiss Private Debt / Bitcoin (Net of Fees)", net,
                       "#B8954A", {"lw": 2.2}),
                      ("Debt only (same model, no BTC)", bench_debt, "#7C8978",
                       {"ls": "--", "lw": 1.6})]
            if bench_bond is not None:
                _lines.append((f"{bench_ticker} (bonds, CHF)", bench_bond,
                               "#7FA7C4", {"lw": 1.3}))
            _fig_evo = render_line_chart(_lines, ylabel="Value (CHF)")

            _fig_sleeve = render_line_chart(
                [("Loan book", ts["debt_value"], "#7C8978", {"lw": 1.8}),
                 ("Bitcoin", ts["btc_value"], "#F7931A", {"lw": 1.8}),
                 ("CHF Cash", ts["cash"], "#D4D4CE", {"lw": 1.5})],
                ylabel="Value (CHF)")

            # Kostenbasis vs. NAV — der Beweis, dass das Kapital unangetastet bleibt
            _fig_basis = render_line_chart(
                [("Cost basis (capital invested)", ts["debt_cost_basis"],
                  "#7C8978", {"lw": 1.8}),
                 ("Loan-book NAV (incl. accrued yield)", ts["debt_value"],
                  "#B8954A", {"lw": 1.8})],
                ylabel="CHF")

            _dd_s = compute_drawdown(net)
            _fig_dd = render_line_chart(
                [("Strategy (Net)", _dd_s * 100, "#B8954A", {"lw": 1.6})],
                ylabel="Drawdown (%)")

            _figs = [("Portfolio Evolution vs. Debt-only & Bond Benchmark", _fig_evo),
                     ("Sleeve Development", _fig_sleeve),
                     ("Cost Basis vs. NAV — the gap is unharvested yield", _fig_basis),
                     ("Drawdown Analysis*", _fig_dd)]
            _figs = [(t, f) for t, f in _figs if f is not None]

            # ---- Zahlen -------------------------------------------------
            _tot_ret = (net.iloc[-1] / _cap_base - 1) * 100
            _kpis_perf = [("Strategy (Net)", fmt_chf(net.iloc[-1])),
                          ("Net CAGR", f"{net_cagr*100:.2f}%"),
                          ("Debt only", fmt_chf(bench_debt.iloc[-1])),
                          ("BTC Contribution", f"{btc_contrib*100:+.2f}% p.a.")]
            _kpis_risk = [("Sharpe Ratio*", f"{m['sharpe']:.2f}"),
                          ("Sortino Ratio*", f"{m['sortino']:.2f}"),
                          ("Max Drawdown*", f"{m['max_drawdown']*100:.2f}%"),
                          ("Volatility*", f"{m['vol_ann']*100:.2f}%")]
            _fees = [("Mgmt Fees", fmt_chf(total_mgmt)),
                     ("Perf Fees", fmt_chf(total_perf)),
                     ("Total Fees", fmt_chf(total_mgmt + total_perf)),
                     ("Fee Drag", f"{fee_drag*100:.2f}% p.a.")]

            _risk_rows = [
                ["Total Return", f"{_tot_ret:.2f}%",
                 f"{(bench_debt.iloc[-1]/_cap_base - 1)*100:.2f}%"],
                ["CAGR", f"{net_cagr*100:.2f}%", f"{debt_cagr*100:.2f}%"],
                ["Volatility*", f"{m['vol_ann']*100:.2f}%", "—"],
                ["Max Drawdown*", f"{m['max_drawdown']*100:.2f}%", "—"],
                ["Sharpe Ratio*", f"{m['sharpe']:.2f}", "—"],
                ["Sortino Ratio*", f"{m['sortino']:.2f}", "—"],
            ]

            _fe = ts[["mgmt_fee_only", "perf_fee_only"]].copy()
            _fe = _fe[(_fe["mgmt_fee_only"] > 0) | (_fe["perf_fee_only"] > 0)]
            _fee_rows = []
            if not _fe.empty:
                _g = _fe.groupby([_fe.index.year, _fe.index.quarter]).sum()
                for (yy, qq), r in _g.iterrows():
                    _fee_rows.append([f"Q{qq} {yy}",
                                      f"CHF {r['mgmt_fee_only']:,.0f}",
                                      f"CHF {r['perf_fee_only']:,.0f}",
                                      f"CHF {r['mgmt_fee_only']+r['perf_fee_only']:,.0f}"])

            _hr = _att["harvest_ratio"]
            _params = [
                ("Allocation Framework",
                 ("OAK Yield Bridge (pure) — the satellite is funded exclusively by "
                  "harvested loan-book yield; the capital base is never touched"
                  if initial_btc_pct <= 0 else
                  f"OAK Yield Bridge with strategic initial allocation — "
                  f"{initial_btc_pct*100:.0f}% of capital allocated to Bitcoin on day 1")),
                ("DCA Share of BTC Gain",
                 "n/a" if _ds != _ds else f"{_ds*100:.1f}% (harvest-funded vs. day-1 lump sum)"),
                ("Initial Capital", f"CHF {initial_capital:,.0f}"),
                ("Subscriptions (total)", fmt_chf(_att["subscriptions"])),
                ("Initial Allocation",
                 f"{(1-initial_btc_pct-initial_cash_pct)*100:.0f}% Loan book / "
                 f"{initial_btc_pct*100:.0f}% BTC / {initial_cash_pct*100:.0f}% Cash"),
                ("Underlying (reference)",
                 "LEND Hypovest, ISIN CH1357099691 — diversified Swiss subordinated "
                 "mortgages (accumulating, no coupon)"),
                ("Net Yield", f"{net_yield*100:.1f}% p.a. (parametric, on cost basis)"),
                ("Credit Losses", f"{credit_loss_rate*100:.1f}% p.a."),
                ("Yield Realisation",
                 "Monthly harvest — only the NAV accretion above cost basis is "
                 "redeemed; the capital base is never sold"),
                ("Redemption Success Rate",
                 f"{redemption_rate*100:.0f}% (best-effort; unharvested yield stays "
                 f"in the certificate and keeps compounding)"),
                ("Harvest Realised",
                 "n/a" if _hr != _hr else f"{_hr*100:.0f}% of accrued yield"),
                ("BTC Band",
                 f"{lower_threshold*100:.0f}% – {upper_threshold*100:.0f}%"),
                ("Investment Rate (harvest)",
                 f"{base_invest_rate*100:.0f}% in band / "
                 f"{boost_invest_rate*100:.0f}% below lower threshold"),
                ("Transaction Cost", f"{tx_cost_bps:.0f} bps per trade"),
                ("Management Fee", f"{mgmt_fee*100:.2f}% p.a."),
                ("Performance Fee",
                 f"{perf_fee*100:.0f}% ({crystallization_freq}, {hurdle_type} "
                 f"{hurdle*100:.1f}% Yr 1)"),
            ]

            _universe = [
                ["Bitcoin", "BTC", "Digital Assets",
                 f"Band {lower_threshold*100:.0f}–{upper_threshold*100:.0f}%"],
                ["CH mortgage loan book (parametric)", "CH1357099691",
                 "Private Debt",
                 f"{(1-initial_btc_pct-initial_cash_pct)*100:.0f}% initial"],
                ["CHF Cash", "—", "Cash", "Fee cushion · buffer"],
            ]

            # ---- Renditezerlegung als eigene PDF-Sektion -----------------
            def _xt(lang):
                de = (lang == "de")
                labels = ([("Ertragsanfall Kreditbuch (brutto)", "debt_income"),
                           ("Kreditausfälle", "credit_losses"),
                           ("Bitcoin — Startallokation (Tag 1)", "btc_initial_gain"),
                           ("Bitcoin — ertragsfinanzierter DCA", "btc_dca_gain"),
                           ("Cash-Zins (Puffer)", "cash_interest"),
                           ("Gebühren", "fees")] if de else
                          [("Loan-book yield accrued (gross)", "debt_income"),
                           ("Credit losses", "credit_losses"),
                           ("Bitcoin — initial allocation (day 1)", "btc_initial_gain"),
                           ("Bitcoin — harvest-funded DCA", "btc_dca_gain"),
                           ("Cash interest (buffer)", "cash_interest"),
                           ("Fees", "fees")])
                rows = [[lab, f"{_att.get(k, 0.0):+,.0f}", f"{_pp(_att.get(k, 0.0)):+.2f}"]
                        for lab, k in labels]
                rows.append([("Total (= NAV − Startkapital − Zeichnungen)" if de else
                              "Total (= NAV − initial capital − subscriptions)"),
                             f"{_att['total_pnl']:+,.0f}",
                             f"{_pp(_att['total_pnl']):+.2f}"])
                _dstxt = "n/a" if _ds != _ds else f"{_ds*100:.1f}%"
                out = [{
                    "eyebrow": "08",
                    "title": "Renditezerlegung" if de else "Return Attribution",
                    "subtitle": (
                        f"DCA-Anteil am BTC-Gewinn: {_dstxt}. Zeichnungen sind kein "
                        f"Gewinn und werden herausgerechnet."
                        if de else
                        f"DCA share of the BTC gain: {_dstxt}. Subscriptions are not a "
                        f"gain and are excluded."),
                    "headers": (["Beitrag", "CHF", "%-Punkte p.a."] if de else
                                ["Contribution", "CHF", "pp p.a."]),
                    "rows": rows,
                    "note": (
                        "Der DCA-Anteil ist invers zum Einstiegsglück: je schlechter "
                        "der Einstiegszeitpunkt, desto grösser der Beitrag des "
                        "ertragsfinanzierten DCA. Höhere Zeichnungen erhöhen die "
                        "Ertragsbasis und damit den DCA-Anteil."
                        if de else
                        "The DCA share is inverse to entry luck: the worse the entry "
                        "point, the larger the contribution of the harvest-funded DCA. "
                        "Higher subscriptions grow the yield base and therefore the "
                        "DCA share."),
                }]
                # Kapitalerhaltung — die Kennzahl, die das Produkt verkauft
                out.append({
                    "eyebrow": "09",
                    "title": ("Kapitalerhaltung — Bitcoin auf null" if de else
                              "Capital Preservation — Bitcoin at zero"),
                    "subtitle": (
                        "Das Kapital liegt im besicherten Kreditbuch. Bitcoin wird "
                        "ausschliesslich aus dem geernteten Ertrag gekauft — das "
                        "eingesetzte Kapital ist strukturell nicht dem Bitcoin-Risiko "
                        "ausgesetzt."
                        if de else
                        "The capital sits in the secured loan book. Bitcoin is bought "
                        "exclusively from harvested yield — the invested capital is "
                        "structurally not exposed to Bitcoin risk."),
                    "headers": (["Kennzahl", "Wert"] if de else ["Metric", "Value"]),
                    "rows": [
                        [("Eingesetztes Kapital" if de else "Capital invested"),
                         f"CHF {_cap_base:,.0f}"],
                        [("NAV, wenn Bitcoin auf null ginge" if de else
                          "NAV if Bitcoin went to zero"), f"CHF {_zero_end:,.0f}"],
                        [("Kapitaldeckung" if de else "Capital coverage"),
                         f"{_zero_ratio*100:.1f}%"],
                        [("Tiefste Deckung im Zeitverlauf" if de else
                          "Lowest coverage over time"),
                         f"{_cov_min*100:.1f}% ({_cov_min_at:%b %Y})"],
                        [("Modellierter Kreditschock" if de else "Credit shock modelled"),
                         ("keiner" if credit_shock_pct <= 0 else
                          f"−{credit_shock_pct*100:.0f}% (CHF {_att['credit_shock']:,.0f})")
                         if de else
                         ("none" if credit_shock_pct <= 0 else
                          f"−{credit_shock_pct*100:.0f}% (CHF {_att['credit_shock']:,.0f})")],
                    ],
                    "note": (
                        "Ein Kreditschock frisst zuerst den aufgelaufenen Ertrag und "
                        "danach die Kapitalsubstanz. Solange der Buchwert unter der "
                        "Kostenbasis liegt, gibt es NICHTS zu ernten — der "
                        "Bitcoin-DCA pausiert vollständig, bis der laufende Ertrag den "
                        "Verlust aufgeholt hat. Das ist der wesentliche Risikofaktor "
                        "dieser Struktur und wichtiger als die Bitcoin-Volatilität."
                        if de else
                        "A credit shock first consumes accrued yield and then eats into "
                        "capital. As long as the book value sits below the cost basis "
                        "there is NOTHING to harvest — the Bitcoin DCA pauses entirely "
                        "until current yield has made good the loss. This is the "
                        "material risk of the structure and matters more than Bitcoin "
                        "volatility."),
                })
                # Ernte & Redemption — die produktspezifische Sektion
                out.append({
                    "eyebrow": "10",
                    "title": ("Ertrags-Ernte & Redemption-Stress" if de else
                              "Yield Harvesting & Redemption Stress"),
                    "subtitle": (
                        "Das Underlying ist thesaurierend. Der Ertrag wird einmal "
                        "jährlich (Januar) geerntet — es wird nur der aufgelaufene "
                        "NAV-Zuwachs über der Kostenbasis redimiert, nie das Kapital."
                        if de else
                        "The underlying is accumulating. Yield is harvested once a "
                        "year (January) — "
                        "only the NAV accretion above the cost basis is redeemed, never "
                        "the capital base."),
                    "headers": (["Kennzahl", "Wert"] if de else ["Metric", "Value"]),
                    "rows": [
                        [("Ertragsanfall (brutto)" if de else "Yield accrued (gross)"),
                         f"CHF {_att['accrued_total']:,.0f}"],
                        [("Davon geerntet" if de else "Of which harvested"),
                         f"CHF {_att['harvest_total']:,.0f}"],
                        [("Ungeerntet (klemmt im Zertifikat)" if de else
                          "Unharvested (stuck in the certificate)"),
                         f"CHF {_att['unharvested']:,.0f}"],
                        [("Redemption-Erfolgsquote" if de else "Redemption success rate"),
                         f"{redemption_rate*100:.0f}%"],
                        [("Kostenbasis Start" if de else "Cost basis at start"),
                         f"CHF {ts['debt_cost_basis'].iloc[0]:,.0f}"],
                        [("Kostenbasis Ende" if de else "Cost basis at end"),
                         f"CHF {ts['debt_cost_basis'].iloc[-1]:,.0f}"],
                    ],
                    "note": (
                        "Rücknahmen erfolgen best-effort (kein Sekundärmarkt). Klemmen "
                        "sie, ist der Ertrag NICHT verloren — er bleibt im Zertifikat, "
                        "verzinst sich weiter und wird nachgeholt, sobald Rücknahmen "
                        "wieder gefüllt werden. Der DCA pausiert lediglich, und zwar "
                        "typischerweise in Stressphasen, in denen aggressive Zukäufe "
                        "ohnehin fragwürdig wären. Die Kostenbasis verändert sich "
                        "ausschliesslich durch neue Zeichnungen."
                        if de else
                        "Redemptions are best-effort (no secondary market). If they are "
                        "not filled, the yield is NOT lost — it stays in the certificate, "
                        "keeps compounding, and is caught up once redemptions clear. The "
                        "DCA merely pauses, typically in stress phases where aggressive "
                        "buying would be questionable anyway. The cost basis changes only "
                        "through new subscriptions."),
                })
                return out

            _disc_de = [
                "Dieses Dokument wurde von Oakwood Capital ausschliesslich zu "
                "illustrativen und informativen Zwecken erstellt. Es stellt weder eine "
                "Anlageberatung, eine Empfehlung, ein Angebot noch eine Aufforderung "
                "zum Kauf oder Verkauf eines Finanzinstruments dar.",

                "OAK Swiss Private Debt / Bitcoin ist eine PARAMETRISCHE SIMULATION und "
                "kein marktdatenbasierter Backtest. Für den Debt-Sleeve existiert keine "
                "verwertbare Kurshistorie (das Referenz-Underlying LEND Hypovest, ISIN "
                "CH1357099691, wurde im Juli 2024 emittiert). Die Nettorendite ist eine "
                "gesetzte Annahme, keine realisierte Grösse. Der Debt-Sleeve wird zum "
                "Nennwert geführt und weist konstruktionsbedingt KEINE "
                "Kapitalwertschwankung auf; Volatilität, Sharpe Ratio und "
                "Drawdown-Kennzahlen der Gesamtstrategie sind dadurch systematisch nach "
                "unten verzerrt und nicht mit marktbewerteten Strategien vergleichbar.",

                "Das Referenz-Underlying ist ein THESAURIERENDES Zertifikat auf "
                "nachrangige, immobilienbesicherte Schweizer Kredite — es schüttet "
                "nichts aus. Der Ertrag wird in der Simulation über eine jährliche "
                "Ernte (Default Januar) realisiert (Rücknahme des NAV-Zuwachses über der Kostenbasis). "
                "Rücknahmen erfolgen best-effort; es besteht kein Sekundärmarkt und "
                "damit keine Garantie, dass die Ernte in der modellierten Höhe "
                "tatsächlich möglich ist. Nachrangige Kredite tragen das "
                "First-Loss-Risiko; es besteht kein Kapitalschutz. Gebühren des "
                "Underlyings (Investor Fee, Zeichnungsgebühr) kommen zu den hier "
                "ausgewiesenen Gebühren HINZU und sind in der Nettorendite-Annahme zu "
                "berücksichtigen.",

                "Der Bitcoin-Anteil basiert auf historischen Marktpreisen (BTC/USD, in "
                "Schweizer Franken umgerechnet). Digitale Vermögenswerte sind "
                "hochvolatil und können zum Totalverlust des eingesetzten Kapitals "
                "führen.",

                "Neue Zeichnungen sind KEIN Anlageerfolg. Sämtliche Renditekennzahlen "
                "sind auf die Summe aus Anfangskapital und Zeichnungen bezogen. Steuern "
                "sind nicht modelliert. Die simulierte Performance ist hypothetisch, "
                "unterliegt dem Vorteil der Rückschau und ist kein verlässlicher "
                "Indikator für zukünftige Ergebnisse.",

                "Dieses Material ist streng vertraulich und ausschliesslich für den "
                "Empfänger bestimmt.",
            ]
            _disc_en = [
                "This document has been prepared by Oakwood Capital for illustrative and "
                "informational purposes only. It does not constitute investment advice, a "
                "recommendation, an offer, or a solicitation to buy or sell any financial "
                "instrument.",

                "OAK Swiss Private Debt / Bitcoin is a PARAMETRIC SIMULATION, not a "
                "market-data backtest. No usable price history exists for the debt sleeve "
                "(the reference underlying, LEND Hypovest, ISIN CH1357099691, was issued "
                "in July 2024). The net yield is an assumption, not a realised figure. The "
                "debt sleeve is carried at par and by construction exhibits NO capital-value "
                "volatility; volatility, Sharpe ratio and drawdown figures for the overall "
                "strategy are therefore systematically understated and not comparable to "
                "market-priced strategies.",

                "The reference underlying is an ACCUMULATING certificate on subordinated, "
                "real-estate-secured Swiss loans — it pays no coupon. In the simulation the "
                "yield is realised through an annual harvest (default January, redeeming "
                "the NAV accretion "
                "above the cost basis). Redemptions are best-effort; there is no secondary "
                "market and therefore no guarantee that the harvest is achievable at the "
                "modelled level. Subordinated loans carry first-loss risk; there is no "
                "capital protection. Fees of the underlying (investor fee, subscription "
                "fee) come IN ADDITION to the fees shown here and must be reflected in the "
                "net-yield assumption.",

                "The Bitcoin sleeve is based on historical market prices (BTC/USD converted "
                "to CHF). Digital assets are highly volatile and may result in total loss.",

                "New subscriptions are NOT investment performance. All return figures are "
                "based on the sum of initial capital and subscriptions. Taxes are not "
                "modelled. Simulated performance is hypothetical, benefits from hindsight, "
                "and is not a reliable indicator of future results.",

                "This material is strictly confidential and intended solely for the "
                "recipient.",
            ]

            pdf_bytes = build_bilingual_tearsheet(
                strategy_name="OAK Swiss Private Debt / Bitcoin",
                strategy_subtitle_de=(
                    "Immobilienbesichertes Schweizer Kreditbuch mit struktureller "
                    "Bitcoin-Allokation, ertragsfinanziertem DCA und "
                    "schwellenwertbasiertem Risikomanagement."),
                strategy_subtitle_en=(
                    "Swiss real-estate-secured loan book with a structural Bitcoin "
                    "allocation, harvest-funded DCA and threshold-based risk "
                    "management."),
                period_str=f"{net.index[0]:%Y-%m-%d} to {net.index[-1]:%Y-%m-%d}",
                kpis_performance=_kpis_perf,
                kpis_risk=_kpis_risk,
                fee_summary=_fees,
                risk_table_headers=["Metric", "Strategy (Net)", "Debt only"],
                risk_table_rows=_risk_rows,
                fee_table_headers=["Period", "Mgmt Fee", "Perf Fee", "Total Cost"],
                fee_table_rows=_fee_rows,
                figures=_figs,
                params_summary=_params,
                universe_rows=_universe,
                period_returns=compute_period_returns(net, bench_debt),
                perf_summary_sub_de=(
                    "Nach Gebühren und Transaktionskosten · *Der Debt-Sleeve wird zum "
                    "Nennwert geführt — Risikokennzahlen sind nach unten verzerrt"),
                perf_summary_sub_en=(
                    "Net of fees and transaction costs · *The debt sleeve is carried at "
                    "par — risk metrics are understated"),
                benchmark_label_de="Nur Kreditbuch (gleiches Modell)",
                benchmark_label_en="Debt only (same model)",
                universe_sub_de=(
                    "Drei Sleeves: ein immobilienbesichertes Schweizer Kreditbuch "
                    "(parametrisch, thesaurierend), eine bandgesteuerte "
                    "Bitcoin-Allokation und ein CHF-Cash-Puffer. Siehe Methodik und "
                    "Hinweise."),
                universe_sub_en=(
                    "Three sleeves: a Swiss real-estate-secured loan book (parametric, "
                    "accumulating), a band-managed Bitcoin allocation, and a CHF cash "
                    "buffer. See Methodology and Disclosures."),
                disclaimer_paragraphs_de=_disc_de,
                disclaimer_paragraphs_en=_disc_en,
                extra_tables_de=_xt("de"),
                extra_tables_en=_xt("en"),
            )

            _fn = f"OAK_Swiss_PrivateDebt_BTC_{date.today():%Y%m%d}.pdf"
            st.download_button("PDF herunterladen", data=pdf_bytes, file_name=_fn,
                               mime="application/pdf")
            st.success("Tearsheet erzeugt (DE + EN).")
        except Exception as exc:
            import traceback
            st.error(f"PDF-Erzeugung fehlgeschlagen: {exc}")
            st.code(traceback.format_exc())

st.markdown(
    f"""<div class='oak-footer'>
    Zu illustrativen Zwecken · Keine Anlageberatung · Parametrische Simulation ·
    Vergangene Wertentwicklung ist kein Indikator für zukünftige Ergebnisse
    <span class='oak-mark'>Oakwood Capital · Quantitatives Research</span>
    </div>""", unsafe_allow_html=True)
