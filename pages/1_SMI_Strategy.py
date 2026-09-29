"""
SMI/BTC Strategy Backtester — Oakwood Capital
=============================================
Integrated daily simulation with:
  - Initial allocation (calibrated default 85% SMI / 15% BTC)
  - Dividend harvesting → 6-month DCA into BTC (calibrated; Sharpe/Calmar-
    optimal among 3/6/9/12-month windows tested)
  - Threshold-based rebalancing: when BTC > upper threshold (25%, calibrated),
    sell down to target (15%) and reallocate to SMI by weight
  - Annual SMI rebalancing to target weights (September, calibrated —
    aligned with the real SIX index-review date)
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
# Aktien-Sleeve: zwanzig Einzeltitel oder ein SMI-ETF
# ---------------------------------------------------------------------------
# Alle drei ETFs sind physisch replizierend. Entscheidend ist der Unterschied
# zwischen ausschuettend und thesaurierend, und zwar NICHT steuerlich, sondern
# mechanisch:
#
#   ausschuettend   Dividende wird Kasse, Kasse wird ueber DCA-Fenster in
#                   Bitcoin investiert. Das braucht ein Tranchenregister und
#                   traegt Zustand ueber die Termine hinaus.
#   thesaurierend   keine Ausschuettung. Der Bitcointeil wird ueber einen
#                   festen monatlichen Abverkauf finanziert. Kein Register,
#                   kein Zustand, zwei Orderzeilen je Monat.
#
# ZUR VERRECHNUNGSSTEUER: keiner dieser ETFs loest sie. Ein Schweizer Fonds
# loest mit der Thesaurierung die Steuer nach Art. 4 Abs. 1 lit. c VStG
# genauso aus wie mit einer Ausschuettung. Und ein luxemburgischer Fonds
# traegt auf Schweizer Dividenden nach Verwahrstellenpraxis die vollen
# 35 Prozent ohne Abkommensermaessigung. Was der Fonds an Quellensteuer
# traegt, steckt bereits in seinem Kurs, der Backtest bildet es also
# automatisch ab.
#
# Die TER steckt ebenfalls im Marktkurs und wird deshalb NICHT zusaetzlich
# abgezogen. Sie ist hier nur zur Anzeige hinterlegt.
#
# Die thesaurierende UBS-SMI-Tranche (CH1447931341, SMIA) ist bewusst nicht
# aufgefuehrt: sie existiert erst seit Juni 2025 und reicht fuer keinen
# aussagekraeftigen Backtest.
def _kostenstruktur(cfg):
    """Loest den Eintrag kosten_titel eines Sleeves in Gewichte auf.

    "SMI" bedeutet: die Gebuehren werden so gerechnet, als wuerde der
    Aktienteil ueber die zwanzig SMI-Titel mit ihren Zielgewichten gehalten.
    Eine Zahl bedeutet gleichmaessige Aufteilung auf so viele Titel."""
    if not cfg:
        return None
    k = cfg.get("kosten_titel")
    if not k:
        return None
    if k == "SMI":
        return [min(v[1], 18.0) for v in SMI_CONSTITUENTS.values()]
    return k


def entnahme_wortlaut(pct_je_monat, alle_n_monate, kurz=False):
    """Entnahme in Worten, Satz und Termin sauber getrennt.

    pct_je_monat ist der Satz JE MONAT, alle_n_monate die Frequenz der
    Ausfuehrung. An jedem Termin wird pct mal n entnommen, der Jahresbetrag
    bleibt gleich. Beide Zahlen unkommentiert nebeneinander zu stellen
    ("monatliche Entnahme 0.250 %, quartalsweise Termine") liest sich wie
    ein Widerspruch, deshalb steht der Betrag je Termin vorne und der
    Monatssatz als Herleitung dahinter."""
    p = float(pct_je_monat) * 100.0
    m = max(int(alle_n_monate or 1), 1)
    if m == 1:
        return ("Entnahme %.3f %% je Monat" % p if kurz
                else "monatliche Entnahme von %.3f %%" % p)
    termin = {3: "je Quartal", 6: "je Halbjahr", 12: "je Jahr"}.get(
        m, "alle %d Monate" % m)
    if kurz:
        return "Entnahme %.3f %% %s (%.3f %%/Mt.)" % (p * m, termin, p)
    adj = {3: "quartalsweise", 6: "halbjährliche", 12: "jährliche"}.get(m)
    if not adj:
        return "Entnahme von %.3f %% alle %d Monate, das sind %.3f %% je Monat" % (
            p * m, m, p)
    return "%s Entnahme von %.3f %%, das sind %.3f %% je Monat" % (adj, p * m, p)


EQUITY_SLEEVES = {
    "Einzeltitel · 20 SMI-Titel (heutige Struktur)": None,
    "Ausschüttend · iShares SMI ETF (CSSMI)": {
        "ticker": "CSSMI.SW", "name": "iShares SMI® ETF (CH)",
        "isin": "CH0008899764", "ter": 0.0035, "domizil": "Schweiz",
        "ausschuettung": "ausschüttend, ad hoc", "seit": "1999",
        "thesaurierend": False, "index": "SMI",
    },
    "Ausschüttend · UBS SMI ETF (SMICHA)": {
        "ticker": "SMICHA.SW", "name": "UBS ETF (CH) SMI® (CHF) A-dis",
        "isin": "CH0017142719", "ter": 0.0020, "domizil": "Schweiz",
        "ausschuettung": "ausschüttend, jährlich", "seit": "2003",
        "thesaurierend": False, "index": "SMI",
    },
    "Thesaurierend · UBS SMI ETF (SMIA), Historie rekonstruiert": {
        "ticker": "SMIA.SW",
        "name": "UBS SMI\u00ae ETF CHF acc",
        "isin": "CH1447931341", "ter": 0.0020, "domizil": "Schweiz",
        "ausschuettung": "thesaurierend", "seit": "Juni 2025",
        "thesaurierend": True, "index": "SMI",
        # Eigene Historie erst ab Juni 2025. Die Reihe wird deshalb aus der
        # ausschuettenden Tranche desselben Fonds rekonstruiert.
        "synth_from": "SMICHA.SW",
        "synth_seit": "2003",
    },
    "Kontrolle · SMI-Kurse, Kosten wie 20 Einzeltitel": {
        "ticker": "CSSMI.SW", "name": "iShares SMI\u00ae ETF (CH)",
        "isin": "CH0008899764", "ter": 0.0035, "domizil": "Schweiz",
        "ausschuettung": "ausschüttend, ad hoc", "seit": "1999",
        "thesaurierend": False, "index": "SMI",
        # Keine eigene Anlagevariante, sondern die Kontrollrechnung: echte
        # Indexkurse ohne Verzerrung, aber mit der Kostenstruktur von
        # zwanzig Einzeltiteln. Zeigt, was die Zahl der gehandelten Titel
        # kostet, bei sonst gleicher Anlage.
        "kosten_titel": "SMI",   # Aufteilung nach den SMI-Zielgewichten
    },
    "Thesaurierend · UBS MSCI Switzerland 20/35 (SW2CHB)": {
        "ticker": "SW2CHB.SW",
        "name": "UBS MSCI Switzerland 20/35 UCITS ETF CHF acc",
        "isin": "LU0977261329", "ter": 0.0020, "domizil": "Luxemburg",
        "ausschuettung": "thesaurierend", "seit": "Okt. 2013",
        "thesaurierend": True, "index": "MSCI Switzerland 20/35",
    },
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
    _default_start = st.session_state.get("scenario_start", date(2015, 1, 1))
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

    st.markdown("### Kapitalfluss")
    st.markdown(
        f"<p style='color:{OAK_SAGE_DIM}; font-size:11px; margin-top:-6px;'>"
        "Laufende Zeichnungen und Rücknahmen am Monatsultimo. Der grösste "
        "Kostenunterschied zwischen Einzeltiteln und ETF liegt genau hier.</p>",
        unsafe_allow_html=True)
    flow_pct = st.slider(
        "Netto-Kapitalfluss je Monat (% des NAV)", min_value=-3.0,
        max_value=5.0, value=0.0, step=0.1,
        help="Positiv sind Zeichnungen, negativ Rücknahmen. Zuflüsse gehen "
             "nach bestehenden Anteilen in Aktien und Bitcoin, der Aktienteil "
             "nach Zielgewichten auf die Titel. Abflüsse kommen zuerst aus der "
             "Kasse, dann anteilig aus den Positionen.\n\n"
             "Jede Zeichnung kostet bei zwanzig Einzeltiteln 21 Orderzeilen "
             "und bei einem ETF zwei, unabhängig vom Betrag.") / 100.0
    flow_chf = st.number_input(
        "zusätzlich fester Betrag je Monat (CHF)", min_value=-5_000_000.0,
        max_value=5_000_000.0, value=0.0, step=10_000.0,
        help="Wird zum prozentualen Fluss addiert. Für ein Produkt, das mit "
             "festen Beträgen wächst statt proportional.")
    if flow_pct or flow_chf:
        st.caption(
            "Mit Kapitalflüssen ist der Endwert keine Rendite mehr, er enthält "
            "das eingezahlte Geld. Vergleichbar ist nur der **Anteilswert**, "
            "der unten ausgewiesen wird.")

    st.markdown("### Aktien-Sleeve")
    equity_sleeve = st.radio(
        "Aufbau des Aktienteils", list(EQUITY_SLEEVES.keys()), index=0,
        help="Zwanzig Einzeltitel bilden den SMI nach und erzeugen bei jedem "
             "Vorgang zwanzig Orderzeilen, jede mit eigener Mindestgebuehr. "
             "Ein SMI-ETF erzeugt eine einzige Orderzeile. Die Dividenden "
             "fliessen in beiden Faellen gleich, die Verrechnungssteuer von "
             "35 Prozent faellt in beiden Faellen an. Der Unterschied liegt "
             "allein in den Transaktionskosten und im Aufwand.")
    _sleeve_cfg = EQUITY_SLEEVES[equity_sleeve]
    if _sleeve_cfg:
        st.caption(
            f"{_sleeve_cfg['name']} · {_sleeve_cfg['isin']} · Index "
            f"{_sleeve_cfg['index']} · TER {_sleeve_cfg['ter']*100:.2f}% · "
            f"{_sleeve_cfg['ausschuettung']} · Domizil "
            f"{_sleeve_cfg['domizil']} · eigene Kurshistorie ab "
            f"{_sleeve_cfg['seit']}. TER und Quellensteuer des Fonds stecken "
            "bereits im Marktkurs und werden nicht zusätzlich abgezogen.")
        if _sleeve_cfg.get("kosten_titel"):
            st.warning(
                f"Kontrollrechnung, keine Anlagevariante. Gerechnet wird auf "
                f"den echten Kursen des {_sleeve_cfg['name']}, also ohne "
                "Verzerrung durch die heutige Indexzusammensetzung, aber "
                "belastet wie zwanzig Einzeltitel: jede Aktien-Orderzeile "
                "wird nach den SMI-Zielgewichten aufgeteilt, jeder Teil mit "
                "eigener Mindestgebühr, Teile unter der Bagatellgrenze "
                "entfallen. Das zeigt, was die Zahl der gehandelten Titel "
                "kostet, bei sonst gleicher Anlage.")
        if _sleeve_cfg.get("synth_from"):
            st.info(
                f"Dieser Anteilsklasse fehlt die Historie (erst ab "
                f"{_sleeve_cfg['seit']}). Der Backtest rekonstruiert sie aus "
                f"der ausschüttenden Tranche desselben Fonds "
                f"({_sleeve_cfg['synth_from']}, ab "
                f"{_sleeve_cfg['synth_seit']}), indem die Ausschüttungen netto "
                f"nach {int(WITHHOLDING_TAX*100)}% Verrechnungssteuer "
                "wiederangelegt werden. Die Steuerbelastung bleibt damit "
                "erhalten. Die Rekonstruktion wird unten gegen die echten "
                "Kurse geprüft, soweit sie vorliegen.")
    # Gewichtung und Rebalancing betreffen ausschliesslich die
    # Einzeltitelvariante. Haelt der Aktienteil einen ETF, steckt beides im
    # Instrument: der Fonds bildet den Index physisch nach, samt dessen
    # eigener Kappung, und gewichtet laufend selbst. Die beiden Regler
    # blieben dann wirkungslos, deshalb werden sie ausgeblendet statt
    # scheinbar bedienbar stehenzulassen.
    if _sleeve_cfg:
        weighting_method = "Marktkapitalisierung (Approx. + 18% Cap)"
        rebalance_freq = "Jährlich"
        _kappung = ("35% für den grössten und 20% für die übrigen Titel"
                    if _sleeve_cfg["index"].startswith("MSCI")
                    else "18% je Titel")
        st.caption(
            f"Gewichtung und Rebalancing entfallen: der Fonds bildet den "
            f"{_sleeve_cfg['index']} physisch nach und setzt dessen Kappung "
            f"({_kappung}) selbst durch. Die Einzeltitelvariante im "
            "Strukturvergleich rechnet weiterhin mit Marktkapitalisierung "
            "und jährlichem Rebalancing.")
    else:
        weighting_method = st.radio("SMI Gewichtung",
            ["Marktkapitalisierung (Approx. + 18% Cap)",
             "Equal Weight (5 % je Titel)"])
        rebalance_freq = st.selectbox("SMI Rebalancing-Frequenz",
            ["Jährlich", "Halbjährlich", "Quartalsweise", "Keine"], index=0,
            help="Final kalibriert auf Jährlich (September), an den echten "
                 "SIX-Indexreviewtermin angelehnt. Siehe Handelsreglement §7.")

    st.markdown("### Finanzierung des Bitcointeils")
    _thes = bool(_sleeve_cfg and _sleeve_cfg.get("thesaurierend"))
    _finanz_opt = ["Dividendenernte über DCA-Fenster (heute)",
                   "Monatliche Entnahme aus dem Aktienteil"]
    harvest_choice = st.radio(
        "Woher kommt das Geld für Bitcoin?", _finanz_opt,
        index=(1 if _thes else 0),
        help="Dividendenernte: die Ausschüttung wird vereinnahmt und über das "
             "DCA-Fenster in Tranchen in Bitcoin investiert. Das ist die "
             "heutige Mechanik und braucht ein Tranchenregister.\n\n"
             "Monatliche Entnahme: am Monatsultimo wird ein fester Prozentsatz "
             "der Aktienposition abverkauft und der Erlös unmittelbar in "
             "Bitcoin investiert. Wasserfall: erst Verkauf, dann Reinvestition. "
             "Kein Tranchenregister, kein Zustand über den Termin hinaus, zwei "
             "Orderzeilen je Monat.")
    harvest_mode = ("withdrawal" if harvest_choice.startswith("Monatliche")
                    else "dividend")

    if harvest_mode == "withdrawal":
        withdrawal_pct = st.number_input(
            "Entnahmesatz je Monat (%)", min_value=0.0, max_value=2.0,
            value=0.25, step=0.01, format="%.3f",
            help="Der Satz wird an jedem Termin neu auf den dann geltenden "
                 "Wert des Aktienteils angewendet. Über zwölf Termine "
                 "entzieht er dem Bestand rund 2.96 Prozent der Anteile "
                 "(1 minus 0.9975 hoch 12), unabhängig vom Kursverlauf; der "
                 "Frankenbetrag schwankt dagegen mit dem Kurs.\n\n"
                 "Zum Vergleich: ein thesaurierender SMI-ETF behält netto "
                 "rund 1.75 Prozent im Jahr ein (3 Prozent brutto, abzüglich "
                 "35 Prozent Verrechnungssteuer und 0.20 Prozent "
                 "Pauschalkommission). Der gleichwertige Satz wäre 0.147 "
                 "Prozent je Termin. Alles darüber verschiebt Substanz statt "
                 "nur Ertrag in Bitcoin.") / 100.0
        _freq_opt = {"Monatlich": 1, "Quartalsweise": 3,
                     "Halbjährlich": 6, "Jährlich": 12}
        _freq_wahl = st.selectbox(
            "Entnahmetermine", list(_freq_opt.keys()), index=0,
            help="Der Entnahmesatz oben gilt je Monat und bleibt derselbe. "
                 "Diese Auswahl bestimmt nur die Ausführungstage: an jedem "
                 "Termin wird der Satz mal die Zahl der Monate entnommen. Der "
                 "Jahresbetrag ändert sich nicht.\n\n"
                 "Vorgabe ist monatlich, weil das Produkt monatlich Bitcoin "
                 "kauft.\n\n"
                 "Seltenere Termine sparen nur dann Gebühren, wenn an den "
                 "übrigen Monatsenden ohnehin nichts gehandelt wird: ohne "
                 "Zeichnungen kostet monatlich 24 Mindestgebühren im Jahr, "
                 "quartalsweise 8. Sobald monatlich gezeichnet wird, wird der "
                 "Monatsletzte ohnehin gehandelt, und das Netting legt die "
                 "Entnahme in dieselbe Orderzeile. Dann kostet monatlich "
                 "gleich viel wie quartalsweise, bei feinerer Streuung der "
                 "Einstiegspunkte in Bitcoin.")
        withdrawal_n = _freq_opt[_freq_wahl]
        # Entnahme, nicht Wachstum: die Basis verkleinert sich mit jedem
        # Termin. (1+p)^12-1 waere die Formel fuer Wachstum und zu hoch.
        _jahr = 1 - (1 - withdrawal_pct)**12
        _zeilen_jahr = int(round(24 / withdrawal_n))
        st.caption(
            f"{entnahme_wortlaut(withdrawal_pct, withdrawal_n)}. Entzieht "
            f"dem Aktienteil über zwölf Monate rund {_jahr*100:.2f}% der "
            "Anteile, unabhängig vom Kursverlauf; der Frankenbetrag schwankt "
            "mit dem Kurs. Gleichwertiger Satz einer reinen Ertragsernte "
            f"rund 0.147% je Termin. Diese Frequenz erzeugt rund {_zeilen_jahr} eigene "
            "Orderzeilen im Jahr; wird am selben Tag ohnehin gezeichnet, "
            "fällt die Entnahme durch das Netting mit in dieselbe Zeile. Was "
            "das an Gebühren kostet, steht unten unter Kosten & Gebühren.")
    else:
        withdrawal_pct = 0.0
        # Vorgabe fuer den Strukturvergleich: wird der Aktienteil ueber
        # Dividenden finanziert, gibt es hier keine Entnahme. Der
        # Strukturvergleich rechnet die thesaurierende Variante aber trotzdem
        # mit und braucht dafuer sinnvolle Werte. Monatlich, weil das Produkt
        # monatlich Bitcoin kauft; seltenere Termine sparen nur dann etwas,
        # wenn an den uebrigen Monatsenden ohnehin nichts gehandelt wird.
        withdrawal_n = 1

    dca_months = st.slider("DCA-Zeitraum (Monate)", 1, 24, 6,
                           disabled=(harvest_mode == "withdrawal"),
                           help="Nur für die Dividendenernte. Final kalibriert "
                                "auf 6 Monate (Sharpe-/Calmar-Grid über "
                                "3/6/9/12 Monate).")

    if _thes and harvest_mode == "dividend":
        st.error(
            "Ein thesaurierender ETF schüttet nichts aus. Mit der "
            "Dividendenernte bekommt der Bitcointeil dann gar kein Geld. "
            "Bitte die monatliche Entnahme wählen.")
    if (not _thes) and harvest_mode == "withdrawal" and _sleeve_cfg is None:
        st.info(
            "Einzeltitel mit monatlicher Entnahme: die Dividenden fliessen "
            "weiter und werden beim Entnahmetermin vorrangig verwendet, bevor "
            "Anteile verkauft werden.")
    st.markdown("### Bitcoin-Instrument")
    btc_source = st.radio("Bitcoin-Exposure",
        ["IB1T ETP (Spot − TER, volle Historie)",
         "BTC-USD Spot (ohne TER, Referenz)",
         "IBIT tatsächliche Kurse (ab Jan 2024)",
         "BTC-USD bis 2024, dann IBIT"],
        index=0,
        help="Das Produkt hält Bitcoin NICHT direkt, sondern über das iShares "
             "Bitcoin ETP (IB1T, physisch besichert, Schweizer Domizil, "
             "USD-denominiert, Handel u.a. Xetra in EUR). Standard modelliert "
             "dieses Instrument über die volle Spot-Historie abzüglich der "
             "laufenden TER — das erhält 11 Jahre Kalibrierungstiefe UND "
             "bildet die Kostenbelastung korrekt ab. Die tatsächlichen "
             "Instrumentenkurse existieren erst ab 2024 (IBIT) bzw. 2025 "
             "(IB1T) und reichen für kein einziges vollständiges "
             "3-Jahres-Fenster.")
    etp_ter_pct = st.slider("ETP-TER (% p.a.)", min_value=0.0, max_value=1.0,
                            value=0.25, step=0.05,
                            help="Laufende Gebühr des Bitcoin-ETP, täglich auf die "
                                 "Bitcoin-Position abgegrenzt. IB1T: 0.15% während "
                                 "des Waivers, danach 0.25%. Standard 0.25% = der "
                                 "dauerhafte Satz (konservativ).") / 100.0

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
    min_fee_chf = st.number_input(
        "Mindestgebühr je Orderzeile (CHF)", min_value=0.0, max_value=500.0,
        value=75.0, step=5.0,
        help="Bank Frick rechnet je gehandeltem TITEL ab, nicht je Vorgang. "
             "Bei zwanzig Einzeltiteln fallen pro Umschichtung zwanzig "
             "Mindestgebühren an, bei einem ETF eine einzige. Das ist der "
             "eigentliche Kostentreiber des Produkts. Auf 0 setzen, um das "
             "frühere, rein proportionale Kostenmodell zu reproduzieren.")
    fx_fee_bps = st.slider(
        "Devisengebühr Bitcoin-Leg (bps)", min_value=0.0, max_value=100.0,
        value=30.0, step=5.0,
        help="Das Bitcoin-ETP handelt in EUR, die Zelle rechnet in CHF. "
             "Jede Bitcoin-Bewegung trägt deshalb zusätzlich eine "
             "Devisengebühr. Handelsreglement 12.3.")
    min_order_chf = st.number_input(
        "Bagatellgrenze je Orderzeile (CHF)", min_value=0.0, max_value=100_000.0,
        value=500.0, step=100.0,
        help="Orderzeilen unterhalb dieses Betrags werden in der Praxis nicht "
             "plaziert und hier deshalb auch nicht mit einer Mindestgebühr "
             "belastet. Ohne diese Grenze würde eine Anpassung über CHF 20 "
             "mit CHF 75 belastet, was die Einzeltitelvariante unfair "
             "schlechter rechnet als sie ist.")
    _min_notional = (min_fee_chf / (tx_cost_bps/10000.0)) if tx_cost_bps > 0 else 0.0
    if min_fee_chf > 0 and _min_notional > 0:
        st.caption(
            f"Die Mindestgebühr greift bis zu einem Ordervolumen von CHF "
            f"{_min_notional:,.0f} je Titel. Darunter kostet jede Zeile "
            f"pauschal CHF {min_fee_chf:,.0f}.".replace(",", "'"))
    if harvest_mode == "withdrawal" and min_fee_chf > 0:
        _zj = int(round(24 / max(1, withdrawal_n)))
        st.caption(
            f"Die gewählten Entnahmetermine erzeugen rund {_zj} Orderzeilen "
            f"im Jahr, also etwa CHF {_zj*min_fee_chf:,.0f} Mindestgebühren "
            f"pro Jahr, bevor irgendetwas anderes gehandelt wird."
            .replace(",", "'"))

    netting_on = st.checkbox(
        "Orderzeilen je Ausführungstag netten", value=True,
        help="Alle Bewegungen eines Ausführungstages werden je Instrument zu "
             "einer Order zusammengefasst und die Gebühr auf dem Saldo "
             "berechnet. Das entspricht Handelsreglement 9.4 und der Praxis: "
             "an einem Monatsultimo mit Zeichnung und Entnahme wird der ETF "
             "einmal gehandelt, nicht zweimal.\n\n"
             "Ausschalten zeigt, was jede Teilbewegung einzeln kosten würde. "
             "Das überzeichnet die Kosten, macht aber sichtbar, wie viel das "
             "Netting wert ist.")

    use_tiered_fee = st.checkbox("Gestaffelte Management Fee (volumenabhängig)",
                                 value=True,
                                 help="Festgelegte Gebührenstruktur: 2.00% p.a. als "
                                      "Basissatz, ab CHF 15 Mio. verwaltetem Vermögen "
                                      "1.75%, ab CHF 25 Mio. 1.50%. Der Satz wird "
                                      "täglich anhand des aktuellen NAV bestimmt. "
                                      "Ausschalten, um einen fixen Satz zu testen.")
    if use_tiered_fee:
        _t1 = st.number_input("Basissatz (% p.a.)", 0.0, 5.0, 2.00, 0.05) / 100.0
        _t2 = st.number_input("Satz ab Schwelle 1 (% p.a.)", 0.0, 5.0, 1.75, 0.05) / 100.0
        _s2 = st.number_input("Schwelle 1 (Mio.)", 0.0, 500.0, 15.0, 1.0) * 1e6
        _t3 = st.number_input("Satz ab Schwelle 2 (% p.a.)", 0.0, 5.0, 1.50, 0.05) / 100.0
        _s3 = st.number_input("Schwelle 2 (Mio.)", 0.0, 500.0, 25.0, 1.0) * 1e6
        mgmt_fee_pct = [(0.0, _t1), (_s2, _t2), (_s3, _t3)]
        mgmt_fee_display = _t1
        st.caption(f"Aktive Staffel: {_t1*100:.2f}% → {_t2*100:.2f}% ab "
                   f"{_s2/1e6:.0f} Mio. → {_t3*100:.2f}% ab {_s3/1e6:.0f} Mio.")
    else:
        mgmt_fee_pct = st.slider("Management Fee (% p.a.)", min_value=0.0, max_value=3.0,
                                 value=2.0, step=0.05,
                                 help="Daily accrual, deducted from NAV.") / 100.0
        mgmt_fee_display = mgmt_fee_pct
    if isinstance(mgmt_fee_pct, list):
        _fee_label = " / ".join(
            (f"{r*100:.2f}%" if th <= 0 else f"{r*100:.2f}% ab {th/1e6:.0f} Mio.")
            for th, r in mgmt_fee_pct)
    else:
        _fee_label = f"{mgmt_fee_pct*100:.2f}% p.a."
    perf_fee_pct = st.slider("Performance Fee (%)", min_value=0.0, max_value=30.0,
                             value=0.0, step=1.0,
                             help="Festgelegt auf 0%: das Produkt erhebt ausschliesslich "
                                  "eine Management Fee, keine erfolgsabhängige "
                                  "Komponente.") / 100.0
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
                                         ["Monthly", "Quarterly", "Semi-Annual", "Annual"], index=0,
                                         help="How often the performance fee is crystallized against the HWM. "
                                              "Default Monthly — matches the provider that settles both "
                                              "management and performance fees on a monthly basis.")
    mgmt_fee_freq = st.selectbox("Management Fee Billing Frequency",
                                 ["Monthly", "Quarterly", "Semi-Annual", "Annual"], index=0,
                                 help="How often the management fee is billed/settled — independent "
                                      "of the performance fee crystallization above (some providers "
                                      "bill both monthly, others differ). Does NOT change the daily "
                                      "NAV accrual, which always matches the stated annual rate exactly "
                                      "— this only controls how the already-accrued amounts are "
                                      "grouped into a billing ledger for reporting.")

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Backtest starten", type="primary", use_container_width=True,
                        disabled=(target_btc_pct >= upper_threshold))

    if st.button("🔄 Cache leeren (bei veralteten Daten/nach Code-Update)",
                use_container_width=True,
                help="Leert @st.cache_data — nötig, wenn nach einem Code-Update "
                     "(z.B. Split-Bereinigung) noch alte Kurse angezeigt werden. "
                     "Streamlits Cache erkennt Änderungen an Hilfsfunktionen nicht "
                     "immer zuverlässig. Alternative: 'Manage app' (unten rechts) "
                     "→ 'Reboot app', das räumt zusätzlich den ganzen Prozess auf."):
        st.cache_data.clear()
        st.session_state.pop("smi_has_run", None)
        st.success("Cache geleert. Bitte Backtest neu starten.")
        st.rerun()

    # Make the backtest "sticky": once run, keep showing results across reruns
    # (e.g. when the user clicks the PDF button) instead of clearing the page.
    if run_btn:
        st.session_state["smi_has_run"] = True
    _show_results = run_btn or st.session_state.get("smi_has_run", False)

    _tcf_footer = {"Monatlich (Standard)": "monthly", "Quartalsweise": "quarterly",
                   "Halbjährlich": "semi-annual"}.get(threshold_check_freq, "monthly")

    st.markdown(
        f"<div style='font-size:10px; color:{OAK_SAGE_DIM}; text-transform:uppercase; "
        f"letter-spacing:0.12em; padding-top:24px; margin-top:24px; "
        f"border-top:1px solid {OAK_BORDER};'>"
        "Data Source: Yahoo Finance · Raw Close (split-adjusted only)<br>"
        f"FX: USDCHF Spot · Threshold checks: {_tcf_footer}"
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
def _get_actions(ticker_symbol):
    """Fetch dividends AND stock splits for a ticker in a SINGLE network
    request (tk.history(actions=True) returns both in one response), instead
    of two separate yf.Ticker calls. On a cold cache (after a reboot / cache
    clear) this halves the per-ticker round-trips to Yahoo Finance across
    20 SMI names — previously up to 40 sequential calls (dividends + splits
    each separately), now up to 20. This was a major contributor to slow
    cold-start page loads, worsened by frequent reboots during today's
    calibration/debugging session.

    Returns (dividends_series, splits_series), each possibly empty. Both are
    consumed by _get_dividend_series() and _get_split_series() below, which
    stay as thin, separately-cached wrappers so no other call site needs to
    change.
    """
    tk = yf.Ticker(ticker_symbol)
    divs, splits = pd.Series(dtype=float), pd.Series(dtype=float)
    try:
        hist = tk.history(period="max", actions=True, auto_adjust=False)
        if hist is not None and not hist.empty:
            if "Dividends" in hist.columns:
                d = hist["Dividends"]
                divs = d[d > 0]
            if "Stock Splits" in hist.columns:
                s = hist["Stock Splits"]
                splits = s[s > 0]
    except Exception:
        pass
    # Fallbacks ONLY for whichever piece came back empty — avoids a second
    # network round-trip when the combined call already succeeded for both.
    if divs.empty:
        try:
            d = tk.dividends
            if d is not None and not d.empty:
                divs = d
        except Exception:
            pass
    if splits.empty:
        try:
            s = tk.splits
            if s is not None and not s.empty:
                splits = s
        except Exception:
            pass
    return divs, splits


@st.cache_data(ttl=21600, show_spinner=False)
def _get_split_series(ticker_symbol):
    """Stock-split events (ratio per split date). Thin wrapper around the
    shared _get_actions() fetch \u2014 see its docstring."""
    _, splits = _get_actions(ticker_symbol)
    return splits


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
    """Dividend history. Thin wrapper around the shared _get_actions() fetch
    (see its docstring) \u2014 falls back to the older get_dividends() API only
    if that combined call came back empty."""
    divs, _ = _get_actions(ticker_symbol)
    if divs is not None and not divs.empty:
        return divs
    try:
        tk = yf.Ticker(ticker_symbol)
        d = tk.get_dividends()
        if d is not None and not d.empty:
            return d
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


def synthesize_accumulating(price_series, dividends_df, ticker_src,
                            net_factor=DIVIDEND_NET_FACTOR):
    """Baut aus einer ausschuettenden Anteilsklasse die thesaurierende.

    WARUM DAS NOETIG IST: der thesaurierende UBS SMI ETF (CH1447931341, SMIA)
    existiert erst seit Juni 2025. Fuer eine Kalibrierung ueber rollierende
    Dreijahresfenster reicht das nicht annaehernd. Die ausschuettende Tranche
    desselben Fonds (CH0017142719, SMICHA) gibt es seit 2003.

    Beide Anteilsklassen bilden denselben Index mit derselben TER ab und
    unterscheiden sich nur in der Ertragsverwendung. Die thesaurierende Reihe
    ist deshalb die ausschuettende mit wiederangelegten Ausschuettungen:

        Kurs_thes(t) = Kurs_aus(t) * PROD (1 + netto_i / Kurs_aus(ex_i))

    Wiederangelegt wird der NETTObetrag nach 35 Prozent Verrechnungssteuer,
    waehrend der Kurs am Ex-Tag um den BRUTTObetrag faellt. Die Differenz ist
    genau die Steuerbelastung, die die Zelle nicht zurueckfordern kann, und
    sie bleibt damit in der Reihe erhalten. Eine thesaurierende Schweizer
    Anteilsklasse traegt dieselbe Last: die Thesaurierung loest die
    Verrechnungssteuer nach Art. 4 Abs. 1 lit. c VStG genauso aus wie eine
    Ausschuettung.

    NICHT abgebildet: die Tracking Difference zwischen den beiden
    Anteilsklassen und der exakte Wiederanlagekurs innerhalb des Ex-Tages.
    Beides ist zweiter Ordnung. Die Oberflaeche validiert die Rekonstruktion
    gegen die echten SMIA-Kurse, soweit sie vorliegen.
    """
    s = _clean_index(_to_series(price_series).dropna())
    if s.empty:
        return s
    ausschuettungen = {}
    if dividends_df is not None and not dividends_df.empty:
        for _, r in dividends_df.iterrows():
            if r["ticker"] != ticker_src:
                continue
            k = _norm_ts(r["date"])
            ausschuettungen[k] = ausschuettungen.get(k, 0.0) + float(
                r["dividend_per_share"])
    faktor = 1.0
    werte = []
    for d, p in s.items():
        betrag = ausschuettungen.get(d)
        if betrag and p > 0:
            faktor *= (1.0 + betrag * net_factor / float(p))
        werte.append(float(p) * faktor)
    return pd.Series(werte, index=s.index)


def apply_etp_ter(spot_series, ter_annual):
    """Model a physically-backed Bitcoin ETP (IB1T / IBIT) from a spot series.

    WHY THIS EXISTS: the product does NOT hold Bitcoin directly — it holds the
    iShares Bitcoin ETP (IB1T: physically backed, Swiss-domiciled, USD-
    denominated, traded on Xetra/Euronext/LSE). An ETP's Bitcoin entitlement
    per share DECLINES over time because the sponsor sells Bitcoin to pay the
    fee, so the ETP price systematically lags spot by the accrued TER.

    Using the ACTUAL instrument's price history is not viable for calibration:
    IBIT starts 2024-01-11 and IB1T 2025-03-25 — neither covers a single full
    3-year rolling window of the 2015-2026 calibration period. Splicing spot
    onto instrument data (the legacy "BTC-USD bis 2024, dann IBIT" option)
    silently applies ZERO cost for the pre-2024 stretch, which understates the
    true drag over most of the backtest.

    So instead: keep the full spot history and accrue the TER synthetically.

        entitlement(t) = entitlement(0) * (1 - ter)^(years elapsed)
        etp_price(t)   = spot(t) * entitlement(t)

    This is exactly how the real instrument behaves, is conservative (assumes
    the full standing TER, not the temporary waiver), and preserves 11 years
    of calibration depth. Tracking difference, premium/discount and the small
    gap between the CME CF Reference Rate and the spot exchange print are NOT
    modelled — they are second-order and disclosed as a model assumption.
    """
    if spot_series is None or spot_series.empty or ter_annual <= 0:
        return spot_series
    idx = spot_series.index
    years_elapsed = (idx - idx[0]).days / 365.25
    drag = (1.0 - ter_annual) ** years_elapsed
    return spot_series * pd.Series(drag, index=idx)


# ---------------------------------------------------------------------------
# Integrated Strategy Simulation
# ---------------------------------------------------------------------------
def get_execution_dates(idx, convention):
    """Ausfuehrungstermine fuer die DCA-Tranchen nach einer benannten Konvention.

    Dient dem INDIFFERENZ-TEST: nicht um den historisch besten Tag zu suchen
    (das waere Market-Timing und widerspraeche der prognosefreien Positionierung),
    sondern um zu pruefen, OB die Wahl des Tages ueberhaupt einen materiellen
    Unterschied macht. Ist der Unterschied zwischen den Konventionen klein
    gegenueber der Streuung zwischen den Zeitfenstern, ist die Wahl operativ
    frei und kann im Reglement mit Betriebsargumenten begruendet werden.

    Alle Konventionen liefern genau EINEN Termin je Kalendermonat und fallen
    auf einen tatsaechlichen Handelstag des uebergebenen Index.
    """
    df = pd.DataFrame(index=idx)
    df["ym"] = df.index.to_period("M")
    out = set()
    for _ym, sub in df.groupby("ym"):
        days = sub.index
        if len(days) == 0:
            continue
        if convention == "Monatsultimo":
            out.add(days[-1])
        elif convention == "Monatsanfang":
            out.add(days[0])
        elif convention == "Monatsmitte":
            # Handelstag am naechsten zum 15. des Monats
            target = pd.Timestamp(year=days[0].year, month=days[0].month, day=15)
            out.add(min(days, key=lambda d: abs((d - target).days)))
        elif convention == "Letzter Montag":
            mondays = [d for d in days if d.weekday() == 0]
            out.add(mondays[-1] if mondays else days[-1])
        elif convention == "Erster Montag":
            mondays = [d for d in days if d.weekday() == 0]
            out.add(mondays[0] if mondays else days[0])
        else:
            out.add(days[-1])
    return out


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
                 threshold_check_dates_set=None, cap_dates_set=None,
                 weight_cap=None, dca_execution_dates_set=None,
                 min_fee_chf=0.0, fx_fee_bps=0.0, min_order_chf=0.0,
                 harvest_mode="dividend", withdrawal_pct_monthly=0.0,
                 withdrawal_every_n_months=1,
                 monthly_flow_pct=0.0, monthly_flow_chf=0.0, netting=True,
                 cost_titles=None):
    """Integrated daily simulation.
    Returns: timeseries_df, transactions_df, threshold_events_df

    threshold_check_dates_set: dates on which the Bitcoin band (upper_threshold
    -> target_btc_pct) is evaluated. Defaults to EVERY month-end (None ->
    falls back to month_ends below) — the finalized, calibrated rule
    (Handelsreglement §6.1: monthly check; equity rebalancing is separately
    annual/September per §7). Pass a coarser date set (e.g. via
    get_rebalance_dates) to check
    less often — this only changes HOW OFTEN the band is evaluated, never
    whether DCA purchases happen (DCA always executes at every month-end,
    independent of this parameter).

    tx_cost_bps: transaction cost in basis points applied to the traded
    notional of EACH ORDER LINE (initial buy, DCA buys, threshold sells, and
    equity rebalancing). 10 bps = 0.10%.

    min_fee_chf: Mindestgebuehr je Orderzeile in CHF. Bank Frick rechnet je
    gehandeltem Titel ab, nicht je Vorgang. Bei zwanzig Einzeltiteln entstehen
    also zwanzig Mindestgebuehren je Umschichtung, bei einem ETF nur eine.
    Das ist der eigentliche Kostentreiber des Produkts und der Grund, weshalb
    dieser Parameter existiert. Vorgabe 0.0 = altes Verhalten.

    fx_fee_bps: Devisengebuehr in Basispunkten auf jede Bitcoin-Bewegung
    (das ETP handelt in EUR, die Zelle rechnet in CHF). Vorgabe 0.0.

    min_order_chf: Bagatellgrenze. Orderzeilen unterhalb dieses Betrags
    werden in der Praxis nicht plaziert und deshalb auch nicht mit einer
    Mindestgebuehr belastet. Die Zielgewichte werden trotzdem angesteuert,
    der Effekt auf den NAV ist vernachlaessigbar.

    harvest_mode: wie der Bitcointeil finanziert wird.
      "dividend"   heutige Mechanik: Dividende wird vereinnahmt und ueber
                   dca_months Monate in Tranchen in Bitcoin investiert.
      "withdrawal" thesaurierende Mechanik: am Monatsultimo wird
                   withdrawal_pct_monthly der Aktienposition abverkauft und
                   der Erloes unmittelbar in Bitcoin investiert. Kein
                   Tranchenregister, kein Zustand ueber den Termin hinaus.

    withdrawal_pct_monthly: Entnahmesatz je Monat, Anteil der Aktienposition.
    0.0025 entspricht 0.25 Prozent im Monat, also 3.04 Prozent im Jahr.
    Zum Vergleich: die heutige Nettodividendenernte betraegt rund 1.95
    Prozent im Jahr (3 Prozent brutto mal 0.65).

    withdrawal_every_n_months: Abstand der Entnahmetermine in Monaten.
    1 = jeden Monatsultimo, 3 = quartalsweise, 6 = halbjaehrlich. Der Satz
    bleibt derselbe und wird je Termin mit der Zahl der verstrichenen Monate
    multipliziert, der Jahresbetrag aendert sich also nicht. Was sich aendert,
    sind die Gebuehren (je Termin zwei Mindestgebuehren) und die Streuung der
    Einstiegspunkte in Bitcoin.

    Wasserfall im Entnahmemodus: erst Verkauf, dann Reinvestition. Beide
    Stufen tragen ihre eigene Gebuehr, die Bitcoinstufe zusaetzlich die
    Devisengebuehr. Vorhandene Kasse wird vor dem Abverkauf verwendet.

    monthly_flow_pct / monthly_flow_chf: monatlicher Netto-Kapitalfluss am
    Monatsultimo, als Anteil des NAV und als fester Betrag, beide werden
    addiert. Positiv sind Zeichnungen, negativ Ruecknahmen. Zufluesse gehen
    nach bestehenden Anteilen in Aktien und Bitcoin, der Aktienteil nach
    Zielgewichten auf die Titel (Handelsreglement 9.4 Schritt 1). Abfluesse
    kommen zuerst aus der Kasse, dann anteilig aus den Positionen
    (Handelsreglement 7.3).

    Mit Kapitalfluessen ist total_value kein Renditemass mehr, es enthaelt
    das eingezahlte Geld. Dafuer gibt es die Spalte nav_per_unit: den
    Anteilswert, der von Zeichnungen und Ruecknahmen unberuehrt bleibt.
    ts.attrs["net_flow"] haelt den kumulierten Nettofluss.

    cost_titles: rechnet die Gebuehren so, als bestuende der Aktienteil aus
    mehreren Titeln, unabhaengig davon, wie viele tatsaechlich gehalten
    werden. Entweder eine Zahl (gleichmaessige Aufteilung) oder eine Liste
    von Zielgewichten (Aufteilung nach diesen Gewichten, realistischer).
    Jede Aktien-Orderzeile wird entsprechend zerlegt, jeder Teil mit eigener
    Mindestgebuehr, Teile unterhalb der Bagatellgrenze entfallen.
    Positionen, Gewichte und Renditen bleiben unberuehrt, nur die Kosten
    aendern sich.

    Wozu: der direkte Vergleich zwanzig Einzeltitel gegen ETF vermischt die
    Kostenfrage mit einer Verzerrung, weil die Einzeltitelvariante mit den
    heutigen Indexmitgliedern zurueckrechnet. Mit cost_titles laesst sich die
    Aktienrendite konstant halten und allein die Kostenstruktur variieren.

    netting: fasst alle Bewegungen eines Ausfuehrungstages je Instrument zu
    einer Orderzeile zusammen und berechnet die Gebuehr auf dem Saldo. Das
    entspricht Handelsreglement 9.4 und der Praxis: an einem Monatsultimo mit
    Zeichnung und Entnahme wird der ETF einmal gehandelt, nicht zweimal.
    Ausgeschaltet zaehlt jede Teilbewegung einzeln, was die Kosten
    ueberzeichnet.
    """
    tx_cost = tx_cost_bps / 10000.0  # bps -> fraction
    fx_fee = fx_fee_bps / 10000.0
    min_fee = float(min_fee_chf or 0.0)
    min_order = float(min_order_chf or 0.0)
    total_tx_costs = 0.0  # accumulator in CHF

    # Kostenstatistik: Orderzeilen zaehlen, nicht nur Franken summieren.
    # Erst diese Zahlen machen den Unterschied zwischen zwanzig Einzeltiteln
    # und einem ETF sichtbar.
    cost_stats = {"lines": 0, "lines_at_min": 0, "fee_equity": 0.0,
                  "fee_btc": 0.0, "fee_fx": 0.0, "lines_skipped": 0}

    # Tagesbuch: Saldo je Instrument fuer den laufenden Ausfuehrungstag.
    tagesbuch = {}
    tagesbuch_fx = [0.0]

    # Kostenstruktur-Ueberlagerung: Anteile, nach denen eine Aktienbewegung
    # fuer die Gebuehrenrechnung aufgeteilt wird. Normalisiert auf 1.
    _kostenanteile = None
    if cost_titles:
        if isinstance(cost_titles, (list, tuple)) and len(cost_titles) > 1:
            _summe = float(sum(cost_titles))
            if _summe > 0:
                _kostenanteile = [float(x) / _summe for x in cost_titles]
        elif not isinstance(cost_titles, (list, tuple)) and int(cost_titles) > 1:
            _kostenanteile = [1.0 / int(cost_titles)] * int(cost_titles)

    def _zeilengebuehr(v, leg):
        """Gebuehr und Zahl der Orderzeilen fuer ein Handelsvolumen v.

        Normalfall: eine Zeile. Ist die Kostenstruktur-Ueberlagerung gesetzt
        und betrifft die Bewegung den Aktienteil, wird sie nach den
        hinterlegten Anteilen aufgeteilt, jeder Teil mit eigener
        Mindestgebuehr.

        Aufgeteilt wird nach den ZIELGEWICHTEN, nicht gleichmaessig. Das ist
        entscheidend: real bekommt ein Titel mit einem Prozent Zielgewicht
        auch nur ein Prozent des Betrags, und faellt damit oft unter die
        Bagatellgrenze. Eine gleichmaessige Aufteilung wuerde die Kosten der
        Einzeltitelvariante deutlich ueberzeichnen."""
        if leg == "btc" or not _kostenanteile:
            prop = v * tx_cost
            f = max(prop, min_fee)
            return f, 1, (1 if prop < min_fee else 0)
        gesamt = 0.0
        zeilen = 0
        zeilen_min = 0
        for anteil in _kostenanteile:
            je = v * anteil
            if je < min_order or je <= 0.005:
                continue
            prop = je * tx_cost
            f = max(prop, min_fee)
            gesamt += f
            zeilen += 1
            if prop < min_fee:
                zeilen_min += 1
        if zeilen == 0:
            # Alle Teile unter der Bagatellgrenze: in der Praxis wuerde der
            # Betrag dann auf einen Titel gelegt statt gar nicht gehandelt.
            prop = v * tx_cost
            f = max(prop, min_fee)
            return f, 1, (1 if prop < min_fee else 0)
        return gesamt, zeilen, zeilen_min

    def _fee(notional, leg="equity", key=None):
        """Gebuehr einer einzelnen Orderzeile, inklusive Mindestgebuehr.

        key benennt das Instrument (Ticker oder __BTC__). Ohne Netting wird er
        nicht gebraucht, mit Netting bestimmt er, welche Bewegungen eines
        Ausfuehrungstages zu einer Orderzeile zusammengefasst werden."""
        v = abs(float(notional))
        if v < min_order or v <= 0.005:
            if v > 0.005:
                cost_stats["lines_skipped"] += 1
            return 0.0
        if netting:
            # Nur buchen. Die Gebuehr faellt am Tagesende auf dem Saldo an.
            k = key if key is not None else ("__BTC__" if leg == "btc" else "__EQ__")
            eintrag = tagesbuch.setdefault(k, [0.0, leg])
            eintrag[0] += float(notional)
            return 0.0
        f, zeilen, zeilen_min = _zeilengebuehr(v, leg)
        cost_stats["lines"] += zeilen
        cost_stats["lines_at_min"] += zeilen_min
        cost_stats["fee_" + ("btc" if leg == "btc" else "equity")] += f
        return f

    def _fee_probe(notional, leg="equity"):
        """Gebuehr einer Orderzeile berechnen, ohne sie zu zaehlen oder zu
        buchen. Fuer Vorabpruefungen, ob sich eine Ausfuehrung ueberhaupt
        lohnt. Beruecksichtigt cost_titles, sonst wuerden die Schutzregeln
        bei aufgeteilten Zeilen zu niedrig ansetzen."""
        v = abs(float(notional))
        if v < min_order or v <= 0.005:
            return 0.0
        return _zeilengebuehr(v, leg)[0]

    def _fx(notional):
        """Devisengebuehr auf eine Bitcoin-Bewegung."""
        if netting:
            tagesbuch_fx[0] += float(notional)
            return 0.0
        f = abs(float(notional)) * fx_fee
        cost_stats["fee_fx"] += f
        return f

    def _tag_abrechnen(d, row, active_today, btc_px_chf):
        """Rechnet das Tagesbuch ab: je Instrument eine Orderzeile auf dem
        Saldo. Gibt die Gesamtgebuehr des Tages zurueck."""
        if not netting:
            return 0.0
        gesamt = 0.0
        for k, (saldo, leg) in tagesbuch.items():
            v = abs(saldo)
            if v < min_order or v <= 0.005:
                if v > 0.005:
                    cost_stats["lines_skipped"] += 1
                continue
            f, zeilen, zeilen_min = _zeilengebuehr(v, leg)
            cost_stats["lines"] += zeilen
            cost_stats["lines_at_min"] += zeilen_min
            cost_stats["fee_" + ("btc" if leg == "btc" else "equity")] += f
            gesamt += f
        if tagesbuch_fx[0]:
            f = abs(tagesbuch_fx[0]) * fx_fee
            cost_stats["fee_fx"] += f
            gesamt += f
        tagesbuch.clear()
        tagesbuch_fx[0] = 0.0
        return gesamt
    total_wht = 0.0       # gross dividend withheld at source (35%, non-reclaimable)
    # Normalize the index to tz-naive, midnight Timestamps so that comparisons
    # against the (cleaned) BTC/FX series and the dividend/rebalance keys stay
    # consistent under pandas 3.x.
    prices = _clean_index(prices.copy())
    rebalance_dates_set = {_norm_ts(x) for x in rebalance_dates_set}
    cap_dates_set = {_norm_ts(x) for x in (cap_dates_set or set())}
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
    # AUSFUEHRUNGSKONVENTION: standardmaessig der Monatsultimo (kalibrierte
    # Regel, Handelsreglement 5.1). Wird eine alternative Terminmenge
    # uebergeben, ersetzt sie den Monatsultimo als DCA-Ausfuehrungstag —
    # ausschliesslich fuer den Indifferenz-Test, der prueft, OB der gewaehlte
    # Tag ueberhaupt einen Unterschied macht. Die Schwellenpruefung bleibt
    # davon unberuehrt (eigener Parameter).
    if dca_execution_dates_set:
        month_ends = {_norm_ts(x) for x in dca_execution_dates_set}

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
        # Je Titel eine eigene Orderzeile mit eigener Mindestgebuehr.
        cost = sum(_fee(initial_capital * w_t0[t], key=t) for t in active_t0)
        cost = min(cost, max(initial_capital - 1.0, 0.0))
        total_tx_costs += cost
        investable = initial_capital - cost
        for t in active_t0:
            smi_shares[t] = (investable * w_t0[t]) / prices_clean.loc[first_day, t]
        btc_held = 0.0
        btc_u_init = 0.0
        btc_u_dca = 0.0
    else:
        # Cost charged on both equity and BTC legs of the initial allocation.
        # Aktienseite: eine Orderzeile je Titel. Bitcoinseite: eine Zeile plus
        # Devisengebuehr.
        cost_eq = sum(_fee(initial_smi_chf * w_t0[t], key=t) for t in active_t0)
        cost_btc = _fee(initial_btc_chf, leg="btc", key="__BTC__") + _fx(initial_btc_chf)
        # Schutz: die Gebuehr darf den jeweiligen Anlagebetrag nie erreichen.
        cost_eq = min(cost_eq, max(initial_smi_chf - 1.0, 0.0))
        cost_btc = min(cost_btc, max(initial_btc_chf - 1.0, 0.0))
        cost = cost_eq + cost_btc
        total_tx_costs += cost
        smi_invest = initial_smi_chf - cost_eq
        for t in active_t0:
            smi_shares[t] = (smi_invest * w_t0[t]) / prices_clean.loc[first_day, t]
        btc_invest = initial_btc_chf - cost_btc
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
    # Anteilsrechnung: der Startbestand entspricht initial_capital Anteilen zu
    # je CHF 1. Zeichnungen schaffen Anteile zum aktuellen Anteilswert,
    # Ruecknahmen loeschen sie. Der Anteilswert bleibt damit von
    # Kapitalfluessen unberuehrt.
    _anteile = float(initial_capital)
    _netto_fluss = 0.0       # kumulierter Nettokapitalfluss
    _me_zaehler = 0          # gezaehlte Monatsultimi, steuert die Entnahmetermine
    _entnahme_n = max(1, int(withdrawal_every_n_months or 1))

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
                    # Im Entnahmemodus entstehen keine DCA-Fenster. Eine
                    # Ausschuettung landet dort in der Kasse und wird beim
                    # naechsten Entnahmetermin vorrangig verwendet.
                    if harvest_mode != "withdrawal":
                        pending_dca.append({"remaining": dca_months,
                                            "monthly_chf": cash / dca_months,
                                            "ticker": t})

        # 1b. KAPITALFLUSS: Zeichnungen und Ruecknahmen am Monatsultimo.
        # Handelsreglement 9.4 Schritt 1 (Zufluss nach bestehenden Anteilen)
        # und 7.3 (Abfluss zuerst aus der Kasse, dann anteilig).
        if (d in month_ends and (monthly_flow_pct or monthly_flow_chf)
                and active_today):
            _sv = _smi_value_on(d, row)
            _bv = (btc_held * btc_price_d * fx_d
                   if (btc_price_d and fx_d) else 0.0)
            _nav_vor = _sv + _bv + dividend_cash
            _fluss = _nav_vor * monthly_flow_pct + monthly_flow_chf

            if _fluss > 0 and _nav_vor > 0:
                # Anteile zum Wert VOR dem Zufluss ausgeben
                _awert = _nav_vor / _anteile if _anteile > 0 else 0.0
                if _awert > 0:
                    _anteile += _fluss / _awert
                _netto_fluss += _fluss
                # Nach bestehenden Anteilen, die Kasse bleibt aussen vor:
                # sie ist ein Durchlaufposten und soll nicht anwachsen.
                _basis = _sv + _bv
                _aq = (_sv / _basis) if _basis > 0 else 1.0
                _bq = 1.0 - _aq
                _eq_teil = _fluss * _aq
                _btc_teil = _fluss - _eq_teil
                _geb_fluss = 0.0
                if _eq_teil > 0:
                    _wa = w[active_today] / w[active_today].sum()
                    for t in active_today:
                        _v = _eq_teil * float(_wa[t])
                        if _v > 0:
                            _geb_fluss += _fee(_v, key=t)
                            smi_shares[t] += _v / row[t]
                    att_equity_invested += _eq_teil
                if _btc_teil > 0 and btc_price_d and fx_d and fx_d > 0:
                    _geb_fluss += _fee(_btc_teil, leg="btc", key="__BTC__")
                    _geb_fluss += _fx(_btc_teil)
                    _zu = _btc_teil / (btc_price_d * fx_d)
                    btc_held += _zu
                    btc_u_init += _zu          # neues Kapital, nicht DCA
                    att_btc_init_invested += _btc_teil
                elif _btc_teil > 0:
                    # Ohne Bitcoinkurs bleibt der Anteil in der Kasse
                    dividend_cash += _btc_teil
                # Ohne Netting faellt die Gebuehr sofort an und wird den
                # Positionen belastet. Mit Netting gibt _fee null zurueck, die
                # Abrechnung erfolgt dann am Tagesende auf dem Saldo.
                if _geb_fluss > 0:
                    total_tx_costs += _geb_fluss
                    _sv2 = _smi_value_on(d, row)
                    _bv2 = (btc_held * btc_price_d * fx_d
                            if (btc_price_d and fx_d) else 0.0)
                    _b2 = _sv2 + _bv2
                    _g = min(_geb_fluss, max(_b2 - 1.0, 0.0))
                    if _b2 > 0 and _g > 0:
                        _ge = _g * (_sv2 / _b2)
                        _gb = _g - _ge
                        if _ge > 0 and _sv2 > 0:
                            _sh = (_sv2 - _ge) / _sv2
                            for t in available:
                                smi_shares[t] *= _sh
                        if _gb > 0 and btc_price_d and fx_d and btc_held > 0:
                            _w2 = min(_gb / (btc_price_d * fx_d), btc_held)
                            btc_held -= _w2
                            _t2 = btc_u_init + btc_u_dca
                            _f2 = (btc_u_init / _t2) if _t2 > 0 else 0.0
                            btc_u_init -= _w2 * _f2
                            btc_u_dca -= _w2 * (1 - _f2)
                transactions.append({
                    "date": d, "type": "BUY", "reason": "ZEICHNUNG",
                    "btc_amount": 0.0, "chf_amount": _fluss,
                    "usd_amount": 0.0,
                    "btc_price_usd": btc_price_d or 0.0, "usdchf": fx_d or 0.0,
                })

            elif _fluss < 0 and _nav_vor > 0:
                # Nie den ganzen NAV entnehmen: es muss Substanz bleiben,
                # aus der die Gebuehren des Tages getragen werden koennen.
                _ab = min(-_fluss, max(_nav_vor * 0.99, 0.0))
                _awert = _nav_vor / _anteile if _anteile > 0 else 0.0
                if _awert > 0:
                    _anteile = max(_anteile - _ab / _awert, 0.0)
                _netto_fluss -= _ab
                # ANTEILIG ueber alle drei Toepfe, nicht Kasse zuerst.
                #
                # Handelsreglement 7.3 nennt die Cash-Position als erste
                # Quelle fuer Abfluesse. Dieses Modell kennt aber keine
                # eigene Cash-Position: dividend_cash IST der Ernte-Pool,
                # also die vereinnahmten Dividenden, auf die die offenen
                # DCA-Tranchen ausgestellt sind. Wuerde eine Ruecknahme
                # zuerst dort zugreifen, loeschte ein laufender Abfluss das
                # ganze DCA-Programm, und der Bitcointeil bekaeme nie wieder
                # Geld. Das ist weder gewollt noch richtig.
                #
                # Wirtschaftlich nimmt ein Ruecknehmer seinen Anteil an
                # ALLEM mit, auch an der noch nicht angelegten Ernte. Genau
                # das bildet die anteilige Entnahme ab, und die offenen
                # Tranchen schrumpfen dabei um exakt seinen Anteil.
                _quote = _ab / _nav_vor if _nav_vor > 0 else 0.0
                _kasse_vor = max(dividend_cash, 0.0)
                _aus_kasse = _kasse_vor * _quote
                if _aus_kasse > 0:
                    for _e in pending_dca:
                        _e["monthly_chf"] *= (1.0 - _quote)
                dividend_cash -= _aus_kasse
                _rest = _ab - _aus_kasse
                # Schritt 2: anteilig aus den Positionen
                _basis = _sv + _bv
                _geb_ab = 0.0
                if _rest > 0 and _basis > 0:
                    _eq_ab = _rest * (_sv / _basis)
                    _btc_ab = _rest - _eq_ab
                    if _eq_ab > 0 and _sv > 0:
                        for t in active_today:
                            _anteil = (smi_shares[t] * row[t]) / _sv
                            _v = _eq_ab * _anteil
                            if _v > 0:
                                _geb_ab += _fee(_v, key=t)
                                smi_shares[t] -= _v / row[t]
                        att_equity_invested -= _eq_ab
                    if _btc_ab > 0 and btc_price_d and fx_d and fx_d > 0:
                        _geb_ab += _fee(_btc_ab, leg="btc", key="__BTC__")
                        _geb_ab += _fx(_btc_ab)
                        _weg = _btc_ab / (btc_price_d * fx_d)
                        _weg = min(_weg, btc_held)
                        btc_held -= _weg
                        # Lots pro rata reduzieren, nie FIFO
                        _tu = btc_u_init + btc_u_dca
                        _fi = (btc_u_init / _tu) if _tu > 0 else 0.0
                        btc_u_init -= _weg * _fi
                        btc_u_dca -= _weg * (1 - _fi)
                        att_btc_init_invested -= _btc_ab * _fi
                        att_btc_dca_invested -= _btc_ab * (1 - _fi)
                if _geb_ab > 0:
                    total_tx_costs += _geb_ab
                    _sv3 = _smi_value_on(d, row)
                    _bv3 = (btc_held * btc_price_d * fx_d
                            if (btc_price_d and fx_d) else 0.0)
                    _b3 = _sv3 + _bv3
                    _g3 = min(_geb_ab, max(_b3 - 1.0, 0.0))
                    if _b3 > 0 and _g3 > 0:
                        _ge3 = _g3 * (_sv3 / _b3)
                        _gb3 = _g3 - _ge3
                        if _ge3 > 0 and _sv3 > 0:
                            _sh3 = (_sv3 - _ge3) / _sv3
                            for t in available:
                                smi_shares[t] *= _sh3
                        if _gb3 > 0 and btc_price_d and fx_d and btc_held > 0:
                            _w3 = min(_gb3 / (btc_price_d * fx_d), btc_held)
                            btc_held -= _w3
                            _t3 = btc_u_init + btc_u_dca
                            _f3 = (btc_u_init / _t3) if _t3 > 0 else 0.0
                            btc_u_init -= _w3 * _f3
                            btc_u_dca -= _w3 * (1 - _f3)
                if True:
                    transactions.append({
                        "date": d, "type": "SELL", "reason": "RUECKNAHME",
                        "btc_amount": 0.0, "chf_amount": -_ab,
                        "usd_amount": 0.0,
                        "btc_price_usd": btc_price_d or 0.0,
                        "usdchf": fx_d or 0.0,
                    })

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

            # Schutz: kostet die Ausfuehrung mehr als sie bewegt, wird in
            # diesem Monat nicht gekauft. Die Tranchen bleiben offen und
            # kommen im Folgemonat zusammen mit den dann faelligen erneut zur
            # Ausfuehrung, bis der Betrag die Gebuehr traegt. Das Geld bleibt
            # als Dividendenkasse Teil des NAV, es geht nichts verloren.
            # Sicherheitsnetz: nie mehr anlegen, als in der Kasse liegt.
            # Greift, wenn eine Ruecknahme die Kasse unter die offenen
            # Tranchen gedrueckt hat.
            _verfuegbar = max(dividend_cash, 0.0)
            if total_dca_chf > _verfuegbar:
                _skal = (_verfuegbar / total_dca_chf) if total_dca_chf > 0 else 0.0
                for entry in pending_dca:
                    entry["monthly_chf"] *= _skal
                total_dca_chf = _verfuegbar

            _probe = (_fee_probe(total_dca_chf) + abs(total_dca_chf) * fx_fee)
            if total_dca_chf > 0 and _probe >= total_dca_chf:
                total_dca_chf = 0.0

            if total_dca_chf > 0 and btc_price_d and fx_d and fx_d > 0:
                # Consume one tranche per entry ONLY now that the buy executes
                # (previously tranches were consumed even when BTC/FX quotes
                # were missing — that money silently vanished).
                for entry in pending_dca:
                    if entry["remaining"] > 0:
                        entry["remaining"] -= 1
                pending_dca = [e for e in pending_dca if e["remaining"] > 0]
                cost = _fee(total_dca_chf, leg="btc", key="__BTC__") + _fx(total_dca_chf)
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

        # 2b. ENTNAHME (Wasserfall). Ersetzt im Entnahmemodus die
        # Dividendenernte samt DCA-Fenstern: am Monatsultimo wird ein fester
        # Prozentsatz der Aktienposition abverkauft und der Erloes unmittelbar
        # in Bitcoin investiert. Erst Verkauf, dann Reinvestition, jede Stufe
        # mit eigener Gebuehr.
        if is_month_end:
            _me_zaehler += 1
        _ist_entnahmetermin = (is_month_end and _me_zaehler % _entnahme_n == 0)

        if (harvest_mode == "withdrawal" and _ist_entnahmetermin
                and withdrawal_pct_monthly > 0 and active_today
                and btc_price_d and fx_d and fx_d > 0):
            _eq_val = _smi_value_on(d, row)
            # Der Satz gilt je Monat, der Termin deckt _entnahme_n Monate ab.
            _ziel = _eq_val * withdrawal_pct_monthly * _entnahme_n
            # Vorhandene Kasse zuerst, Rest aus dem Abverkauf.
            _aus_kasse = min(max(dividend_cash, 0.0), _ziel)
            _aus_verkauf = max(_ziel - _aus_kasse, 0.0)

            # Verkaufszeilen je Titel, anteilig am aktuellen Bestand. Bei
            # einem ETF ist das genau eine Zeile.
            _zeilen = {}
            if _aus_verkauf > 0 and _eq_val > 0:
                for t in active_today:
                    _anteil = (smi_shares[t] * row[t]) / _eq_val
                    if _anteil > 0:
                        _zeilen[t] = _aus_verkauf * _anteil

            # Vorabpruefung: lohnt der ganze Vorgang? Gezaehlt wird hier
            # nichts, _fee_probe bucht nicht.
            _gv_probe = sum(_fee_probe(v) for v in _zeilen.values())
            _erloes_probe = _ziel - _gv_probe
            _gk_probe = (_fee_probe(_erloes_probe)
                         + abs(_erloes_probe) * fx_fee) if _erloes_probe > 0 else 0.0

            if _erloes_probe > 0 and (_gv_probe + _gk_probe) < _ziel:
                # Stufe 1: verkaufen
                _gv = 0.0
                for t, v in _zeilen.items():
                    _gv += _fee(v, key=t)
                    smi_shares[t] -= v / row[t]
                _erloes = _ziel - _gv
                # Stufe 2: Bitcoin kaufen
                _gk = _fee(_erloes, leg="btc", key="__BTC__") + _fx(_erloes)
                total_tx_costs += _gv + _gk
                _netto = _erloes - _gk
                dividend_cash -= _aus_kasse
                usd = _netto / fx_d
                _gekauft = usd / btc_price_d
                btc_held += _gekauft
                btc_u_dca += _gekauft
                att_btc_dca_invested += _ziel
                att_equity_invested -= _aus_verkauf
                transactions.append({
                    "date": d, "type": "BUY", "reason": "ENTNAHME",
                    "btc_amount": _gekauft, "chf_amount": _ziel,
                    "usd_amount": usd, "btc_price_usd": btc_price_d,
                    "usdchf": fx_d,
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

                    # Transaction cost on the BTC sale and the equity re-purchase.
                    # Die Bitcoinseite ist eine Orderzeile plus Devisengebuehr,
                    # die Aktienseite eine Orderzeile JE TITEL.
                    # Vorabpruefung mit der ECHTEN Gebuehr, damit der Schutz
                    # auch bei eingeschaltetem Netting greift.
                    _p_btc = _fee_probe(sell_chf) + abs(sell_chf) * fx_fee
                    _p_erl = sell_chf - _p_btc
                    w_active = None
                    _p_eq = 0.0
                    if active_today:
                        w_active = w[active_today] / w[active_today].sum()
                        _p_eq = sum(_fee_probe(_p_erl * w_active[t])
                                    for t in active_today)
                    _skip_threshold = (_p_btc + _p_eq) >= sell_chf
                    cost = 0.0
                    if not _skip_threshold:
                        cost_btc = _fee(sell_chf, leg="btc", key="__BTC__") + _fx(sell_chf)
                        proceeds = sell_chf - cost_btc
                        cost_eq = 0.0
                        if active_today:
                            cost_eq = sum(_fee(proceeds * w_active[t], key=t)
                                          for t in active_today)
                        cost = cost_btc + cost_eq
                    total_tx_costs += cost
                    net_to_smi = sell_chf - cost

                    if _skip_threshold:
                        # Verkauf zurueckdrehen: Position und Attribution
                        # bleiben, als haette der Termin nicht stattgefunden.
                        btc_held += sell_btc
                        btc_u_init += sell_btc * _f_init
                        btc_u_dca += sell_btc * (1 - _f_init)
                        att_sold_gross_init -= sell_chf * _f_init
                        att_sold_gross_dca -= sell_chf * (1 - _f_init)
                        att_equity_invested -= sell_chf

                    # Reallocate net proceeds to active tickers by renormalized weights
                    if active_today and w_active is not None and not _skip_threshold:
                        for t in active_today:
                            extra_chf = net_to_smi * w_active[t]
                            smi_shares[t] += extra_chf / row[t]

                    if not _skip_threshold:
                        transactions.append({
                            "date": d, "type": "SELL", "reason": "THRESHOLD",
                            "btc_amount": -sell_btc, "chf_amount": -sell_chf,
                            "usd_amount": -sell_usd, "btc_price_usd": btc_price_d,
                            "usdchf": fx_d,
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
                # Jede Gewichtsanpassung ist eine eigene Orderzeile mit eigener
                # Mindestgebuehr. Die frueher verwendete halbierte Summe
                # (one-way turnover) unterschaetzt das: es werden tatsaechlich
                # sowohl Kauf- als auch Verkaufsseite als Order aufgegeben.
                cost = 0.0
                for t in active_today:
                    current_val = smi_shares[t] * row[t]
                    target_value = smi_value * w_active[t]
                    cost += _fee(target_value - current_val, key=t)
                # Schutz: die Gebuehr darf den Aktienbestand nie aufzehren.
                cost = min(cost, max(smi_value - 1.0, 0.0))
                total_tx_costs += cost
                # Apply rebalance to active tickers, then scale to absorb cost
                for t in active_today:
                    target_value = smi_value * w_active[t]
                    smi_shares[t] = target_value / row[t]
                if smi_value > 0:
                    shrink = (smi_value - cost) / smi_value
                    for t in active_today:
                        smi_shares[t] *= shrink

        # 4b. QUARTERLY CAP-ONLY ADJUSTMENT (real SIX two-tier mechanic).
        # SIX runs two DISTINCT operations, and this models both faithfully:
        #   - Annually (3rd Friday September): full review — composition change
        #     plus complete re-weighting. That is block 4 above.
        #   - Quarterly (3rd Friday Mar/Jun/Sep/Dec): capping ONLY — any single
        #     constituent whose weight has drifted above the cap is trimmed back
        #     to the cap and the excess redistributed across the others. Weights
        #     BELOW the cap are left to drift; there is no full reset.
        # Modelling only the annual reset (the previous behaviour) would let a
        # single title run far above the cap for up to a year, which the real
        # index never permits. September is skipped here because the full
        # reset that same day already enforces the cap.
        if (weight_cap and d in cap_dates_set and d not in rebalance_dates_set
                and d != first_day and active_today):
            smi_value = _smi_value_on(d, row)
            # FEASIBILITY GUARD: with n active titles, a cap of c is only
            # satisfiable if n * c >= 1. For the SMI (20 titles, 18% cap) that
            # holds comfortably (3.6), but a shrunken active set — heavy data
            # outage, or a stress configuration — can make it impossible. In
            # that case the capping loop would cap everything and silently
            # DROP the unallocatable remainder, destroying NAV. Skip instead.
            _n_act = len(active_today)
            _feasible = (_n_act * weight_cap) >= (1.0 - 1e-9)
            if smi_value > 0 and _feasible:
                cur_val = {t: smi_shares[t] * row[t] for t in active_today}
                cur_w = {t: cur_val[t] / smi_value for t in active_today}
                if any(v > weight_cap + 1e-12 for v in cur_w.values()):
                    # Iterative capping: trimming one title raises the others,
                    # which can push a second title over the cap in turn.
                    target_w = dict(cur_w)
                    for _ in range(_n_act + 2):
                        over = [t for t in active_today if target_w[t] > weight_cap + 1e-12]
                        if not over:
                            break
                        excess = sum(target_w[t] - weight_cap for t in over)
                        for t in over:
                            target_w[t] = weight_cap
                        under = [t for t in active_today if target_w[t] < weight_cap - 1e-12]
                        base = sum(target_w[t] for t in under)
                        if not under or base <= 0:
                            break
                        for t in under:
                            target_w[t] += excess * (target_w[t] / base)
                    # SAFETY NET: whatever the loop did (converged, hit the
                    # iteration limit, or broke out early), the weights MUST
                    # still sum to 1 — renormalise so no NAV can ever leak out
                    # of this block.
                    _wsum = sum(target_w.values())
                    if _wsum > 0:
                        for t in active_today:
                            target_w[t] /= _wsum
                        cost = sum(_fee(smi_value * target_w[t] - cur_val[t], key=t)
                                   for t in active_today)
                        cost = min(cost, max(smi_value - 1.0, 0.0))
                        total_tx_costs += cost
                        for t in active_today:
                            smi_shares[t] = (smi_value * target_w[t]) / row[t]
                        shrink = (smi_value - cost) / smi_value
                        for t in active_today:
                            smi_shares[t] *= shrink

        # 4c. TAGESABRECHNUNG. Alle Bewegungen des Tages wurden je Instrument
        # saldiert, jetzt faellt je Instrument eine Gebuehr an. Sie wird
        # zuerst der Kasse belastet, danach anteilig den Positionen.
        if netting:
            _px_chf = (btc_price_d * fx_d) if (btc_price_d and fx_d) else 0.0
            _tg = _tag_abrechnen(d, row, active_today, _px_chf)
            total_tx_costs += _tg
            if _tg > 0:
                # Die Gebuehr wird ausschliesslich den Positionen belastet,
                # nie der Dividendenkasse. Die Kasse ist ein Durchlaufposten
                # fuer vereinnahmte Ausschuettungen: wuerde die Gebuehr dort
                # abgehen, fehlte sie in der Renditezerlegung, weil ihr kein
                # Attributionsposten gegenuebersteht.
                _rest = _tg
                if _rest > 0:
                    _sv = _smi_value_on(d, row)
                    _bv = btc_held * _px_chf if _px_chf else 0.0
                    _basis = _sv + _bv
                    # Schutz: die Gebuehr darf die Positionen nie aufzehren.
                    _rest = min(_rest, max(_basis - 1.0, 0.0))
                    if _basis > 0 and _rest > 0:
                        _eq_ab = _rest * (_sv / _basis)
                        _btc_ab = _rest - _eq_ab
                        if _eq_ab > 0 and _sv > 0:
                            _shrink = (_sv - _eq_ab) / _sv
                            for t in available:
                                smi_shares[t] *= _shrink
                        if _btc_ab > 0 and _px_chf > 0 and btc_held > 0:
                            _weg = min(_btc_ab / _px_chf, btc_held)
                            btc_held -= _weg
                            _tu = btc_u_init + btc_u_dca
                            _fi = (btc_u_init / _tu) if _tu > 0 else 0.0
                            btc_u_init -= _weg * _fi
                            btc_u_dca -= _weg * (1 - _fi)

        # 4d. SCHUTZKLEMME. In degenerierten Faellen (winziger Basket,
        # absurde Mindestgebuehr, extremer Abfluss) kann eine Position
        # rechnerisch knapp unter null laufen. Das Produkt soll dann auf null
        # stehen bleiben, nicht ins Minus kippen.
        for t in available:
            if smi_shares[t] < 0:
                smi_shares[t] = 0.0
        if btc_held < 0:
            btc_held = 0.0
            btc_u_init = max(btc_u_init, 0.0)
            btc_u_dca = max(btc_u_dca, 0.0)
        if dividend_cash < 0:
            dividend_cash = 0.0

        # 5. Record state of day
        smi_value = _smi_value_on(d, row)
        btc_value_chf = btc_held * btc_price_d * fx_d if (btc_price_d and fx_d) else 0
        total = smi_value + btc_value_chf + dividend_cash
        records.append({
            "date": d, "smi_value": smi_value, "btc_value_chf": btc_value_chf,
            "btc_held": btc_held, "dividend_cash": dividend_cash,
            "total_value": total,
            # Anteilswert, indexiert auf das Startkapital: von Zeichnungen und
            # Ruecknahmen unberuehrt und deshalb die einzige Groesse, die sich
            # ueber verschiedene Zuflussszenarien vergleichen laesst.
            "nav_per_unit": ((total / _anteile * initial_capital)
                             if _anteile > 0 else 0.0),
            "btc_pct": btc_value_chf / total if total > 0 else 0,
        })

    ts = pd.DataFrame(records).set_index("date")
    txs = pd.DataFrame(transactions) if transactions else pd.DataFrame()
    evts = pd.DataFrame(threshold_events) if threshold_events else pd.DataFrame()
    ts.attrs["total_tx_costs"] = total_tx_costs
    ts.attrs["total_wht"] = total_wht
    ts.attrs["net_flow"] = _netto_fluss
    ts.attrs["netting"] = bool(netting)
    ts.attrs["units_end"] = _anteile
    ts.attrs["cost_stats"] = dict(cost_stats)

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
    # Eingezahltes Geld ist keine Rendite: der kumulierte Nettofluss wird
    # herausgerechnet, sonst weist die Zerlegung Zeichnungen als Gewinn aus.
    _pnl_gross = _nav_end - initial_capital - _netto_fluss
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
               crystallization_freq="Quarterly", hurdle_type="Hard Hurdle",
               mgmt_fee_freq="Monthly"):
    """Apply management fee (daily NAV accrual, periodic billing ledger) +
    performance fee (period-end, HWM).

    crystallization_freq: 'Monthly', 'Quarterly', 'Semi-Annual', or 'Annual'
      \u2014 how often the PERFORMANCE fee is crystallized against the HWM.
    mgmt_fee_freq: 'Monthly', 'Quarterly', 'Semi-Annual', or 'Annual' \u2014 how
      often the MANAGEMENT fee is billed/settled. Independent of
      crystallization_freq: some providers bill both on the same cadence,
      others differ (e.g. management fee monthly, performance fee quarterly).
      IMPORTANT: this does NOT change the NAV path \u2014 the management fee is
      always accrued daily against NAV (standard fund practice, and the only
      way to match a stated annual rate exactly regardless of billing
      cadence). mgmt_fee_freq only controls how the already-accrued amounts
      are grouped into a billing ledger (net.attrs['mgmt_fee_events']) for
      reporting \u2014 purely a reporting/cash-settlement view, not a mechanism
      change.
    hurdle_type:
      - 'Hard Hurdle': performance fee charged only on the NAV gain ABOVE the
        hurdle-grown HWM (the hurdle return is fee-free).
      - 'Soft Hurdle': if NAV clears the hurdle-grown HWM, the fee applies to the
        ENTIRE gain above the plain HWM (catch-up over the hurdle).
      - 'No Hurdle (HWM only)': fee on all gains above the HWM (no hurdle).

    Returns (net_series, total_mgmt_chf, total_perf_chf, fee_events_df).
    The management-fee billing ledger is attached as
    net_series.attrs['mgmt_fee_events'] (DataFrame: date, period, amount).
    """
    if gross_values is None or gross_values.empty:
        return gross_values, 0.0, 0.0, pd.DataFrame()

    def _months_for(freq):
        if freq == "Monthly":
            return set(range(1, 13)), 12
        elif freq == "Quarterly":
            return {3, 6, 9, 12}, 4
        elif freq == "Semi-Annual":
            return {6, 12}, 2
        else:
            return {12}, 1

    crystal_months, periods_per_year = _months_for(crystallization_freq)
    mgmt_bill_months, _ = _months_for(mgmt_fee_freq)

    # Adaptive observation frequency: derive periods-per-year from the actual
    # index (≈252 on an equity calendar, ≈365 on the BTC/RE daily calendar) so
    # the management-fee accrual matches the stated annual rate exactly.
    _span_years = max((gross_values.index[-1] - gross_values.index[0]).days
                      / 365.25, 1e-9)
    obs_per_year = max((len(gross_values) - 1) / _span_years, 1.0)

    # GESTAFFELTE MANAGEMENT FEE (volumenabhaengig).
    # mgmt_fee_annual kann entweder eine Zahl sein (fixer Satz, Altverhalten)
    # oder eine aufsteigend sortierte Staffel [(AuM-Schwelle, Satz), ...].
    # Bei einer Staffel wird der Satz TAEGLICH anhand des aktuellen NAV
    # bestimmt: waechst das Produkt ueber eine Schwelle, greift der guenstigere
    # Satz ab diesem Tag; faellt es zurueck, greift wieder der hoehere. Das
    # bildet ab, wie eine volumenabhaengige Gebuehr real abgerechnet wird.
    _fee_tiers = None
    if isinstance(mgmt_fee_annual, (list, tuple)):
        _fee_tiers = sorted(mgmt_fee_annual, key=lambda x: x[0])
        _flat_mgmt = _fee_tiers[0][1]
    else:
        _flat_mgmt = float(mgmt_fee_annual)

    def _mgmt_rate_for(nav):
        if not _fee_tiers:
            return _flat_mgmt
        rate = _fee_tiers[0][1]
        for threshold, r in _fee_tiers:
            if nav >= threshold:
                rate = r
        return rate

    daily_mgmt = _flat_mgmt / obs_per_year
    net = pd.Series(index=gross_values.index, dtype=float)
    # Start the net series at the actual day-0 gross value so the initial
    # transaction-cost drag is reflected in the net NAV as well (previously
    # net was rebased to initial_capital, silently dropping that cost).
    net.iloc[0] = float(gross_values.iloc[0])
    hwm = float(initial_capital)            # plain high water mark (post-fee highs)
    prev_cryst_date = gross_values.index[0]  # for pro-rata hurdle on partial periods
    total_mgmt = 0.0
    period_mgmt = 0.0        # mgmt fee accrued within the current PERF-crystallization period (for the combined ledger)
    mgmt_bill_accum = 0.0    # mgmt fee accrued within the current INDEPENDENT mgmt-billing period
    total_perf = 0.0
    fee_events = []
    mgmt_events = []

    def _period_label(freq, date):
        if freq == "Monthly":
            return date.strftime("%b %Y")
        elif freq == "Semi-Annual":
            return f"H{1 if date.month <= 6 else 2} {date.year}"
        elif freq == "Annual":
            return f"{date.year}"
        else:  # Quarterly
            return f"Q{(date.month - 1) // 3 + 1} {date.year}"

    for i in range(1, len(gross_values)):
        d = gross_values.index[i]
        gross_today = float(gross_values.iloc[i])
        gross_prev = float(gross_values.iloc[i - 1])
        gross_ret = (gross_today / gross_prev - 1.0) if gross_prev > 0 else 0.0

        nv = net.iloc[i - 1] * (1.0 + gross_ret)
        mgmt_today = nv * (_mgmt_rate_for(nv) / obs_per_year)
        nv -= mgmt_today
        total_mgmt += mgmt_today
        period_mgmt += mgmt_today
        mgmt_bill_accum += mgmt_today

        is_last = (i == len(gross_values) - 1)
        if is_last:
            is_period_end = True
        else:
            next_d = gross_values.index[i + 1]
            is_period_end = (d.month in crystal_months and next_d.month != d.month)
            is_mgmt_bill_end = (d.month in mgmt_bill_months and next_d.month != d.month)
        if is_last:
            is_mgmt_bill_end = True

        # Unabhängige Management-Fee-Abrechnungsledger \u2014 rein für die
        # Zahlungs-/Rechnungsansicht, ändert die tägliche NAV-Belastung NICHT.
        if is_mgmt_bill_end:
            mgmt_events.append({
                "date": d, "period": _period_label(mgmt_fee_freq, d),
                "year": d.year, "mgmt_fee": mgmt_bill_accum,
            })
            mgmt_bill_accum = 0.0

        if is_period_end:
            period_label = _period_label(crystallization_freq, d)

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

    net.attrs["mgmt_fee_events"] = pd.DataFrame(mgmt_events)
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
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    if _sleeve_cfg:
        # ETF-Sleeve: ein einziger Titel mit Gewicht 100 Prozent. Weder
        # Kappung noch Rebalancing haben dann eine Wirkung, die Engine
        # rechnet sie zu einer Nulloperation ohne Gebuehr.
        tickers = [_sleeve_cfg["ticker"]]
        weights = {_sleeve_cfg["ticker"]: 100.0}
    else:
        tickers = list(SMI_CONSTITUENTS.keys())
        if weighting_method.startswith("Equal"):
            weights = {t: 5.0 for t in tickers}
        else:
            weights = {t: v[1] for t, v in SMI_CONSTITUENTS.items()}
            weights = {t: min(w, 18.0) for t, w in weights.items()}

    _synth_quelle = _sleeve_cfg.get("synth_from") if _sleeve_cfg else None
    if _synth_quelle:
        # Rekonstruierte thesaurierende Reihe: Kurse und Ausschuettungen der
        # ausschuettenden Tranche holen, daraus die thesaurierende bauen. Die
        # Strategie sieht danach einen Titel ohne Ausschuettungen.
        with st.spinner("Loading distributing share class ..."):
            _p_src = fetch_prices([_synth_quelle], start_str, end_str)
            _d_src = fetch_dividends([_synth_quelle], start_str, end_str)
        if _p_src is None or _p_src.empty or _synth_quelle not in _p_src.columns:
            st.error(
                f"Keine Kurse für {_synth_quelle} erhalten. Ohne die "
                "ausschüttende Tranche lässt sich die thesaurierende nicht "
                "rekonstruieren.")
            st.stop()
        prices = pd.DataFrame({
            _sleeve_cfg["ticker"]: synthesize_accumulating(
                _p_src[_synth_quelle], _d_src, _synth_quelle)})
        divs = pd.DataFrame(columns=["date", "ticker", "dividend_per_share"])
        st.session_state["synth_debug"] = {
            "quelle": _synth_quelle, "ziel": _sleeve_cfg["ticker"],
            "n_divs": 0 if _d_src is None or _d_src.empty else len(_d_src),
        }
    else:
        with st.spinner("Loading SMI constituent prices ..."):
            prices = fetch_prices(tickers, start_str, end_str)
        with st.spinner("Loading dividend history ..."):
            divs = fetch_dividends(tickers, start_str, end_str)
    with st.spinner("Loading FX (USDCHF) ..."):
        fx = fetch_series("USDCHF=X", start_str, end_str)
    with st.spinner("Loading Bitcoin series ..."):
        btc_spot = fetch_series("BTC-USD", start_str, end_str)
        if btc_source.startswith("IB1T ETP"):
            # DEFAULT: model the actual instrument (IB1T ETP) from the full
            # spot history minus the accrued TER. See apply_etp_ter().
            btc_series = apply_etp_ter(btc_spot, etp_ter_pct)
        elif btc_source.startswith("BTC-USD Spot"):
            # Reference only: raw spot, no instrument cost. Useful to quantify
            # exactly what the ETP wrapper costs, not for the official figures.
            btc_series = btc_spot
        elif btc_source.startswith("IBIT tatsächliche"):
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
    # SIX-Zweistufigkeit: neben dem jaehrlichen Voll-Reset zusaetzlich die
    # quartalsweisen Kappungstermine. Der 18%-Cap ist Bestandteil der
    # Marktkapitalisierungs-Methode; bei Gleichgewichtung (5% je Titel) gibt
    # es nichts zu kappen.
    _uses_cap = weighting_method.startswith("Marktkapitalisierung")
    cap_dates = get_rebalance_dates(prices.index, "Quartalsweise") if _uses_cap else set()
    weight_cap_val = 0.18 if _uses_cap else None

    # =====================================================================
    # DATENQUALITÄT — Ausreisser-Diagnose
    # Findet den genauen Titel + das genaue Datum hinter unplausiblen
    # Tagesbewegungen (z.B. eine fehlerhafte Split-Zuordnung oder ein
    # schlechter Yahoo-Finance-Tick), BEVOR sie sich unbemerkt in die
    # Monatsrenditen-Heatmap durchschlagen. Kein Rätselraten — jede
    # auffällige Bewegung wird mit Titel, Datum und Vorher/Nachher-Kurs
    # ausgewiesen.
    # =====================================================================
    if _synth_quelle:
        with st.expander("🧪 Aktien-Sleeve: Rekonstruktion gegen die echten Kurse"):
            st.caption(
                f"Die thesaurierende Reihe wird aus {_synth_quelle} "
                f"rekonstruiert, weil {_sleeve_cfg['ticker']} erst seit "
                f"{_sleeve_cfg['seit']} handelt. Seitdem gibt es echte Kurse, "
                "an denen sich die Rekonstruktion messen lässt. Der Zeitraum "
                "ist kurz, deshalb ist das eine Plausibilitätsprüfung und kein "
                "Beweis.")
            try:
                _echt = fetch_prices([_sleeve_cfg["ticker"]], start_str, end_str)
            except Exception:
                _echt = None
            if (_echt is None or _echt.empty
                    or _sleeve_cfg["ticker"] not in _echt.columns):
                st.warning(
                    f"Für {_sleeve_cfg['ticker']} kamen keine Kurse zurück. "
                    "Entweder führt Yahoo Finance diese Anteilsklasse nicht "
                    "unter diesem Kürzel, oder der Backtest-Zeitraum endet vor "
                    "ihrer Auflegung. Die Rekonstruktion wird trotzdem "
                    "verwendet, bleibt aber ungeprüft.")
            else:
                _e = _clean_index(_echt[_sleeve_cfg["ticker"]].dropna())
                _s = _clean_index(prices[_sleeve_cfg["ticker"]].dropna())
                _ue = pd.concat([_e, _s], axis=1, join="inner").dropna()
                _ue.columns = ["echt", "rekonstruiert"]
                if len(_ue) < 20:
                    st.warning(
                        f"Nur {len(_ue)} gemeinsame Handelstage. Das reicht "
                        "für keine belastbare Aussage.")
                else:
                    _j = max((_ue.index[-1] - _ue.index[0]).days / 365.25, 1e-9)
                    _re = (_ue["echt"].iloc[-1]/_ue["echt"].iloc[0])**(1/_j) - 1
                    _rr = (_ue["rekonstruiert"].iloc[-1]
                           / _ue["rekonstruiert"].iloc[0])**(1/_j) - 1
                    _diff = _rr - _re
                    _v1, _v2, _v3 = st.columns(3)
                    _v1.metric("echte Anteilsklasse", f"{_re*100:.2f}% p.a.")
                    _v2.metric("Rekonstruktion", f"{_rr*100:.2f}% p.a.")
                    _v3.metric("Abweichung", f"{_diff*100:+.2f} pp",
                               f"über {len(_ue)} Handelstage")
                    if abs(_diff) < 0.005:
                        st.success(
                            f"Die Rekonstruktion liegt {abs(_diff)*100:.2f} "
                            "Prozentpunkte neben der echten Anteilsklasse. Das "
                            "ist im Bereich der Tracking Difference zwischen "
                            "zwei Anteilsklassen und bestätigt die Annahme.")
                    else:
                        st.warning(
                            f"Die Rekonstruktion weicht um {_diff*100:+.2f} "
                            "Prozentpunkte ab. Das ist mehr als eine Tracking "
                            "Difference erklärt. Mögliche Ursachen: "
                            "unvollständige Ausschüttungsdaten bei Yahoo, ein "
                            "falsches Kürzel, oder eine andere steuerliche "
                            "Behandlung als die angenommenen "
                            f"{int(WITHHOLDING_TAX*100)}%. Vor einem "
                            "Produktentscheid klären.")
                    _figv = go.Figure()
                    _basis_e = float(_ue["echt"].iloc[0])
                    _basis_r = float(_ue["rekonstruiert"].iloc[0])
                    _figv.add_trace(go.Scatter(
                        x=_ue.index, y=_ue["echt"]/_basis_e*100,
                        name="echte Anteilsklasse", mode="lines",
                        line=dict(width=2, color=OAK_SAGE)))
                    _figv.add_trace(go.Scatter(
                        x=_ue.index, y=_ue["rekonstruiert"]/_basis_r*100,
                        name="Rekonstruktion", mode="lines",
                        line=dict(width=2, color=OAK_GOLD, dash="dash")))
                    _figv.update_layout(yaxis_title="indexiert auf 100")
                    st.plotly_chart(style_plotly(_figv, height=340),
                                    use_container_width=True)

    with st.expander("🧪 Instrument — Validierung des ETP-Modells gegen echte Kurse"):
        st.caption(
            "Das Produkt hält Bitcoin über das iShares Bitcoin ETP (IB1T), nicht "
            "direkt. Da IB1T erst seit 25.03.2025 und IBIT erst seit 11.01.2024 "
            "handelt, deckt kein tatsächlicher Instrumentenkurs auch nur ein "
            "vollständiges 3-Jahres-Fenster der Kalibrierung ab. Das Modell "
            "bildet das Instrument daher aus der vollen Spot-Historie abzüglich "
            "laufender TER nach. Dieser Test prüft empirisch, ob diese Annahme "
            "das reale Instrumentenverhalten trifft — gemessen an IBIT, das die "
            "längste verfügbare Historie hat.")
        if st.button("Modell gegen echte IBIT-Kurse prüfen", key="smi_etp_val_go"):
            st.session_state["smi_etp_val_run"] = True
        if st.session_state.get("smi_etp_val_run"):
            _ib = fetch_series("IBIT", "2024-01-11", end_str)
            if _ib.empty or btc_spot.empty:
                st.warning("IBIT- oder Spot-Kurse nicht verfügbar — Prüfung übersprungen.")
            else:
                _ov = pd.concat([btc_spot, _ib], axis=1, join="inner").dropna()
                _ov.columns = ["spot", "ibit"]
                if len(_ov) < 60:
                    st.warning(f"Nur {len(_ov)} gemeinsame Handelstage — zu wenig für eine "
                               "belastbare Aussage.")
                else:
                    _yrs = max((_ov.index[-1] - _ov.index[0]).days / 365.25, 1e-9)
                    _rel = (_ov["ibit"] / _ov["ibit"].iloc[0]) / (_ov["spot"] / _ov["spot"].iloc[0])
                    _implied = 1 - _rel.iloc[-1] ** (1 / _yrs)
                    v1, v2, v3 = st.columns(3)
                    with v1:
                        st.metric("Implizierte Gebühr (real)", f"{_implied*100:.3f}% p.a.")
                        st.caption(f"aus {len(_ov)} Handelstagen, {_yrs:.2f} Jahre")
                    with v2:
                        st.metric("Im Modell angesetzt", f"{etp_ter_pct*100:.3f}% p.a.")
                        st.caption("Sidebar-Einstellung")
                    with v3:
                        _gap_bp = (_implied - etp_ter_pct) * 10000
                        st.metric("Abweichung", f"{_gap_bp:+.1f} bp p.a.")
                        st.caption("real minus Modell")
                    st.caption(
                        f"Kumulative Abweichung IBIT gegenüber Spot über den "
                        f"Überlappungszeitraum: {(_rel.iloc[-1]-1)*100:+.2f}%. "
                        "Eine implizierte Gebühr nahe der offiziellen TER bestätigt "
                        "die Modellannahme. Grössere Abweichungen sind erwartbar und "
                        "stammen aus Prämie/Abschlag zum NAV, dem Unterschied zwischen "
                        "CME-CF-Referenzkurs und Spot-Börsenkurs sowie den "
                        "unterschiedlichen Handelszeiten (Spot 24/7, ETP nur "
                        "Börsenstunden) — allesamt zweitrangig und im Reglement als "
                        "Modellannahme offengelegt.")
                    if abs(_gap_bp) > 25:
                        st.warning(
                            f"⚑ Abweichung über 25 bp p.a. — vor der Einreichung prüfen, "
                            "ob der angesetzte TER-Wert noch zum tatsächlichen Instrument "
                            "passt (Waiver ausgelaufen? falsches Instrument gewählt?).")
                    else:
                        st.success("✓ Modellannahme durch reale Instrumentenkurse gestützt.")

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
            # zu vermuten. TEUER (Netzwerk-Aufrufe pro Titel) — deshalb hinter
            # einem Button, läuft NICHT mehr automatisch bei jedem Skript-
            # Durchlauf (das hatte die ganze Plattform spürbar verlangsamt).
            _flagged_tickers = sorted(set(
                r["Ticker"] for r in _outlier_rows if r["Ticker"] != "—"))
            if _flagged_tickers:
                st.caption(f"{len(_flagged_tickers)} auffällige(r) Aktientitel. Rohkurs-"
                           "Diagnose lädt zusätzliche Daten von Yahoo Finance nach — "
                           "nur auf Wunsch, um die Seite nicht bei jedem Aufruf zu verlangsamen.")
                if st.button("🔍 Rohe Split-Ereignisse nachladen und prüfen",
                            key="smi_dq_raw_go"):
                    st.session_state["smi_dq_raw_has_run"] = True

            if _flagged_tickers and st.session_state.get("smi_dq_raw_has_run"):
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
                        # KORREKTUR (zweite Runde): der erste Versuch prüfte die
                        # Kontinuität gegen prices[_ft] — das ist aber schon die
                        # BEREITS bereinigte Serie (Ausgabe von fetch_prices). Ist
                        # die Bereinigung korrekt gelaufen, gibt es dort gar keinen
                        # Sprung mehr zu erklären, und der Test schlägt fälschlich
                        # fehl — unabhängig davon, ob die eigentliche Bereinigung
                        # stimmt. Für einen echten Test brauchen wir die ROHEN,
                        # unbereinigten Kurse — dafür separat und gezielt nur für
                        # diesen einen auffälligen Titel neu laden.
                        _raw_dl = _download_with_retry([_ft], start_str, end_str)
                        _raw_t = None
                        if _raw_dl is not None and not _raw_dl.empty:
                            try:
                                if isinstance(_raw_dl.columns, pd.MultiIndex):
                                    _raw_t = _raw_dl[_ft]["Close"] if (_ft, "Close") in _raw_dl.columns or _ft in _raw_dl.columns.get_level_values(0) else None
                                else:
                                    _raw_t = _raw_dl["Close"]
                                if _raw_t is not None:
                                    _raw_t = _clean_index(_raw_t.dropna())
                            except Exception:
                                _raw_t = None
                        if _raw_t is None or _raw_t.empty:
                            st.caption(f"Konnte rohe Kursdaten für {_fname} nicht separat "
                                       "nachladen — Kontinuitätsprüfung hier übersprungen.")
                            continue
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
                                           f"beobachteten ROHEN Kurssprung (implizite Ratio "
                                           f"{_implied:.2f}, Rohkurs davor {_pb:.2f} → danach "
                                           f"{_pa:.2f}) — sieht nach echtem Split aus, wird "
                                           "in der Bereinigung angewendet.")
                            else:
                                st.warning(
                                    f"⚑ Ratio {_r:.2f} am {_d:%Y-%m-%d} erklärt den "
                                    f"beobachteten ROHEN Kurssprung NICHT (implizite Ratio "
                                    f"{_implied:.2f}, Rohkurs davor {_pb:.2f} → danach "
                                    f"{_pa:.2f}) — wird verworfen, vermutlich Datenfehler "
                                    "des Anbieters.")
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
            cap_dates_set=cap_dates,
            weight_cap=(None if _sleeve_cfg else weight_cap_val),
            min_fee_chf=min_fee_chf, fx_fee_bps=fx_fee_bps,
            min_order_chf=min_order_chf,
            harvest_mode=harvest_mode, withdrawal_pct_monthly=withdrawal_pct,
            withdrawal_every_n_months=withdrawal_n,
            monthly_flow_pct=flow_pct, monthly_flow_chf=flow_chf,
            netting=netting_on,
            cost_titles=_kostenstruktur(_sleeve_cfg),
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
            mgmt_fee_freq=mgmt_fee_freq,
        )
        mgmt_fee_events_df = ts_net.attrs.get("mgmt_fee_events", pd.DataFrame())
        ts["total_value_net"] = ts_net

    # =====================================================================
    # STATUSBAND: was wird hier eigentlich gerechnet?
    # Ohne das steht man nach dem Scrollen vor Zahlen ohne Zuordnung, und
    # die Sleeves unterscheiden sich nur in Nuancen.
    # =====================================================================
    _sb_sleeve = (f"**{_sleeve_cfg['name']}** ({_sleeve_cfg['ticker']}, "
                  f"{_sleeve_cfg['isin']}, Index {_sleeve_cfg['index']}, "
                  f"TER {_sleeve_cfg['ter']*100:.2f}%, "
                  f"{_sleeve_cfg['ausschuettung']})"
                  if _sleeve_cfg else
                  "**20 SMI-Einzeltitel**, Marktkapitalisierung mit 18%-Cap"
                  if weighting_method.startswith("Markt") else
                  "**20 SMI-Einzeltitel**, Gleichgewichtung")
    if _sleeve_cfg and _sleeve_cfg.get("synth_from"):
        _sb_sleeve += (f" · Kurshistorie rekonstruiert aus "
                       f"{_sleeve_cfg['synth_from']}")
    if _sleeve_cfg and _sleeve_cfg.get("kosten_titel"):
        _sb_sleeve += " · **Kontrollrechnung**, Kosten wie 20 Einzeltitel"

    _sb_fin = (entnahme_wortlaut(withdrawal_pct, withdrawal_n)
               if harvest_mode == "withdrawal"
               else f"Dividendenernte über {dca_months} Monate")

    _sb_fluss = "keine Zeichnungen"
    if flow_pct or flow_chf:
        _teile = []
        if flow_pct:
            _teile.append(f"{flow_pct*100:+.1f} % des NAV")
        if flow_chf:
            _teile.append(("%+,.0f CHF" % flow_chf).replace(",", "'"))
        _sb_fluss = "Kapitalfluss " + " und ".join(_teile) + " je Monat"

    # Tausendertrenner NUR auf der Zahl ersetzen, nicht auf dem ganzen Satz:
    # sonst werden auch die Satzkommata zu Apostrophen.
    _sb_minfee = f"{min_fee_chf:,.0f}".replace(",", "'")
    st.info(
        f"**Gerechnet wird:** {_sb_sleeve}\n\n"
        f"Bitcoin über {_sb_fin} · {_sb_fluss} · "
        f"Band {upper_threshold*100:.0f} % auf {target_btc_pct*100:.0f} %\n\n"
        f"Kosten {tx_cost_bps:.0f} bps, mindestens CHF {_sb_minfee} je "
        f"Orderzeile, Devisengebühr {fx_fee_bps:.0f} bps, Netting "
        f"{'ein' if netting_on else 'AUS'} · Zeitraum "
        f"{ts.index[0]:%d.%m.%Y} bis {ts.index[-1]:%d.%m.%Y} "
        f"({(ts.index[-1]-ts.index[0]).days/365.25:.1f} Jahre)")

    # =====================================================================
    # TRANSAKTIONSKOSTEN: Orderzeilen statt nur Franken
    # =====================================================================
    def _chf_ch(x):
        """CHF mit Schweizer Tausendertrenner. fmt_chf kuerzt ab einer Million
        ab und verwendet darunter ein englisches Komma, was fuer die
        Kostenzahlen hier unpassend ist."""
        try:
            return "CHF " + f"{float(x):,.0f}".replace(",", "'")
        except (TypeError, ValueError):
            return "n/a"

    _cs = ts.attrs.get("cost_stats", {})
    if _cs.get("lines", 0) > 0:
        st.markdown("## Transaktionskosten")
        _jahre_k = max((ts.index[-1] - ts.index[0]).days / 365.25, 1e-9)
        _k1, _k2, _k3, _k4 = st.columns(4)
        _k1.metric("Transaktionskosten gesamt",
                   _chf_ch(ts.attrs.get("total_tx_costs", 0.0)),
                   f"{ts.attrs.get('total_tx_costs', 0.0)/initial_capital/_jahre_k*100:.2f}% p.a. auf Startkapital")
        _k2.metric("Orderzeilen", f"{_cs.get('lines', 0):,}".replace(",", "'"),
                   f"{_cs.get('lines', 0)/_jahre_k:.0f} je Jahr")
        _anteil = (_cs.get("lines_at_min", 0) / _cs.get("lines", 1) * 100)
        _k3.metric("davon zur Mindestgebühr",
                   f"{_cs.get('lines_at_min', 0):,}".replace(",", "'"),
                   f"{_anteil:.0f}% aller Zeilen")
        _k4.metric("Devisengebühr Bitcoin", _chf_ch(_cs.get("fee_fx", 0.0)),
                   f"{fx_fee_bps:.0f} bps je Bewegung")

        _mon_k = max(len({(x.year, x.month) for x in ts.index}), 1)
        st.caption(
            f"Das sind **{_cs.get('lines', 0)/_mon_k:.1f} Orderzeilen je "
            f"Monat**. Netting ist "
            + ("eingeschaltet, alle Bewegungen eines Ausführungstages werden "
               "je Instrument zu einer Order zusammengefasst."
               if netting_on else
               "ausgeschaltet, jede Teilbewegung zählt einzeln."))

        _nf = ts.attrs.get("net_flow", 0.0)
        if abs(_nf) > 0.005:
            _awert_end = float(ts["nav_per_unit"].iloc[-1])
            _f1, _f2, _f3 = st.columns(3)
            _f1.metric("Basketwert am Ende", _chf_ch(ts["total_value"].iloc[-1]),
                       "enthält das eingezahlte Geld")
            _f2.metric("kumulierter Nettofluss", _chf_ch(_nf),
                       "Zeichnungen minus Rücknahmen")
            _f3.metric("Anteilswert", _chf_ch(_awert_end),
                       f"{(_awert_end/initial_capital-1)*100:+.1f}% gegenüber Start")
            st.warning(
                "Es sind Kapitalflüsse aktiv. Alle Rendite- und "
                "Risikokennzahlen weiter unten beziehen sich auf den "
                "Basketwert und enthalten damit das eingezahlte Geld. Sie sind "
                "keine Renditen. Vergleichbar über verschiedene "
                "Zuflussszenarien ist allein der **Anteilswert** oben: "
                "Zeichnungen schaffen Anteile zum aktuellen Wert, Rücknahmen "
                "löschen sie, genau wie beim Zertifikat.")
        if _anteil > 50:
            st.warning(
                f"In {_anteil:.0f} Prozent aller Orderzeilen greift die "
                f"Mindestgebühr, der Prozentsatz spielt also keine Rolle mehr. "
                f"Das ist der Bereich, in dem die Zahl der gehandelten Titel "
                f"den Preis bestimmt und nicht das Volumen."
                + (" Ein ETF-Sleeve erzeugt hier eine Orderzeile statt zwanzig."
                   if not _sleeve_cfg else ""))

    # =====================================================================
    # STRUKTURVERGLEICH: Einzeltitel gegen SMI-ETF
    # =====================================================================
    st.markdown("## Strukturvergleich des Aktien-Sleeves")
    st.caption(
        "Derselbe Zeitraum, dieselben Parameter, dieselbe Bitcoin- und "
        "Devisenreihe. Unterschiedlich ist allein der Aufbau des Aktienteils. "
        "Gerechnet wird mit dem eingestellten Kostenmodell, also inklusive "
        "Mindestgebühr je Orderzeile. Alle Werte sind netto nach Management "
        "Fee.")

    def _sleeve_lauf(cfg):
        """Fuehrt die Strategie fuer einen Aktien-Sleeve aus und gibt die
        Kennzahlen zurueck. cfg=None bedeutet die zwanzig Einzeltitel."""
        if cfg is None:
            _tk = list(SMI_CONSTITUENTS.keys())
            if weighting_method.startswith("Equal"):
                _w = {t: 5.0 for t in _tk}; _cap = None
            else:
                _w = {t: min(v[1], 18.0) for t, v in SMI_CONSTITUENTS.items()}
                _cap = 0.18
        else:
            _tk = [cfg["ticker"]]; _w = {cfg["ticker"]: 100.0}; _cap = None
        # Jede Variante laeuft in der Betriebsart, die zu ihr passt: ein
        # thesaurierender ETF ueber die monatliche Entnahme, alles andere
        # ueber die Dividendenernte.
        _thes_v = bool(cfg and cfg.get("thesaurierend"))
        _hm = "withdrawal" if _thes_v else "dividend"
        _wp = (withdrawal_pct if withdrawal_pct > 0 else 0.0025) if _thes_v else 0.0
        _q = cfg.get("synth_from") if cfg else None
        if _q:
            _ps = fetch_prices([_q], start_str, end_str)
            if _ps is None or _ps.empty or _q not in _ps.columns:
                return None
            _ds = fetch_dividends([_q], start_str, end_str)
            _px = pd.DataFrame({_tk[0]: synthesize_accumulating(
                _ps[_q], _ds, _q)})
            _dv = pd.DataFrame(columns=["date", "ticker", "dividend_per_share"])
        else:
            _px = fetch_prices(_tk, start_str, end_str)
            if _px is None or _px.empty:
                return None
            _dv = fetch_dividends(_tk, start_str, end_str)
        _rb = get_rebalance_dates(_px.index, rebalance_freq)
        _cd = get_rebalance_dates(_px.index, "Quartalsweise") if _cap else set()
        _th = (None if _tcf is None else get_rebalance_dates(_px.index, _tcf))
        _t, _, _ = run_strategy(
            _px, _dv, btc_series, fx, initial_capital, _w,
            initial_btc_pct, upper_threshold, target_btc_pct,
            _rb, dca_months, tx_cost_bps=tx_cost_bps,
            threshold_check_dates_set=_th, cap_dates_set=_cd,
            weight_cap=_cap, min_fee_chf=min_fee_chf,
            fx_fee_bps=fx_fee_bps, min_order_chf=min_order_chf,
            harvest_mode=_hm, withdrawal_pct_monthly=_wp,
            withdrawal_every_n_months=(withdrawal_n if _thes_v else 1),
            monthly_flow_pct=flow_pct, monthly_flow_chf=flow_chf,
            netting=netting_on,
            cost_titles=_kostenstruktur(cfg))
        if _t is None or _t.empty or "total_value" not in _t.columns:
            return None
        _net, _, _, _ = apply_fees(
            _t["total_value"], initial_capital, mgmt_fee_annual=mgmt_fee_pct,
            perf_fee_rate=perf_fee_pct, hwm_hurdle=hwm_hurdle_pct,
            crystallization_freq=crystallization_freq,
            hurdle_type=hurdle_type, mgmt_fee_freq=mgmt_fee_freq)
        _s = _t.attrs.get("cost_stats", {})
        _j = max((_t.index[-1] - _t.index[0]).days / 365.25, 1e-9)
        _e = float(_net.iloc[-1])

        # ANTEILSWERTREIHE, netto nach Gebühren. Mit Kapitalflüssen enthält
        # total_value das eingezahlte Geld, deshalb sind Rendite und
        # Rückgang darauf gerechnet sinnlos: die Zuflüsse treiben den Wert
        # und maskieren die Einbrüche. Die Gebührenbelastung ist eine
        # tägliche proportionale Abgrenzung, sie lässt sich deshalb als
        # Verhältnis auf die Anteilswertreihe übertragen.
        if "nav_per_unit" in _t.columns:
            _quote = (_net / _t["total_value"]).replace(
                [float("inf"), float("-inf")], float("nan")).fillna(1.0)
            _reihe = _t["nav_per_unit"] * _quote
        else:
            _reihe = _net
        _awert = float(_reihe.iloc[-1])
        _dd = compute_drawdown(_reihe)
        _mon = max(len({(x.year, x.month) for x in _t.index}), 1)
        return {
            "finanzierung": (entnahme_wortlaut(_wp, withdrawal_n, kurz=True)
                             if _hm == "withdrawal" else "Dividendenernte"),
            "start": _t.index[0], "ende": _t.index[-1], "jahre": _j,
            # Rendite und Rückgang IMMER aus der Anteilswertreihe. netto
            # bleibt der Basketwert, damit die Grössenordnung sichtbar ist.
            "netto": _e, "cagr": (_awert/initial_capital)**(1/_j) - 1,
            "anteilswert": _awert, "zeilen_monat": _s.get("lines", 0)/_mon,
            "kosten": _t.attrs.get("total_tx_costs", 0.0),
            "vst": _t.attrs.get("total_wht", 0.0),
            "zeilen": _s.get("lines", 0), "zeilen_min": _s.get("lines_at_min", 0),
            "mdd": (float(_dd.min()) if not _dd.empty else 0.0),
            "reihe": _reihe,
        }

    if st.button("Strukturvergleich rechnen", key="cmp_run",
                 help="Rechnet die Strategie für jeden Aufbau des Aktienteils "
                      "einmal durch. Das dauert je nach Zeitraum einen Moment."):
        _erg = {}
        _prog = st.progress(0.0, text="Rechne Varianten ...")
        for _i, (_lbl, _cfg) in enumerate(EQUITY_SLEEVES.items()):
            _prog.progress((_i)/len(EQUITY_SLEEVES), text=f"Rechne {_lbl} ...")
            try:
                _erg[_lbl] = _sleeve_lauf(_cfg)
            except Exception as _e:
                _erg[_lbl] = None
                st.warning(f"{_lbl}: Berechnung fehlgeschlagen ({_e}).")
        _prog.empty()
        st.session_state["cmp_result"] = _erg

    _cmp = st.session_state.get("cmp_result")
    if _cmp:
        _gut = {k: v for k, v in _cmp.items() if v}
        if not _gut:
            st.error("Keine Variante konnte gerechnet werden. Meist fehlen die "
                     "ETF-Kursdaten bei Yahoo Finance. Cache leeren und erneut "
                     "versuchen.")
        else:
            # Gemeinsames Fenster ausweisen: die ETFs starten frueher oder
            # spaeter als die Einzeltitel, ein Vergleich ueber verschiedene
            # Zeitraeume waere wertlos.
            _starts = {k: v["start"] for k, v in _gut.items()}
            _enden = {k: v["ende"] for k, v in _gut.items()}
            if (max(_starts.values()) - min(_starts.values())).days > 5 or \
               (max(_enden.values()) - min(_enden.values())).days > 5:
                st.warning(
                    "Die Varianten decken nicht denselben Zeitraum ab: "
                    + " · ".join(f"{k}: {v['start']:%d.%m.%Y} bis {v['ende']:%d.%m.%Y}"
                                 for k, v in _gut.items())
                    + ". Der Vergleich der Endwerte ist dann nur eingeschränkt "
                      "aussagekräftig. Backtest-Zeitraum entsprechend kürzen.")

            _basis = _gut.get("Einzeltitel · 20 SMI-Titel (heutige Struktur)")
            _zeilen = []
            for _lbl, _v in _gut.items():
                _d = (_v["netto"] - _basis["netto"]) if _basis else None
                _zeilen.append({
                    "Aufbau des Aktienteils": _lbl,
                    "Finanzierung Bitcoin": _v["finanzierung"],
                    "Anteilswert": _chf_ch(_v["anteilswert"]),
                    "Rendite p.a. (Anteilswert)": f"{_v['cagr']*100:.2f}%",
                    "grösster Rückgang (Anteilswert)": f"{_v['mdd']*100:.1f}%",
                    "Orderzeilen je Monat": f"{_v['zeilen_monat']:.1f}",
                    "Basketwert am Ende": _chf_ch(_v["netto"]),
                    "Transaktionskosten": _chf_ch(_v["kosten"]),
                    "Kosten p.a.": f"{_v['kosten']/initial_capital/_v['jahre']*100:.2f}%",
                    "Orderzeilen": f"{_v['zeilen']:,}".replace(",", "'"),
                    "davon Mindestgebühr": f"{_v['zeilen_min']:,}".replace(",", "'"),
                    "gegenüber Einzeltiteln": ("Basis" if _basis and _lbl ==
                        "Einzeltitel · 20 SMI-Titel (heutige Struktur)"
                        else (f"{_d:+,.0f}".replace(",", "'") if _d is not None else "n/a")),
                })
            st.dataframe(pd.DataFrame(_zeilen), use_container_width=True,
                         hide_index=True)

            if _basis:
                _best = max(_gut.items(), key=lambda kv: kv[1]["netto"])
                _vor = _best[1]["netto"] - _basis["netto"]
                _ersp = _basis["kosten"] - _best[1]["kosten"]
                if _best[0] != "Einzeltitel · 20 SMI-Titel (heutige Struktur)":
                    # Tausendertrenner nur auf den Zahlen ersetzen, nicht auf
                    # dem ganzen Satz: sonst werden auch die Satzkommata zu
                    # Apostrophen.
                    _z_basis = f"{_basis['zeilen']:,}".replace(",", "'")
                    _z_best = f"{_best[1]['zeilen']:,}".replace(",", "'")
                    st.success(
                        f"**{_best[0]}** liegt über den Zeitraum um "
                        f"{_chf_ch(_vor)} vorn, das sind "
                        f"{_vor/_basis['netto']*100:.1f} Prozent. Die Ersparnis "
                        f"bei den Transaktionskosten beträgt {_chf_ch(_ersp)}, "
                        f"die Zahl der Orderzeilen sinkt von "
                        f"{_z_basis} auf {_z_best}.")
                else:
                    st.warning(
                        "Die Einzeltitel liegen in diesem Zeitraum vorn, aber "
                        "dieser Vergleich trägt nicht. Die Einzeltitelvariante "
                        "rechnet mit den HEUTIGEN Indexmitgliedern zurück bis "
                        "zum Startdatum. Wer damals im Index war und später "
                        "ausschied, fehlt; wer heute drin ist, ist es, weil er "
                        "gut lief. Das ist Survivorship-Bias und kann über zehn "
                        "Jahre leicht einen Prozentpunkt pro Jahr ausmachen. "
                        "Die TER von 0.20 bis 0.35 Prozent erklärt einen "
                        "solchen Abstand nicht.\n\n"
                        "Belastbar ist die Zeile **Kontrolle · SMI-Kurse, "
                        "Kosten wie 20 Einzeltitel**. Sie rechnet "
                        "auf echten Indexkursen, also ohne Verzerrung, trägt "
                        "aber die Kostenstruktur der Einzeltitel. Nur ihr "
                        "Abstand zu den ETF-Varianten misst, was die Zahl der "
                        "gehandelten Titel wirklich kostet.")

            _fig_c = go.Figure()
            for _i, (_lbl, _v) in enumerate(_gut.items()):
                _fig_c.add_trace(go.Scatter(
                    x=_v["reihe"].index, y=_v["reihe"].values, name=_lbl,
                    mode="lines",
                    line=dict(width=2,
                              color=CHART_BAR_COLORS[_i % len(CHART_BAR_COLORS)])))
            _fig_c.update_layout(yaxis_title="Anteilswert netto (CHF)")
            st.plotly_chart(style_plotly(_fig_c, height=420),
                            use_container_width=True)

            st.caption(
                "Zur Einordnung: keine dieser Varianten löst die "
                "Verrechnungssteuer. Ein Schweizer Fonds löst sie mit der "
                "Thesaurierung nach Art. 4 Abs. 1 lit. c VStG genauso aus wie "
                "mit einer Ausschüttung, und ein luxemburgischer Fonds trägt "
                "auf Schweizer Dividenden nach Verwahrstellenpraxis die vollen "
                "35 Prozent ohne Abkommensermässigung. Was der Fonds an "
                "Quellensteuer trägt, steckt bereits in seinem Kurs, dieser "
                "Vergleich bildet es also automatisch ab. Der Unterschied "
                "liegt in den Transaktionskosten und im Betriebsaufwand: der "
                "thesaurierende ETF mit monatlicher Entnahme braucht kein "
                "Tranchenregister und keine Dividendenkasse, sondern zwei "
                "Orderzeilen je Monat.")

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
              f"{mgmt_fee_display*100:.2f}% p.a. on NAV", delta_color="off")
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
                                       step_months, dd_crash_threshold,
                                       cache_token=None):
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
                _dw_candidates, [0.10, 0.20], 0.10, 3, 12, _dw_dd,
                cache_token=(weighting_method, start_str, end_str, btc_source, etp_ter_pct))

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
                               perf_rate, cache_token=None):
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
                hwm_hurdle_pct, perf_fee_pct, cache_token=(weighting_method, start_str, end_str, btc_source, etp_ter_pct))

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
                        & (np.isclose(sgrid["fee"], mgmt_fee_display))]
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
                    f"Bei {_cur*100:.0f}% Startallokation und {mgmt_fee_display*100:.2f}% Fee. "
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
    # ======================================================================
    # KALIBRIERUNG — Ausführungskonvention (INDIFFERENZ-Test)
    # Beantwortet NICHT "welcher Tag bringt die hoechste Rendite" (das waere
    # Market-Timing und widerspraeche der prognosefreien Positionierung),
    # sondern "macht die Tageswahl ueberhaupt einen materiellen Unterschied".
    # Drei Belege: (1) Groesse des Effekts gegenueber der Fensterstreuung,
    # (2) Rangstabilitaet ueber die Fenster, (3) Out-of-sample-Persistenz.
    # ======================================================================
    st.markdown("---")
    # ======================================================================
    # HALTEDAUER-ANALYSE — Trefferquote und Mechanismus-Beitrag
    # Beantwortet die zwei Aussagen, mit denen das Produkt verkauft wird:
    #   1) "Wir schlagen die reine Indexanlage" -> ueber WELCHE Haltedauer?
    #   2) "Der Mechanismus traegt bei"         -> ab WELCHER Haltedauer?
    # Beides ist haltedauerabhaengig; eine pauschale Aussage ("fast immer")
    # haelt der Pruefung nicht stand, eine nach Haltedauer aufgeschluesselte
    # schon.
    #
    # METHODIK Trefferquote: Ein Anleger kauft in ein LAUFENDES Produkt und
    # haelt N Jahre. Es wird daher die bestehende NAV-Reihe in Fenster
    # geschnitten, NICHT die Engine je Fenster neu gestartet — Letzteres
    # unterstellte, das Produkt starte fuer jeden Anleger neu, was falsch waere.
    # METHODIK DCA-Anteil: hier IST ein Neustart korrekt, denn die Attribution
    # misst, wie viel des Bitcoin-Gewinns AB Auflage aus dem Ertragsmechanismus
    # stammt. Deshalb je Fenster ein echter Engine-Lauf.
    # ======================================================================
    st.markdown("---")
    st.markdown("## Haltedauer — Trefferquote & Beitrag des Mechanismus")
    st.markdown(
        "<p style='color:#A9B5A4;margin-top:-6px'>Beide Kernaussagen des "
        "Produkts sind haltedauerabhängig. Kurzfristig verliert eine Strategie "
        "mit Bitcoin-Anteil regelmässig gegen einen reinen Aktienindex; "
        "langfristig wächst zugleich der Beitrag des ertragsfinanzierten "
        "Mechanismus, weil sich mehr Dividenden angesammelt haben. Diese "
        "Sektion beziffert beides.</p>", unsafe_allow_html=True)

    hd1, hd2 = st.columns([1, 2])
    with hd1:
        _hd_step = st.selectbox("Fenster-Schritt", ["monatlich", "quartalsweise"],
                                index=0, key="smi_hd_step")
    with hd2:
        st.caption("")
        _hdgo = st.button("Haltedauer-Analyse starten", key="smi_hd_go")
    if _hdgo:
        st.session_state["smi_hd_run"] = True

    if st.session_state.get("smi_hd_run"):
        _hd_periods = [1, 2, 3, 5]
        _hd_freq = "MS" if _hd_step == "monatlich" else "3MS"
        _net = ts["total_value_net"].dropna()
        _bm = bench["smi_tr"].dropna() if not bench.empty else pd.Series(dtype=float)

        if _bm.empty or len(_net) < 300:
            st.warning("Benchmark- oder Strategiereihe zu kurz für die Analyse.")
        else:
            _rows = []
            for _yrs in _hd_periods:
                _starts = pd.date_range(_net.index[0],
                                        _net.index[-1] - pd.DateOffset(years=_yrs),
                                        freq=_hd_freq)
                _wins, _exc = 0, []
                for _s in _starts:
                    _e = _s + pd.DateOffset(years=_yrs)
                    _ns = _net[(_net.index >= _s) & (_net.index <= _e)]
                    _bs = _bm[(_bm.index >= _s) & (_bm.index <= _e)]
                    if len(_ns) < 100 or len(_bs) < 100:
                        continue
                    _t = max((_ns.index[-1] - _ns.index[0]).days / 365.25, 1e-9)
                    _cs = (_ns.iloc[-1] / _ns.iloc[0]) ** (1 / _t) - 1
                    _cb = (_bs.iloc[-1] / _bs.iloc[0]) ** (1 / _t) - 1
                    _exc.append(_cs - _cb)
                    if _cs > _cb:
                        _wins += 1
                if _exc:
                    _rows.append({
                        "Haltedauer": f"{_yrs} Jahr" + ("" if _yrs == 1 else "e"),
                        "Fenster": len(_exc),
                        "Trefferquote": _wins / len(_exc),
                        "Median-Mehrrendite": float(np.median(_exc)),
                        "P25": float(np.percentile(_exc, 25)),
                        "Schlechtestes": float(np.min(_exc)),
                    })
            _hd = pd.DataFrame(_rows)

            if _hd.empty:
                st.warning("Keine auswertbaren Fenster.")
            else:
                st.markdown("##### Trefferquote gegenüber dem SMI Total Return")
                _d = _hd.copy()
                _d["Trefferquote"] = (_d["Trefferquote"] * 100).round(1).astype(str) + "%"
                _d["Median-Mehrrendite"] = (_d["Median-Mehrrendite"] * 100).round(2).astype(str) + "pp p.a."
                _d["P25"] = (_d["P25"] * 100).round(2).astype(str) + "pp"
                _d["Schlechtestes"] = (_d["Schlechtestes"] * 100).round(2).astype(str) + "pp"
                st.dataframe(_d, use_container_width=True, hide_index=True)

                figh = go.Figure()
                figh.add_trace(go.Bar(
                    x=_hd["Haltedauer"], y=_hd["Trefferquote"] * 100,
                    marker_color=OAK_GOLD,
                    text=[f"{v*100:.0f}%" for v in _hd["Trefferquote"]],
                    textposition="outside"))
                figh.add_hline(y=50, line=dict(color=OAK_CREAM_DIM, dash="dot"),
                               annotation_text="50% — Münzwurf")
                figh.update_yaxes(title_text="Anteil Fenster vor dem Index (%)",
                                  range=[0, 105])
                figh.update_xaxes(title_text="Haltedauer")
                figh = style_plotly(figh, height=340)
                st.plotly_chart(figh, use_container_width=True)
                st.caption(
                    "Ein Anleger kauft in das LAUFENDE Produkt und hält die "
                    "angegebene Dauer. Gerechnet auf der bestehenden NAV-Reihe "
                    "gegen den SMI Total Return über identische Zeiträume, "
                    "netto nach allen Gebühren.")

                # ---- Beitrag des Mechanismus nach Haltedauer ----
                st.markdown("##### Beitrag des Ertragsmechanismus nach Haltedauer")
                with st.spinner("Rechne Attribution je Haltedauer…"):
                    _att_rows = []
                    for _yrs in _hd_periods:
                        _starts = pd.date_range(prices.index[0],
                                                prices.index[-1] - pd.DateOffset(years=_yrs),
                                                freq="6MS")
                        _shares = []
                        for _s in _starts:
                            _w = prices.index[(prices.index >= _s) &
                                              (prices.index <= _s + pd.DateOffset(years=_yrs))]
                            if len(_w) < 100:
                                continue
                            try:
                                _tsx, _, _ = run_strategy(
                                    prices.loc[_w], divs, btc_series, fx,
                                    initial_capital=initial_capital, weights=weights,
                                    initial_btc_pct=initial_btc_pct,
                                    upper_threshold=upper_threshold,
                                    target_btc_pct=target_btc_pct,
                                    rebalance_dates_set=rebal_dates, dca_months=dca_months,
                                    tx_cost_bps=tx_cost_bps, cap_dates_set=cap_dates,
                                    weight_cap=weight_cap_val)
                            except Exception:
                                continue
                            if _tsx.empty:
                                continue
                            _sh = _tsx.attrs.get("attribution", {}).get("dca_share")
                            if _sh is not None and _sh == _sh:
                                _shares.append(_sh)
                        if _shares:
                            _att_rows.append({
                                "Haltedauer": f"{_yrs} Jahr" + ("" if _yrs == 1 else "e"),
                                "Fenster": len(_shares),
                                "Median DCA-Anteil": float(np.median(_shares)),
                            })
                    # Volle Periode als Referenzpunkt ergaenzen
                    _full = ts.attrs.get("attribution", {}).get("dca_share")
                    _full_yrs = (ts.index[-1] - ts.index[0]).days / 365.25
                    if _full is not None and _full == _full:
                        _att_rows.append({
                            "Haltedauer": f"{_full_yrs:.1f} Jahre (voll)",
                            "Fenster": 1, "Median DCA-Anteil": float(_full)})
                _att = pd.DataFrame(_att_rows)
                if not _att.empty:
                    _ad = _att.copy()
                    _ad["Median DCA-Anteil"] = (_ad["Median DCA-Anteil"] * 100).round(1).astype(str) + "%"
                    st.dataframe(_ad, use_container_width=True, hide_index=True)
                    figa = go.Figure()
                    figa.add_trace(go.Bar(
                        x=_att["Haltedauer"], y=_att["Median DCA-Anteil"] * 100,
                        marker_color=OAK_SAGE,
                        text=[f"{v*100:.0f}%" for v in _att["Median DCA-Anteil"]],
                        textposition="outside"))
                    figa.update_yaxes(title_text="Anteil des BTC-Gewinns aus dem Mechanismus (%)")
                    figa.update_xaxes(title_text="Haltedauer")
                    figa = style_plotly(figa, height=340)
                    st.plotly_chart(figa, use_container_width=True)
                    st.caption(
                        "Anteil des Bitcoin-Gewinns, der aus dem dividendenfinanzierten "
                        "Mechanismus stammt statt aus der Startallokation — je Haltedauer "
                        "über mehrere Startzeitpunkte, Median. Hier ist ein Neustart je "
                        "Fenster korrekt, weil die Attribution ab Auflage misst. Der "
                        "Anteil wächst mit der Haltedauer, weil sich über die Zeit mehr "
                        "Dividenden ansammeln.")

    st.markdown("## Kalibrierung — Ausführungskonvention (Indifferenz-Test)")
    st.markdown(
        "<p style='color:#A9B5A4;margin-top:-6px'>Prüft, ob der gewählte "
        "Ausführungstag innerhalb des Monats überhaupt materiell ist \u2014 "
        "ausdrücklich NICHT, welcher Tag historisch die höchste Rendite "
        "gebracht hätte. Ein Tag, der nach Rückschau-Rendite gewählt wird, "
        "wäre Market-Timing und out-of-sample wertlos. Ist der Effekt klein "
        "gegenüber der Streuung zwischen den Einstiegszeitpunkten, darf die "
        "Wahl betrieblich begründet werden \u2014 und genau das hält das "
        "Handelsreglement dann fest.</p>", unsafe_allow_html=True)

    @st.cache_data(ttl=3600, show_spinner=False)
    def compute_execution_convention_grid(_prices, _divs, _btc, _fx, _weights, cap,
                                          conventions, alloc, upper, target, dca_m,
                                          txbps, win_years, step_months, fee,
                                          _rebal, _capd, wcap, cache_token=None):
        full = _prices.index
        if len(full) < 400:
            return pd.DataFrame()
        starts = pd.date_range(full[0], full[-1] - pd.DateOffset(years=win_years),
                               freq=f"{step_months}MS")
        rows = []
        for s in starts:
            w = full[(full >= s) & (full <= s + pd.DateOffset(years=win_years))]
            if len(w) < 300:
                continue
            rec = {"start": s}
            ok = True
            for conv in conventions:
                try:
                    _ts, _, _ = run_strategy(
                        _prices.loc[w], _divs, _btc, _fx, initial_capital=cap,
                        weights=_weights, initial_btc_pct=alloc,
                        upper_threshold=upper, target_btc_pct=target,
                        rebalance_dates_set=_rebal, dca_months=dca_m,
                        tx_cost_bps=txbps, cap_dates_set=_capd, weight_cap=wcap,
                        dca_execution_dates_set=get_execution_dates(w, conv))
                except Exception:
                    ok = False
                    break
                if _ts.empty:
                    ok = False
                    break
                _net, _, _, _ = apply_fees(_ts["total_value"], cap, mgmt_fee_annual=fee,
                                           perf_fee_rate=0.0)
                _yrs = max((_net.index[-1] - _net.index[0]).days / 365.25, 1e-9)
                rec[conv] = (_net.iloc[-1] / cap) ** (1 / _yrs) - 1
            if ok:
                rows.append(rec)
        return pd.DataFrame(rows)

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        _ecw = st.selectbox("Fensterlänge (Jahre)", [3, 5], index=0, key="smi_ec_win")
    with ec2:
        _ecstep = st.selectbox("Fenster-Schritt", ["quartalsweise", "halbjährlich"],
                               index=0, key="smi_ec_step")
    with ec3:
        st.caption("")
        _ecgo = st.button("Indifferenz-Test starten", key="smi_ec_go")
    if _ecgo:
        st.session_state["smi_ec_run"] = True

    if st.session_state.get("smi_ec_run"):
        _convs = ["Monatsultimo", "Monatsanfang", "Monatsmitte",
                  "Letzter Montag", "Erster Montag"]
        _ecsm = 3 if _ecstep == "quartalsweise" else 6
        with st.spinner("Rechne fünf Ausführungskonventionen über alle Fenster…"):
            _ecdf = compute_execution_convention_grid(
                prices, divs, btc_series, fx, weights, initial_capital, _convs,
                initial_btc_pct, upper_threshold, target_btc_pct, dca_months,
                tx_cost_bps, _ecw, _ecsm, mgmt_fee_display, rebal_dates,
                cap_dates, weight_cap_val,
                cache_token=(weighting_method, start_str, end_str, btc_source, etp_ter_pct))

        if _ecdf.empty or len(_ecdf) < 4:
            st.warning("Zu wenig Fenster für eine belastbare Aussage.")
        else:
            _n = len(_ecdf)
            st.caption(f"{_n} rollierende {_ecw}-Jahres-Fenster × {len(_convs)} Konventionen")

            _summ = pd.DataFrame({
                "Konvention": _convs,
                "Median-CAGR": [_ecdf[c].median() for c in _convs],
                "P25": [_ecdf[c].quantile(.25) for c in _convs],
                "P75": [_ecdf[c].quantile(.75) for c in _convs],
            })
            _ranks = _ecdf[_convs].rank(axis=1, ascending=False)
            _summ["Anteil Rang 1"] = [float((_ranks[c] == 1).mean()) for c in _convs]

            _spread_conv = float((_ecdf[_convs].max(axis=1)
                                  - _ecdf[_convs].min(axis=1)).median())
            _spread_wind = float(_ecdf["Monatsultimo"].max() - _ecdf["Monatsultimo"].min())
            _ratio = _spread_conv / _spread_wind if _spread_wind > 0 else float("nan")
            _cagr_range = float(_summ["Median-CAGR"].max() - _summ["Median-CAGR"].min())

            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Spanne der Median-CAGR", f"{_cagr_range*100:.3f}pp")
                st.caption("beste minus schlechteste Konvention")
            with k2:
                st.metric("Streuung zwischen Fenstern", f"{_spread_wind*100:.1f}pp")
                st.caption("Einstiegszeitpunkt, Monatsultimo")
            with k3:
                st.metric("Verhältnis", f"{_ratio*100:.1f}%")
                st.caption("Konvention vs. Einstiegszeitpunkt")

            _d = _summ.copy()
            _d["Median-CAGR"] = (_d["Median-CAGR"]*100).round(3).astype(str) + "%"
            _d["P25"] = (_d["P25"]*100).round(2).astype(str) + "%"
            _d["P75"] = (_d["P75"]*100).round(2).astype(str) + "%"
            _d["Anteil Rang 1"] = (_d["Anteil Rang 1"]*100).round(1).astype(str) + "%"
            st.dataframe(_d, use_container_width=True, hide_index=True)

            # Out-of-sample: Sieger der ersten Haelfte -> Rang in der zweiten
            _h = _n // 2
            _w1 = _ecdf.iloc[:_h][_convs].median().idxmax()
            _r2 = float(_ecdf.iloc[_h:][_convs].median().rank(ascending=False)[_w1])
            _rand = (len(_convs) + 1) / 2
            st.markdown("##### Out-of-sample — hält der historische Sieger?")
            o1, o2 = st.columns(2)
            with o1:
                st.metric("Sieger der ersten Fensterhälfte", _w1)
            with o2:
                st.metric("Dessen Rang in der zweiten Hälfte", f"{_r2:.0f} von {len(_convs)}")
                st.caption(f"Zufallserwartung: {_rand:.1f}")

            if _ratio < 0.10:
                st.success(
                    f"✓ **Indifferent.** Die Tageswahl erklärt {_ratio*100:.1f}% dessen, "
                    f"was der Einstiegszeitpunkt erklärt; die Spanne der Median-CAGR "
                    f"beträgt {_cagr_range*100:.3f} Prozentpunkte. Die Konvention darf "
                    "betrieblich begründet werden (Bewertungsstichtag, Berichtsperiode, "
                    "Liquidität) statt renditeorientiert.")
            else:
                st.warning(
                    f"⚑ **Nicht indifferent** ({_ratio*100:.1f}%). Die Tageswahl ist "
                    "materiell \u2014 vor einer Festlegung die Ursache klären.")
            st.caption(
                "**Zur Rangfolge:** Eine von 20% abweichende Trefferquote ist kein "
                "Beweis für einen ausnutzbaren Renditevorteil. Der strukturelle "
                "Treiber ist die Liegezeit zwischen Dividendenvereinnahmung und "
                "Investition (Cash-Drag): Dividenden fliessen über den Monat "
                "verteilt, weshalb der Monatsultimo im Schnitt am längsten wartet. "
                "Das ist ein operatives Argument, kein Timing-Argument \u2014 und "
                "nur als solches darf es in die Begründung einfliessen. "
                "Entscheidend bleibt die Grössenordnung: ist die CAGR-Spanne im "
                "Bereich weniger Basispunkte, überwiegen die betrieblichen Vorteile "
                "des Monatsultimo (Gleichlauf mit Bewertung, Reporting und "
                "Gebührenabgrenzung).")

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
                                        win_years, step_months, fee,
                                        cache_token=None):
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
                tx_cost_bps, _tfw, _tf_sm, mgmt_fee_display,
                cache_token=(weighting_method, start_str, end_str, btc_source, etp_ter_pct))

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
                               win_years, step_months, fee, cache_token=None):
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
                _ac_allocs, 0.10, dca_months, tx_cost_bps, _acw, _ac_sm, mgmt_fee_display,
                cache_token=(weighting_method, start_str, end_str, btc_source, etp_ter_pct))

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
    # KALIBRIERUNG DER ENTNAHMEMECHANIK (Reglement Fassung 4.0)
    # Das Band 15/25 wurde unter der Dividendenernte kalibriert. Unter der
    # monatlichen Entnahme fliesst dem Satelliten rund die Haelfte mehr zu,
    # das Band wird haeufiger beruehrt. Ziffer 15 sperrt die Parameter ab
    # Emission; ob sie unter der neuen Mechanik tragen, wird hier geprueft.
    #
    # Der Entnahmesatz wird bewusst NICHT nach Rendite gewaehlt: ueber
    # 2015 bis 2026 hat jeder zusaetzliche Franken in Bitcoin die Rendite
    # erhoeht, ein Raster erklaert daher mechanisch den hoechsten Satz zum
    # Sieger. Gezeigt wird, was jede Stufe kostet.
    # ======================================================================
    st.markdown("---")
    st.markdown("## Kalibrierung der Entnahmemechanik")
    st.markdown(
        "<p style='color:#A9B5A4;margin-top:-6px'>Prüft, ob das Band, das "
        "unter der Dividendenernte kalibriert wurde, unter der monatlichen "
        "Entnahme noch trägt, und zeigt, was jeder Entnahmesatz kostet. "
        "Gerechnet wird auf dem thesaurierenden UBS SMI ETF (SMIA, Historie "
        "rekonstruiert). Startallokation und Ziel stehen fest auf 15%. "
        "Kostenmodell wie in der Seitenleiste, ohne Zeichnungen und ohne "
        "Zertifikatsgebühr, damit allein die Mechanik verglichen wird.</p>",
        unsafe_allow_html=True)
    st.warning(
        "**Der Entnahmesatz wird hier nicht nach Rendite gewählt.** In "
        "jedem Zeitraum, in dem Bitcoin die Aktien geschlagen hat, erhöht "
        "jeder zusätzliche Franken in Bitcoin die Rendite, und ein Raster "
        "erklärt mechanisch den höchsten Satz zum Sieger. Das wäre eine "
        "Auswahl nach historischer Rendite, die das Reglement in Ziffer 9.1 "
        "ausdrücklich ablehnt. Ein späteres Startdatum löst das nicht: auch "
        "Fenster, die an den Hochs von 2017 oder 2021 beginnen, endeten drei "
        "Jahre später über dem Einstieg. Der saubere Test ist der "
        "Bitcoin-Pfad ohne Trend unten: dieselben Einbrüche und Erholungen, "
        "aber ohne den Aufwärtstrend. Dort zeigt sich, was jede Stufe "
        "kostet, wenn Bitcoin die Aktien nicht schlägt.")

    @st.cache_data(ttl=3600, show_spinner=False)
    def compute_entnahme_one_combo(_px, _btc, _fx, cap, upper, wpct, txbps,
                                   minfee, fxbps, minorder, win_years,
                                   step_months, cache_token=None):
        """EINE Kombination aus oberer Schwelle und Entnahmesatz ueber alle
        rollierenden Fenster und zusaetzlich ueber den ganzen Zeitraum. Je
        Kombination gecacht, damit ein abgebrochener Lauf nicht von vorn
        beginnt."""
        full = _px.index
        if len(full) < 400:
            return []
        tk = _px.columns[0]
        leer = pd.DataFrame(columns=["date", "ticker", "dividend_per_share"])

        def _ein_lauf(idx):
            _ts, _, _rb = run_strategy(
                _px.loc[idx], leer, _btc, _fx, cap, {tk: 100.0},
                0.15, upper, 0.15, set(), 6, tx_cost_bps=txbps,
                threshold_check_dates_set=None, cap_dates_set=set(),
                weight_cap=None, min_fee_chf=minfee, fx_fee_bps=fxbps,
                min_order_chf=minorder, harvest_mode="withdrawal",
                withdrawal_pct_monthly=wpct, withdrawal_every_n_months=1,
                monthly_flow_pct=0.0, monthly_flow_chf=0.0, netting=True)
            if _ts is None or _ts.empty or "total_value" not in _ts.columns:
                return None
            _rm = risk_metrics(_ts["total_value"])
            _j = max((_ts.index[-1] - _ts.index[0]).days / 365.25, 1e-9)
            _n = 0 if _rb is None else len(_rb)
            _vol = (float(_rb["chf_to_smi"].sum())
                    if _n and "chf_to_smi" in _rb.columns else 0.0)
            _mittel = float(_ts["total_value"].mean())
            _st = _ts.attrs.get("cost_stats", {}) or {}
            _kost = float(_ts.attrs.get("total_tx_costs", 0.0) or 0.0)
            return {
                "upper": upper, "wpct": wpct,
                "cagr": _rm.get("cagr", np.nan),
                "vol": _rm.get("vol", np.nan),
                "sharpe": _rm.get("sharpe", np.nan),
                "max_dd": _rm.get("max_dd", np.nan),
                "calmar": _rm.get("calmar", np.nan),
                "rueck_pa": _n / _j,
                "rueckvol_pa": (_vol / _mittel / _j) if _mittel > 0 else np.nan,
                "btc_quote": float(_ts["btc_pct"].mean()),
                "zeilen_pa": _st.get("lines", 0) / _j,
                "kosten_pa": (_kost / _mittel / _j) if _mittel > 0 else np.nan,
            }

        rows = []
        starts = pd.date_range(
            full[0], full[-1] - pd.DateOffset(years=win_years),
            freq=f"{step_months}MS")
        for s in starts:
            e = s + pd.DateOffset(years=win_years)
            w = full[(full >= s) & (full <= e)]
            if len(w) < 300:
                continue
            try:
                r = _ein_lauf(w)
            except Exception:
                r = None
            if r:
                r["start"] = s
                r["voll"] = False
                rows.append(r)
        try:
            r = _ein_lauf(full)
        except Exception:
            r = None
        if r:
            r["start"] = full[0]
            r["voll"] = True
            rows.append(r)
        return rows

    ek1, ek2, ek3 = st.columns(3)
    with ek1:
        _ek_win = st.selectbox("Fensterlänge (Jahre)", [3, 5], index=0,
                               key="ek_win")
    with ek2:
        _ek_step = st.selectbox(
            "Fenster-Schritt", ["halbjährlich", "quartalsweise"], index=0,
            key="ek_step",
            help="Halbjährlich entspricht der ursprünglichen Kalibrierung nach "
                 "Anhang A. Quartalsweise verdoppelt die Zahl der Fenster und "
                 "die Laufzeit.")
    with ek3:
        st.caption("")
        _ek_go = st.button("Raster starten", key="ek_go")

    _ek_schwellen_pct = st.multiselect(
        "Obere Schwelle (%)",
        options=[20.0, 22.5, 25.0, 27.5, 30.0, 35.0, 40.0],
        default=[20.0, 25.0, 30.0, 35.0], key="ek_upper",
        help="Bitcoin-Anteil, ab dem auf 15% zurückgeführt wird. Reglement "
             "Fassung 4.0: 25%.")
    _ek_saetze_pct = st.multiselect(
        "Entnahmesatz je Monat (%)",
        options=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
        default=[0.15, 0.20, 0.25, 0.35, 0.50], key="ek_wpct",
        help="Anteil des Aktienteils, der an jedem Monatsultimo verkauft "
             "wird. Reglement Fassung 4.0: 0.25%. Der reine Ertrag des ETF "
             "entspricht rund 0.147%.")
    _ek_pfade = ["Historisch",
                 "Ohne Überrendite: Bitcoin wächst wie der SMI-ETF",
                 "Seitwärts: Bitcoin ohne Trend",
                 "Baisse: Bitcoin verliert 10% im Jahr"]
    _ek_pfad = st.selectbox(
        "Bitcoin-Pfad", _ek_pfade, index=0, key="ek_pfad",
        help="Die drei Alternativen behalten jede Tagesbewegung von "
             "Bitcoin, also alle Einbrüche und Erholungen in derselben "
             "Reihenfolge, und entfernen nur den Trend über den "
             "Gesamtzeitraum. Innerhalb einzelner Fenster steigt und fällt "
             "Bitcoin weiterhin; über den ganzen Zeitraum wächst es genau "
             "so stark wie im gewählten Pfad.")

    if _ek_go:
        st.session_state["ek_has_run"] = True

    if st.session_state.get("ek_has_run"):
        _ek_schwellen = sorted(v / 100.0 for v in _ek_schwellen_pct) or [0.25]
        _ek_saetze = sorted(v / 100.0 for v in _ek_saetze_pct) or [0.0025]
        _ek_sm = 6 if _ek_step == "halbjährlich" else 3

        # Kursreihe des thesaurierenden SMI-ETF, auf demselben Weg wie im
        # Strukturvergleich aus der ausschuettenden Klasse rekonstruiert.
        _cfg_smia = next((c for c in EQUITY_SLEEVES.values()
                          if c and c.get("ticker") == "SMIA.SW"), None)
        _px_e = None
        if _cfg_smia and _cfg_smia.get("synth_from"):
            _q = _cfg_smia["synth_from"]
            _ps = fetch_prices([_q], start_str, end_str)
            if _ps is not None and not _ps.empty and _q in _ps.columns:
                _ds = fetch_dividends([_q], start_str, end_str)
                _px_e = pd.DataFrame({_cfg_smia["ticker"]:
                                      synthesize_accumulating(_ps[_q], _ds, _q)})

        if _px_e is None or _px_e.empty:
            st.error("Die Kursreihe des SMIA konnte nicht geladen werden. "
                     "Ohne sie lässt sich das Raster nicht rechnen.")
        else:
            # Bitcoin-Pfad: Trend im Logarithmus ersetzen, Tagesbewegungen
            # behalten. log(P'(t)) = log(P(t)) + (Ziel - Mu) * t
            _btc_e = btc_series
            _b = btc_series.dropna()
            _b = _b[_b > 0]
            _bt = (_b.index - _b.index[0]).days.values / 365.25
            _blg = np.log(_b.values.astype(float))
            _bmu = (_blg[-1] - _blg[0]) / max(float(_bt[-1]), 1e-9)
            _bziel = _bmu
            if _ek_pfad != _ek_pfade[0]:
                if _ek_pfad == _ek_pfade[1]:
                    _sr = _px_e.iloc[:, 0].dropna()
                    _sj = max((_sr.index[-1] - _sr.index[0]).days / 365.25, 1e-9)
                    _bziel = float(np.log(_sr.iloc[-1] / _sr.iloc[0]) / _sj)
                elif _ek_pfad == _ek_pfade[2]:
                    _bziel = 0.0
                else:
                    _bziel = float(np.log(0.90))
                _btc_e = pd.Series(np.exp(_blg + (_bziel - _bmu) * _bt),
                                   index=_b.index, name=btc_series.name)
            st.caption(
                f"Bitcoin im geladenen Zeitraum historisch "
                f"{(np.exp(_bmu)-1)*100:+.1f}% p.a., im gewählten Pfad "
                f"{(np.exp(_bziel)-1)*100:+.1f}% p.a. Tagesbewegungen und "
                f"Einbrüche sind in beiden Fällen dieselben.")
            _ek_slug = {_ek_pfade[0]: "historisch", _ek_pfade[1]: "wie_smi",
                        _ek_pfade[2]: "seitwaerts",
                        _ek_pfade[3]: "baisse"}.get(_ek_pfad, "pfad")

            _ek_combos = [(u, w) for u in _ek_schwellen for w in _ek_saetze]
            _prog = st.progress(0.0, text=f"0 / {len(_ek_combos)} Kombinationen")
            _t0 = _time.time()
            _ek_rows = []
            for _i, (_u, _w) in enumerate(_ek_combos):
                _ek_rows.extend(compute_entnahme_one_combo(
                    _px_e, _btc_e, fx, initial_capital, _u, _w,
                    tx_cost_bps, min_fee_chf, fx_fee_bps, min_order_chf,
                    _ek_win, _ek_sm,
                    cache_token=(start_str, end_str, btc_source, etp_ter_pct,
                                 _ek_pfad)))
                _el = _time.time() - _t0
                _eta = _el / (_i + 1) * (len(_ek_combos) - _i - 1)
                _prog.progress((_i + 1) / len(_ek_combos),
                               text=f"{_i+1} / {len(_ek_combos)} Kombinationen, "
                                    f"{_el:.0f}s gelaufen, noch ca. {_eta:.0f}s")
            _prog.empty()
            _eg = pd.DataFrame(_ek_rows)

            if _eg.empty or "voll" not in _eg.columns:
                st.warning("Zu wenig Daten für die Fensteranalyse.")
            else:
                _fen = _eg[~_eg["voll"]]
                _voll = _eg[_eg["voll"]]
                _nf = _fen["start"].nunique()
                st.caption(
                    f"{len(_eg):,} Engine-Läufe · {len(_ek_combos)} "
                    f"Kombinationen × {_nf} rollierende {_ek_win}-Jahres-"
                    f"Fenster, dazu je ein Lauf über den ganzen Zeitraum · "
                    f"Zeitraum ab {start_str} · Bitcoin-Pfad: {_ek_pfad}")

                _es = _fen.groupby(["upper", "wpct"]).agg(
                    Median_CAGR=("cagr", "median"),
                    Worst_CAGR=("cagr", "min"),
                    Median_DD=("max_dd", "median"),
                    Worst_DD=("max_dd", "min"),
                    Median_Sharpe=("sharpe", "median"),
                    Median_Calmar=("calmar", "median"),
                    Rueck_pa=("rueck_pa", "median"),
                    Rueckvol_pa=("rueckvol_pa", "median"),
                    BTC_Quote=("btc_quote", "median"),
                    Zeilen_pa=("zeilen_pa", "median"),
                    Kosten_pa=("kosten_pa", "median"),
                ).reset_index()
                if not _voll.empty:
                    _es = _es.merge(
                        _voll[["upper", "wpct", "cagr", "max_dd"]].rename(
                            columns={"cagr": "Voll_CAGR", "max_dd": "Voll_DD"}),
                        on=["upper", "wpct"], how="left")

                # ---- Heatmaps: Schwelle waagrecht, Entnahmesatz senkrecht
                def _ek_karte(spalte, titel, fmt, skala, hoeher_besser):
                    _pv = _es.pivot(index="wpct", columns="upper", values=spalte)
                    _x = [f"{u*100:g}%" for u in _pv.columns]
                    _y = [f"{w*100:.2f}%" for w in _pv.index]
                    _z = _pv.values
                    _fig = go.Figure(data=go.Heatmap(
                        z=_z, x=_x, y=_y,
                        colorscale=(skala if hoeher_besser else
                                    [[1 - p, c] for p, c in skala][::-1]),
                        text=[[fmt(v) for v in r] for r in _z],
                        texttemplate="%{text}",
                        textfont=dict(size=11, color=OAK_CREAM),
                        showscale=False,
                        hovertemplate=("Schwelle %{x} · Entnahme %{y}<br>"
                                       + titel + " %{text}<extra></extra>")))
                    if "25%" in _x and "0.25%" in _y:
                        _fig.add_annotation(
                            x="25%", y="0.25%", text="Reglement",
                            showarrow=False, yshift=-15,
                            font=dict(size=9, color=OAK_CREAM))
                    _fig.update_layout(title=titel)
                    _fig = style_plotly(_fig, height=330)
                    _fig.update_xaxes(title_text="Obere Schwelle",
                                      type="category")
                    _fig.update_yaxes(title_text="Entnahmesatz je Monat",
                                      type="category")
                    return _fig

                _skala = [[0, OAK_RED], [0.5, OAK_GREEN_3], [1, OAK_GOLD]]
                _pct = lambda v: "" if pd.isna(v) else f"{v*100:.1f}%"
                _k1, _k2 = st.columns(2)
                with _k1:
                    st.plotly_chart(_ek_karte(
                        "Median_DD", "Max. Drawdown, Median", _pct, _skala,
                        True), use_container_width=True)
                with _k2:
                    st.plotly_chart(_ek_karte(
                        "Worst_CAGR", "Schlechtestes Fenster, Rendite p.a.",
                        _pct, _skala, True), use_container_width=True)
                _k3, _k4 = st.columns(2)
                with _k3:
                    st.plotly_chart(_ek_karte(
                        "Rueck_pa", "Rückführungen je Jahr, Median",
                        lambda v: "" if pd.isna(v) else f"{v:.2f}",
                        _skala, False), use_container_width=True)
                with _k4:
                    st.plotly_chart(_ek_karte(
                        "Median_Sharpe", "Sharpe Ratio, Median",
                        lambda v: "" if pd.isna(v) else f"{v:.2f}",
                        _skala, True), use_container_width=True)
                st.caption(
                    "Median über alle rollierenden Fenster. Das Feld "
                    "«Reglement» markiert die Werte der Fassung 4.0. Beim "
                    "Drawdown ist ein kleinerer Verlust besser, bei den "
                    "Rückführungen eine kleinere Zahl: jede Rückführung "
                    "verkauft Bitcoin, die kurz zuvor mit Aktienerlösen "
                    "gekauft wurden.")

                # ---- Grenzkosten: was kostet jede Stufe des Entnahmesatzes?
                _u_ref = 0.25 if 0.25 in _ek_schwellen else _ek_schwellen[0]
                _gk = _es[_es["upper"] == _u_ref].sort_values("wpct").copy()
                if len(_gk) > 1:
                    st.markdown(
                        f"##### Was kostet jede Stufe des Entnahmesatzes? "
                        f"(obere Schwelle {_u_ref*100:g}%)")
                    _gk["Δ Rendite"] = _gk["Median_CAGR"].diff()
                    _gk["Δ Drawdown"] = _gk["Median_DD"].diff()
                    _gk["Δ schlechtestes Fenster"] = _gk["Worst_CAGR"].diff()
                    _gd = pd.DataFrame({
                        "Entnahme je Monat": _gk["wpct"].map(lambda v: f"{v*100:.2f}%"),
                        "Rendite p.a.": _gk["Median_CAGR"].map(_pct),
                        "Δ Rendite": _gk["Δ Rendite"].map(
                            lambda v: "" if pd.isna(v) else f"{v*100:+.2f}pp"),
                        "Max. Drawdown": _gk["Median_DD"].map(_pct),
                        "Δ Drawdown": _gk["Δ Drawdown"].map(
                            lambda v: "" if pd.isna(v) else f"{v*100:+.2f}pp"),
                        "Schlechtestes Fenster": _gk["Worst_CAGR"].map(_pct),
                        "Δ schlechtestes Fenster": _gk["Δ schlechtestes Fenster"].map(
                            lambda v: "" if pd.isna(v) else f"{v*100:+.2f}pp"),
                        "Rückführungen p.a.": _gk["Rueck_pa"].map(lambda v: f"{v:.2f}"),
                        "Bitcoin-Quote Ø": _gk["BTC_Quote"].map(_pct),
                    })
                    st.dataframe(_gd, use_container_width=True, hide_index=True)
                    _r_lo = float(_gk["Rueck_pa"].iloc[0])
                    _r_hi = float(_gk["Rueck_pa"].iloc[-1])
                    if _r_hi > max(_r_lo, 0.05) * 1.5:
                        st.info(
                            f"Die Rückführungen steigen von {_r_lo:.2f} auf "
                            f"{_r_hi:.2f} je Jahr. Ab einem gewissen Satz "
                            "verkauft das Produkt Aktien, um Bitcoin zu kaufen, "
                            "und verkauft dieselben Bitcoin bald darauf wieder, "
                            "um Aktien zu kaufen. Dieser Kreislauf kostet "
                            "Gebühren, ohne die Quote dauerhaft zu erhöhen: "
                            "das Band deckelt sie bei der oberen Schwelle.")

                # ---- Alle Felder
                st.markdown("##### Alle Kombinationen")
                _ad = pd.DataFrame({
                    "Obere Schwelle": _es["upper"].map(lambda v: f"{v*100:g}%"),
                    "Entnahme je Monat": _es["wpct"].map(lambda v: f"{v*100:.2f}%"),
                    "Rendite p.a. (Median)": _es["Median_CAGR"].map(_pct),
                    "Schlechtestes Fenster": _es["Worst_CAGR"].map(_pct),
                    "Max. DD (Median)": _es["Median_DD"].map(_pct),
                    "Max. DD (schlechtestes)": _es["Worst_DD"].map(_pct),
                    "Sharpe": _es["Median_Sharpe"].map(
                        lambda v: "" if pd.isna(v) else f"{v:.2f}"),
                    "Calmar": _es["Median_Calmar"].map(
                        lambda v: "" if pd.isna(v) else f"{v:.2f}"),
                    "Rückführungen p.a.": _es["Rueck_pa"].map(lambda v: f"{v:.2f}"),
                    "Rückführvolumen p.a.": _es["Rueckvol_pa"].map(_pct),
                    "Bitcoin-Quote Ø": _es["BTC_Quote"].map(_pct),
                    "Orderzeilen p.a.": _es["Zeilen_pa"].map(lambda v: f"{v:.1f}"),
                    "Transaktionskosten p.a.": _es["Kosten_pa"].map(
                        lambda v: "" if pd.isna(v) else f"{v*100:.3f}%"),
                })
                if "Voll_CAGR" in _es.columns:
                    _ad["Ganzer Zeitraum, Rendite"] = _es["Voll_CAGR"].map(_pct)
                    _ad["Ganzer Zeitraum, Max. DD"] = _es["Voll_DD"].map(_pct)
                st.dataframe(_ad, use_container_width=True, hide_index=True)
                _ek_tag = f"{start_str[:4]}_{_ek_slug}"
                _d1, _d2 = st.columns(2)
                with _d1:
                    st.download_button(
                        "Raster als CSV",
                        _es.to_csv(index=False).encode("utf-8"),
                        f"kalibrierung_entnahme_{_ek_tag}.csv", "text/csv",
                        key="ek_csv")
                with _d2:
                    st.download_button(
                        "Einzelfenster als CSV",
                        _eg.to_csv(index=False).encode("utf-8"),
                        f"kalibrierung_entnahme_fenster_{_ek_tag}.csv",
                        "text/csv", key="ek_csv_fenster",
                        help="Jedes Fenster einzeln, damit sich prüfen lässt, "
                             "ob eine Rangfolge über die Zeit stabil ist.")

    # ======================================================================
    # KALIBRIERUNG: Ausfuehrungstag unter der Entnahme (Indifferenz-Test)
    # Wie der bisherige Indifferenz-Test, aber mit der Mechanik der Fassung
    # 4.0: SMIA, Entnahme 0.25% je Termin, Band 25/15. Der Tag bestimmt hier
    # Entnahme, Bitcoinkauf und Bandpruefung zugleich. Die Frage bleibt
    # dieselbe: macht die Tageswahl einen materiellen Unterschied, nicht
    # welcher Tag die hoechste Rendite gebracht haette.
    # ======================================================================
    st.markdown("---")
    st.markdown("## Ausführungstag unter der Entnahme (Indifferenz-Test)")
    st.markdown(
        "<p style='color:#A9B5A4;margin-top:-6px'>Derselbe Test wie in der "
        "Sektion zur Ausführungskonvention, aber mit der Mechanik der "
        "Reglementsfassung 4.0: thesaurierender SMI-ETF (SMIA), Entnahme "
        "0.25% je Termin, Rückführung über 25% auf 15%. Der gewählte Tag "
        "bestimmt Entnahme, Bitcoinkauf und Bandprüfung zugleich. Gezählt "
        "wird die Netto-Rendite nach Zertifikatsgebühr, ohne Zeichnungen. "
        "Die Zahlen dieser Sektion gehören in den Kasten zu Ziffer 9.1 des "
        "Reglements.</p>", unsafe_allow_html=True)

    @st.cache_data(ttl=3600, show_spinner=False)
    def compute_tageswahl_entnahme(_px, _btc, _fx, cap, upper, wpct, txbps,
                                   minfee, fxbps, minorder, fee, win_years,
                                   step_months, conventions, cache_token=None):
        full = _px.index
        if len(full) < 400:
            return pd.DataFrame()
        tk = _px.columns[0]
        leer = pd.DataFrame(columns=["date", "ticker", "dividend_per_share"])
        starts = pd.date_range(full[0], full[-1] - pd.DateOffset(years=win_years),
                               freq=f"{step_months}MS")
        rows = []
        for s in starts:
            w = full[(full >= s) & (full <= s + pd.DateOffset(years=win_years))]
            if len(w) < 300:
                continue
            rec = {"start": s}
            ok = True
            for conv in conventions:
                try:
                    _ts, _, _ = run_strategy(
                        _px.loc[w], leer, _btc, _fx, cap, {tk: 100.0},
                        0.15, upper, 0.15, set(), 6, tx_cost_bps=txbps,
                        threshold_check_dates_set=None, cap_dates_set=set(),
                        weight_cap=None,
                        dca_execution_dates_set=get_execution_dates(w, conv),
                        min_fee_chf=minfee, fx_fee_bps=fxbps,
                        min_order_chf=minorder, harvest_mode="withdrawal",
                        withdrawal_pct_monthly=wpct, withdrawal_every_n_months=1,
                        monthly_flow_pct=0.0, monthly_flow_chf=0.0,
                        netting=True)
                except Exception:
                    ok = False
                    break
                if _ts is None or _ts.empty or "total_value" not in _ts.columns:
                    ok = False
                    break
                _net, _, _, _ = apply_fees(_ts["total_value"], cap,
                                           mgmt_fee_annual=fee,
                                           perf_fee_rate=0.0)
                _yrs = max((_net.index[-1] - _net.index[0]).days / 365.25, 1e-9)
                rec[conv] = (_net.iloc[-1] / cap) ** (1 / _yrs) - 1
            if ok:
                rows.append(rec)
        return pd.DataFrame(rows)

    tw1, tw2, tw3 = st.columns(3)
    with tw1:
        _tw_win = st.selectbox("Fensterlänge (Jahre)", [3, 5], index=0,
                               key="tw_win")
    with tw2:
        _tw_step = st.selectbox(
            "Fenster-Schritt", ["quartalsweise", "halbjährlich"], index=0,
            key="tw_step",
            help="Quartalsweise entspricht dem bisherigen Test, auf den sich "
                 "der Kasten zu Ziffer 9.1 bezieht.")
    with tw3:
        st.caption("")
        _tw_go = st.button("Tagestest starten", key="tw_go")
    _tw_pfade = ["Historisch", "Seitwärts: Bitcoin ohne Trend"]
    _tw_pfad = st.selectbox(
        "Bitcoin-Pfad (Tagestest)", _tw_pfade, index=0, key="tw_pfad",
        help="Seitwärts behält jede Tagesbewegung von Bitcoin und entfernt "
             "nur den Trend über den Gesamtzeitraum, wie in der Kalibrierung "
             "der Entnahmemechanik.")
    if _tw_go:
        st.session_state["tw_has_run"] = True

    if st.session_state.get("tw_has_run"):
        _tw_convs = ["Monatsultimo", "Monatsanfang", "Monatsmitte",
                     "Letzter Montag", "Erster Montag"]
        _tw_sm = 3 if _tw_step == "quartalsweise" else 6
        _cfg_tw = next((c for c in EQUITY_SLEEVES.values()
                        if c and c.get("ticker") == "SMIA.SW"), None)
        _px_t = None
        if _cfg_tw and _cfg_tw.get("synth_from"):
            _qt = _cfg_tw["synth_from"]
            _pst = fetch_prices([_qt], start_str, end_str)
            if _pst is not None and not _pst.empty and _qt in _pst.columns:
                _dst = fetch_dividends([_qt], start_str, end_str)
                _px_t = pd.DataFrame({_cfg_tw["ticker"]:
                                      synthesize_accumulating(_pst[_qt], _dst, _qt)})
        if _px_t is None or _px_t.empty:
            st.error("Die Kursreihe des SMIA konnte nicht geladen werden.")
        else:
            _btc_t = btc_series
            if _tw_pfad != _tw_pfade[0]:
                _bb = btc_series.dropna()
                _bb = _bb[_bb > 0]
                _bbt = (_bb.index - _bb.index[0]).days.values / 365.25
                _bbl = np.log(_bb.values.astype(float))
                _bbm = (_bbl[-1] - _bbl[0]) / max(float(_bbt[-1]), 1e-9)
                _btc_t = pd.Series(np.exp(_bbl - _bbm * _bbt), index=_bb.index,
                                   name=btc_series.name)
            with st.spinner("Rechne fünf Konventionen über alle Fenster…"):
                _tw = compute_tageswahl_entnahme(
                    _px_t, _btc_t, fx, initial_capital, 0.25, 0.0025,
                    tx_cost_bps, min_fee_chf, fx_fee_bps, min_order_chf,
                    mgmt_fee_display, _tw_win, _tw_sm, tuple(_tw_convs),
                    cache_token=(start_str, end_str, btc_source, etp_ter_pct,
                                 _tw_pfad))
            if _tw.empty or len(_tw) < 4:
                st.warning("Zu wenig Fenster für eine belastbare Aussage.")
            else:
                _twn = len(_tw)
                st.caption(
                    f"{_twn} rollierende {_tw_win}-Jahres-Fenster, Schritt "
                    f"{_tw_step}, × {len(_tw_convs)} Konventionen · Zeitraum "
                    f"{start_str} bis {end_str} · Bitcoin-Pfad: {_tw_pfad}")
                _tws = pd.DataFrame({
                    "Konvention": _tw_convs,
                    "Median-Rendite": [_tw[c].median() for c in _tw_convs],
                    "P25": [_tw[c].quantile(.25) for c in _tw_convs],
                    "P75": [_tw[c].quantile(.75) for c in _tw_convs],
                })
                _twr = _tw[_tw_convs].rank(axis=1, ascending=False)
                _tws["Anteil Rang 1"] = [float((_twr[c] == 1).mean())
                                         for c in _tw_convs]
                _tw_spanne = float(_tws["Median-Rendite"].max()
                                   - _tws["Median-Rendite"].min())
                _tw_fenster = float(_tw["Monatsultimo"].max()
                                    - _tw["Monatsultimo"].min())
                _tw_je = float((_tw[_tw_convs].max(axis=1)
                                - _tw[_tw_convs].min(axis=1)).median())
                _tw_ratio = _tw_je / _tw_fenster if _tw_fenster > 0 else float("nan")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Spanne der Median-Rendite",
                              f"{_tw_spanne*100:.2f}pp")
                    st.caption("beste minus schlechteste Konvention")
                with m2:
                    st.metric("Streuung zwischen Fenstern",
                              f"{_tw_fenster*100:.1f}pp")
                    st.caption("Einstiegszeitpunkt, Monatsultimo")
                with m3:
                    st.metric("Verhältnis", f"{_tw_ratio*100:.1f}%")
                    st.caption("Konvention je Fenster vs. Einstiegszeitpunkt")
                _twd = _tws.copy()
                for _c in ("Median-Rendite", "P25", "P75"):
                    _twd[_c] = (_twd[_c] * 100).round(2).astype(str) + "%"
                _twd["Anteil Rang 1"] = ((_twd["Anteil Rang 1"] * 100)
                                         .round(1).astype(str) + "%")
                st.dataframe(_twd, use_container_width=True, hide_index=True)
                _twh = _twn // 2
                _tw_s1 = _tw.iloc[:_twh][_tw_convs].median().idxmax()
                _tw_r2 = float(_tw.iloc[_twh:][_tw_convs].median()
                               .rank(ascending=False)[_tw_s1])
                st.markdown("##### Out-of-sample: hält der historische Sieger?")
                o1, o2 = st.columns(2)
                with o1:
                    st.metric("Sieger der ersten Fensterhälfte", _tw_s1)
                with o2:
                    st.metric("Dessen Rang in der zweiten Hälfte",
                              f"{_tw_r2:.0f} von {len(_tw_convs)}")
                    st.caption(f"Zufallserwartung: {(len(_tw_convs)+1)/2:.1f}")
                st.caption(
                    "Diese Werte ersetzen im Kasten zu Ziffer 9.1 die Zahlen "
                    "des früheren Tests, der noch mit Einzeltiteln und "
                    "Dividendenernte gerechnet war.")
                _tw_slug = "historisch" if _tw_pfad == _tw_pfade[0] else "seitwaerts"
                _tw_schritt = "quartal" if _tw_sm == 3 else "halbjahr"
                st.download_button(
                    "Tagestest als CSV",
                    _tw.to_csv(index=False).encode("utf-8"),
                    f"tageswahl_entnahme_{start_str[:4]}_{_tw_schritt}_"
                    f"{_tw_slug}.csv", "text/csv", key="tw_csv")

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
    def compute_smi_sharpe_one_combo(_prices, _divs, _btc, _fx, _weights, cap,
                                     alloc, width, dca_m, txbps, win_years, step_months,
                                     cache_token=None):
        """EINE (Allokation, Bandbreite, DCA-Fenster)-Kombination über alle
        rollierenden Fenster. Pro Kombination separat gecacht — bricht der
        Lauf ab (Tab geschlossen, Verbindung verloren), sind bereits
        berechnete Kombinationen beim nächsten Start nicht verloren, nur die
        fehlenden werden nachgerechnet."""
        full = _prices.index
        if len(full) < 400:
            return []
        starts = pd.date_range(full[0], full[-1] - pd.DateOffset(years=win_years),
                               freq=f"{step_months}MS")
        target = alloc
        upper = min(alloc + width, 0.95)
        rows = []
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
        return rows

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

    st.markdown(
        "<p style='color:#A9B5A4;margin-top:-6px;font-size:0.85em'>"
        "Entscheidung vom 18.07.2026 (korrigiert): unterer Anker/Zielwert "
        "(wohin bei einem Verkauf zurückgeführt wird) fest auf <strong>15%</strong>. "
        "Oberer Schwellenwert (wo ein Verkauf ausgelöst wird) bei <strong>25%</strong> "
        "— das entspricht 10pp Bandbreite ab diesem Anker, bereits in der "
        "Bandbreiten-Auswahl unten enthalten. Weitere Erhöhung des Ankers nach "
        "oben ausgeschlossen (Produktidentität: Satellit, kein Bitcoin-Fonds). "
        "Bandbreite und DCA-Fenster bleiben offen zur Kalibrierung.</p>",
        unsafe_allow_html=True)

    _sh_alloc_options = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25, 27.5, 30, 35, 40, 45, 50]
    _sh_allocs_pct = st.multiselect(
        "Zu testende Startallokationen / unterer Anker (%)", options=_sh_alloc_options,
        default=[15], key="smi_sh_allocs_ms",
        help="Unterer Anker (Start- und Zielwert nach Verkauf) auf 15% "
             "festgelegt. Der obere Schwellenwert ergibt sich aus Anker + "
             "Bandbreite (unten) — bei 10pp Bandbreite also 25%, wie besprochen.")

    _sh_width_options = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]
    _sh_widths_pct = st.multiselect(
        "Zu testende Bandbreiten (pp)", options=_sh_width_options,
        default=[2.5, 5, 7.5, 10], key="smi_sh_widths_ms",
        help="5pp war beim letzten Lauf der untere Rand des Optimums — jetzt "
             "nach unten geöffnet (2.5pp), um zu prüfen, ob ein engeres Band "
             "noch besser ist. Liegt das Optimum wieder am Rand, weiter "
             "nach unten öffnen.")

    _sh_dca_options = [3, 6, 9, 12, 15, 18, 21, 24]
    _sh_dca_pct = st.multiselect(
        "Zu testende DCA-Fenster (Monate)", options=_sh_dca_options,
        default=[3, 6, 9, 12], key="smi_sh_dca_ms",
        help="12 Monate war beim letzten Lauf der untere Rand des Optimums — "
             "jetzt nach unten geöffnet (ab 3 Monate), um eine kürzere DCA-"
             "Spanne zu prüfen.")

    if _shgo:
        st.session_state["smi_sh_has_run"] = True

    if st.session_state.get("smi_sh_has_run"):
        _sh_allocs = sorted(a / 100.0 for a in _sh_allocs_pct) or [0.25]
        _sh_widths = sorted(w / 100.0 for w in _sh_widths_pct) or [0.05]
        _sh_dca = sorted(_sh_dca_pct) or [12]
        _sh_sm = 6 if _shstep == "halbjährlich" else 3
        _combos = [(a, w, d) for a in _sh_allocs for w in _sh_widths for d in _sh_dca]

        _prog = st.progress(0.0, text=f"0 / {len(_combos)} Kombinationen …")
        _t0 = _time.time()
        _all_rows = []
        for _i, (_a, _w, _d) in enumerate(_combos):
            _rows = compute_smi_sharpe_one_combo(
                prices, divs, btc_series, fx, weights, initial_capital,
                _a, _w, _d, tx_cost_bps, _shw, _sh_sm,
                cache_token=(weighting_method, start_str, end_str, btc_source, etp_ter_pct))
            _all_rows.extend(_rows)
            _elapsed = _time.time() - _t0
            _eta = (_elapsed / (_i + 1)) * (len(_combos) - _i - 1)
            _prog.progress((_i + 1) / len(_combos),
                          text=f"{_i+1} / {len(_combos)} Kombinationen · "
                               f"{_elapsed:.0f}s gelaufen · noch ca. {_eta:.0f}s")
        _prog.empty()
        shgrid = pd.DataFrame(_all_rows)

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

                # RAND-ERKENNUNG: liegt das Optimum am Rand des getesteten
                # Bereichs, ist der wahre Gipfel nicht gefunden — nur der Rand
                # der Suche. Automatisch geprüft, nicht mehr von Auge. Bei einer
                # bewusst FIXIERTEN Dimension (nur ein Wert gewählt, z.B.
                # Allokation fest auf 25% verankert) ist Min=Max desselben
                # Einzelwerts — das ist kein Randproblem, sondern Absicht, und
                # wird hier bewusst nicht gewarnt.
                _edge_msgs = []
                if len(_sh_allocs) > 1:
                    if _best["alloc"] == min(_sh_allocs):
                        _edge_msgs.append(f"Startallokation am UNTEREN Rand "
                                          f"({_best['alloc']*100:.1f}%) — nach unten erweitern.")
                    if _best["alloc"] == max(_sh_allocs):
                        _edge_msgs.append(f"Startallokation am OBEREN Rand "
                                          f"({_best['alloc']*100:.1f}%) — nach oben erweitern.")
                if len(_sh_widths) > 1:
                    if _best["width"] == min(_sh_widths):
                        _edge_msgs.append(f"Bandbreite am unteren Rand "
                                          f"({_best['width']*100:.0f}pp) — engere Werte testen.")
                    if _best["width"] == max(_sh_widths):
                        _edge_msgs.append(f"Bandbreite am oberen Rand "
                                          f"({_best['width']*100:.0f}pp) — weitere Werte testen.")
                if len(_sh_dca) > 1:
                    if _best["dca_m"] == min(_sh_dca):
                        _edge_msgs.append(f"DCA-Fenster am unteren Rand "
                                          f"({_best['dca_m']:.0f} Mte) — kürzere Werte testen.")
                    if _best["dca_m"] == max(_sh_dca):
                        _edge_msgs.append(f"DCA-Fenster am oberen Rand "
                                          f"({_best['dca_m']:.0f} Mte) — längere Werte testen.")
                if _edge_msgs:
                    st.warning(
                        "⚠️ **Optimum liegt am Rand des getesteten Bereichs — "
                        "kein echter Gipfel gefunden, nur der Rand der Suche:**\n\n"
                        + "\n".join(f"- {m}" for m in _edge_msgs)
                        + "\n\nBereich erweitern und neu rechnen, bevor dieser Wert "
                          "als Empfehlung verwendet wird.")
                else:
                    st.success("✓ Optimum liegt innerhalb des getesteten Bereichs, "
                               "nicht am Rand — echter Gipfel gefunden.")

                st.markdown("##### Grenznutzen — was bringt der nächste Allokations-Schritt noch?")
                st.markdown(
                    "<p style='color:#A9B5A4;margin-top:-6px'>Sharpe/Calmar kennen "
                    "kein Konzept von \"Satellit\" — sie maximieren die beste "
                    "risikoadjustierte Rendite, notfalls bis zur Konzentration in "
                    "der historisch stärksten Anlageklasse. Ob das noch ein Satellit "
                    "ist oder schon ein Bitcoin-Fonds mit Aktienbeimischung, "
                    "entscheidet diese Tabelle nicht — sie zeigt nur, wie viel "
                    "JEDER zusätzliche Allokations-Schritt noch bringt, damit die "
                    "Grenze auf Basis von Zahlen gezogen werden kann.</p>",
                    unsafe_allow_html=True)
                _marg = (shgrid.groupby("alloc").agg(
                    Median_Sharpe=("sharpe", "median"),
                    Median_Calmar=("calmar", "median"),
                    Median_CAGR=("cagr", "median"),
                    Median_MaxDD=("max_dd", "median")).reset_index()
                    .sort_values("alloc"))
                _marg["Δ Sharpe"] = _marg["Median_Sharpe"].diff()
                _marg["Δ Calmar"] = _marg["Median_Calmar"].diff()
                _marg["Δ CAGR (pp)"] = _marg["Median_CAGR"].diff() * 100
                _mdisp = _marg.copy()
                _mdisp["Startallokation"] = (_mdisp["alloc"] * 100).round(1).astype(str) + "%"
                _mdisp["Sharpe"] = _mdisp["Median_Sharpe"].round(2)
                _mdisp["Calmar"] = _mdisp["Median_Calmar"].round(2)
                _mdisp["CAGR"] = (_mdisp["Median_CAGR"] * 100).round(1).astype(str) + "%"
                _mdisp["MaxDD"] = (_mdisp["Median_MaxDD"] * 100).round(1).astype(str) + "%"
                _mdisp["Δ Sharpe je Schritt"] = _mdisp["Δ Sharpe"].round(3)
                _mdisp["Δ Calmar je Schritt"] = _mdisp["Δ Calmar"].round(3)
                _mdisp["Δ CAGR je Schritt"] = _mdisp["Δ CAGR (pp)"].round(2).astype(str) + "pp"
                st.dataframe(_mdisp[["Startallokation", "Sharpe", "Calmar", "CAGR", "MaxDD",
                                     "Δ Sharpe je Schritt", "Δ Calmar je Schritt",
                                     "Δ CAGR je Schritt"]].fillna("—"),
                            use_container_width=True, hide_index=True)
                st.warning(
                    "⚠️ **Das ist eine Geschäftsentscheidung, keine Rechenfrage:** "
                    "ab welcher Allokation ist das Produkt kein \"SMI-Kern mit "
                    "Bitcoin-Satellit\" mehr, sondern faktisch umgekehrt? Diese "
                    "Grenze legt fest, bis wohin überhaupt getestet werden sollte "
                    "— unabhängig davon, ob Sharpe/Calmar darüber hinaus noch "
                    "(marginal) weiter steigen würden.")

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
            f"<strong style='color:{OAK_CREAM};'>Management Fee:</strong> {_fee_label} "
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

    if not mgmt_fee_events_df.empty:
        st.markdown(f"##### Management-Fee-Abrechnung ({mgmt_fee_freq})")
        st.caption(
            "Unabhängig von der Performance-Fee-Kristallisation frei wählbar — "
            "manche Anbieter rechnen beide Gebühren auf derselben Frequenz ab, "
            "andere unterschiedlich. Ändert NICHT die tägliche NAV-Belastung "
            "(die immer exakt dem angegebenen Jahressatz entspricht) — gruppiert "
            "nur die bereits korrekt aufgelaufenen Beträge zu den tatsächlichen "
            "Zahlungsterminen des Anbieters.")
        _mfe = mgmt_fee_events_df.copy()
        _mfe["date"] = pd.to_datetime(_mfe["date"]).dt.strftime("%Y-%m-%d")
        _mfe_disp = _mfe.rename(columns={
            "date": "Abrechnungsdatum", "period": "Periode",
            "mgmt_fee": "Management Fee",
        })
        _mfe_disp["Management Fee"] = _mfe_disp["Management Fee"].apply(lambda x: f"CHF {x:,.0f}")
        st.dataframe(_mfe_disp[["Periode", "Abrechnungsdatum", "Management Fee"]],
                     use_container_width=True, hide_index=True, height=min(320, 40 + 35 * len(_mfe_disp)))
        st.caption(f"{len(_mfe_disp)} Abrechnungstermine · Summe: "
                   f"CHF {mgmt_fee_events_df['mgmt_fee'].sum():,.0f} "
                   f"(entspricht Total Management Fee links)")

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
                # WICHTIG: das Produkt haelt Bitcoin NICHT direkt, sondern ueber ein
                # physisch besichertes ETP (Handelsreglement Abschnitt 1a). Frueher
                # stand hier nur "Bitcoin / BTC", was dem Reglement direkt
                # widersprach — wer beide Dokumente nebeneinanderlegt, haette den
                # Widerspruch sofort gefunden.
                _btc_instr_name = ("iShares Bitcoin ETP" if btc_source.startswith("IB1T")
                                   else "Bitcoin")
                _btc_instr_ticker = ("IB1T" if btc_source.startswith("IB1T") else "BTC")
                _btc_sector = ("Digital Assets \u00b7 ETP" if btc_source.startswith("IB1T")
                               else "Digital Assets")
                if _sleeve_cfg:
                    _eq_rows = [[_sleeve_cfg["name"], _sleeve_cfg["ticker"],
                                 "Swiss Equity \u00b7 ETF", 100.0]]
                else:
                    _eq_rows = [[v[0], t, v[2], v[1]]
                                for t, v in SMI_CONSTITUENTS.items()]
                universe_rows = (
                    [[_btc_instr_name, _btc_instr_ticker, _btc_sector, btc_weight_label]]
                    + _eq_rows
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

                # --- Bausteine fuer die Methodik-Tabelle -------------------
                # Diese Tabelle ist in BEIDEN Sprachfassungen englisch beschriftet
                # (siehe uebrige Zeilen), daher auch die Werte englisch — und
                # NICHT ueber die Variable `de`, die nur innerhalb von
                # _xtabs_smi(lang) existiert.
                if btc_source.startswith("IB1T ETP"):
                    _btc_instrument_line = ("iShares Bitcoin ETP (IB1T, ISIN XS2940466316) — "
                                            "physically backed, Swiss-domiciled, USD-denominated")
                    _btc_ter_line = f"{etp_ter_pct*100:.2f}% p.a. (within the instrument, modelled)"
                elif btc_source.startswith("IBIT"):
                    _btc_instrument_line = "iShares Bitcoin Trust (IBIT) — actual prices from 2024"
                    _btc_ter_line = "embedded in price history"
                else:
                    _btc_instrument_line = "Bitcoin spot (reference only, no instrument cost)"
                    _btc_ter_line = "none"

                # Gebuehrenlabel: bisher deutsch ("ab 15 Mio.") auch in der
                # englischen Fassung — jetzt durchgaengig englisch wie die
                # uebrigen Werte dieser Tabelle.
                if isinstance(mgmt_fee_pct, list):
                    _fee_label = " / ".join(
                        (f"{r*100:.2f}%" if th <= 0
                         else f"{r*100:.2f}% above CHF {th/1e6:.0f}m")
                        for th, r in mgmt_fee_pct)
                else:
                    _fee_label = f"{mgmt_fee_pct*100:.2f}% p.a."

                _rebal_label_map_de = {
                    "Jährlich": "jährlich (September)", "Halbjährlich": "halbjährlich",
                    "Quartalsweise": "quartalsweise", "Keine": "nicht rebalanciert",
                }
                _rebal_label_map_en = {
                    "Jährlich": "annually (September)", "Halbjährlich": "semi-annually",
                    "Quartalsweise": "quarterly", "Keine": "not rebalanced",
                }
                # Bei einem ETF-Sleeve gibt es kein Rebalancing des
                # Substanzkerns: der Fonds gewichtet selbst.
                if _sleeve_cfg:
                    _rebal_label_de = "im Fonds"
                    _rebal_label_en = "inside the fund"
                else:
                    _rebal_label_de = _rebal_label_map_de.get(
                        rebalance_freq, rebalance_freq.lower())
                    _rebal_label_en = _rebal_label_map_en.get(
                        rebalance_freq, rebalance_freq.lower())

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
                    rebal_freq_label_de=_rebal_label_de,
                    rebal_freq_label_en=_rebal_label_en,
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
                        ("CAGR Impact", f"{fee_drag*100:.2f}% p.a."),
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
                        ("Equity Weighting",
                         (f"Inside the fund ({_sleeve_cfg['index']} replication)"
                          if _sleeve_cfg else _weighting_method_en)),
                        ("Rebalancing Frequency",
                         ("Inside the fund" if _sleeve_cfg else _rebalance_freq_en)),
                        ("DCA Window", f"{dca_months} months per dividend"),
                        ("Transaction Cost", f"{tx_cost_bps:.0f} bps per trade"),
                        ("Dividend Withholding Tax", f"{int(WITHHOLDING_TAX*100)}% (non-reclaimable, AMC)"),
                        ("Bitcoin Instrument", _btc_instrument_line),
                        ("Bitcoin Instrument Charge", _btc_ter_line),
                        ("Management Fee", _fee_label),
                    ] + ([
                        # Kristallisationsfrequenz und Hurdle sind bei einer
                        # Performance Fee von 0% gegenstandslos und stiften nur
                        # Verwirrung — daher nur zeigen, wenn es sie wirklich gibt.
                        ("Performance Fee", f"{perf_fee_pct*100:.0f}% ({_crystallization_en})"),
                        ("Hurdle", f"{hurdle_type}, {hwm_hurdle_pct*100:.1f}% (Year 1)"),
                    ] if perf_fee_pct > 0 else [
                        ("Performance Fee", "none"),
                    ]) + [
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
    _en_rebal = {"Jährlich": "Annual (September)", "Halbjährlich": "Semi-Annual",
                 "Quartalsweise": "Quarterly", "Keine": "None"}.get(rebalance_freq, rebalance_freq)
    _en_rebal_logic = {"Jährlich": "Annual (September)", "Halbjährlich": "Semi-annual",
                       "Quartalsweise": "Quarterly", "Keine": "No scheduled"}.get(rebalance_freq, rebalance_freq)
    _en_tcf = {"Monatlich (Standard)": "Monthly", "Quartalsweise": "Quarterly",
               "Halbjährlich": "Semi-Annual"}.get(threshold_check_freq, "Monthly")
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
<strong style='color:{OAK_CREAM};'>Equity rebalancing.</strong> {_en_rebal_logic} return to target SMI weights.
The calibrated default (annual, September) aligns with the real SIX composition-review date
(third Friday of September).<br><br>
<strong style='color:{OAK_CREAM};'>Risk management — Threshold rebalance.</strong>
At each month-end, if BTC sleeve exceeds upper threshold, sell down to target weight.
Proceeds reinvested across SMI titles by current target weights.<br><br>
<strong style='color:{OAK_CREAM};'>Result.</strong> Long Swiss equity income + structural BTC exposure
with mechanical profit-taking on outsized crypto appreciation.<br><br>
<strong style='color:{OAK_CREAM};'>Benchmarks.</strong> Strategy (net of fees) is compared against
<em>SMI Total Return</em> (dividends, net of the same 35% withholding tax, reinvested into the
same stocks, rebalanced on the same schedule as the strategy's equity core) and the
<em>SMI Price Index</em> (no dividend reinvestment).<br><br>
<strong style='color:{OAK_CREAM};'>Fees.</strong> Management fee accrued daily, performance fee
crystallized at the configured frequency on returns above a High Water Mark with a Year-1 hurdle.
All risk metrics computed on the net-of-fees series.
</div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("### Active Parameters")
        st.caption("Live values from the current sidebar configuration — the defaults "
                   "are the calibrated values per Handelsreglement.")
        st.markdown(f"""
<div style='color:{OAK_CREAM_DIM}; line-height:1.9;'>
<strong style='color:{OAK_SAGE};'>Initial Allocation</strong><br>
{(1-initial_btc_pct)*100:.0f}% SMI · {initial_btc_pct*100:.0f}% BTC<br><br>
<strong style='color:{OAK_SAGE};'>Upper Threshold</strong><br>
{upper_threshold*100:.0f}% — sell-down trigger<br><br>
<strong style='color:{OAK_SAGE};'>Target</strong><br>
{target_btc_pct*100:.0f}% — post-rebalance weight<br><br>
<strong style='color:{OAK_SAGE};'>DCA Window</strong><br>
{dca_months} months per dividend<br><br>
<strong style='color:{OAK_SAGE};'>Rebalancing</strong><br>
{_en_rebal} (SMI) · {_en_tcf} (BTC check)
</div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Configure parameters in the sidebar, then click **Run Backtest** to begin analysis.")
    footer()
