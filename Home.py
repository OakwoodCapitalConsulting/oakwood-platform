"""
Oakwood Capital Strategy Platform
=================================
Multi-page Streamlit app with strategy selection landing page.
"""

import base64
from pathlib import Path
import streamlit as st

# ---------------------------------------------------------------------------
# Brand palette
# ---------------------------------------------------------------------------
OAK_GREEN     = "#293624"
OAK_GREEN_2   = "#1F2A1B"
OAK_GREEN_3   = "#3A4A33"
OAK_SAGE      = "#99A796"
OAK_SAGE_DIM  = "#A9B5A4"
OAK_CREAM     = "#F5F5F1"
OAK_CREAM_DIM = "#D4D4CE"
OAK_GOLD      = "#C9A961"
OAK_BORDER    = "#3D4A36"


def load_logo_base64():
    here = Path(__file__).parent / "assets"
    for name in ("oakwood_logo.png", "logo.png"):
        path = here / name
        if path.exists():
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
    return None


st.set_page_config(
    page_title="Oakwood Capital — Strategy Platform",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

logo_b64 = load_logo_base64()

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', sans-serif !important;
}}
[data-testid="stAppViewContainer"] {{ background-color: {OAK_GREEN}; }}
[data-testid="stAppViewContainer"] > .main {{ background-color: {OAK_GREEN}; color: {OAK_CREAM}; }}
.main .block-container {{ padding-top: 1rem; padding-bottom: 3rem; max-width: 1300px; }}
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

/* Sidebar */
[data-testid="stSidebar"] {{ background-color: {OAK_GREEN_2}; border-right: 1px solid {OAK_BORDER}; }}
[data-testid="stSidebar"] * {{ color: {OAK_CREAM} !important; }}

/* Sidebar page navigation links (Streamlit multipage nav) */
[data-testid="stSidebarNav"] a {{ color: {OAK_CREAM} !important; }}
[data-testid="stSidebarNav"] a span {{ color: {OAK_CREAM} !important; }}
[data-testid="stSidebarNav"] a:hover {{ background-color: {OAK_GREEN_3} !important; }}
[data-testid="stSidebarNav"] li div a span {{ color: {OAK_CREAM} !important; }}

/* page_link cards ("Open ... Strategy") */
[data-testid="stPageLink"] a, [data-testid="stPageLink"] *,
a[data-testid="stPageLink-NavLink"], a[data-testid="stPageLink-NavLink"] * {{
    color: {OAK_CREAM} !important;
}}
[data-testid="stPageLink"] {{
    background-color: {OAK_GREEN_3};
    border: 1px solid {OAK_BORDER};
    border-radius: 10px;
}}
[data-testid="stPageLink"]:hover {{
    border-color: {OAK_GOLD};
}}
[data-testid="stPageLink"] p {{ color: {OAK_CREAM} !important; font-weight: 600; }}

/* Strategy cards */
.strategy-card {{
    background: {OAK_GREEN_2};
    border: 1px solid {OAK_BORDER};
    border-left: 3px solid {OAK_SAGE};
    border-radius: 10px;
    padding: 28px 32px;
    min-height: 320px;
    height: 100%;
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 16px rgba(0,0,0,0.20);
    transition: border-left-color 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease;
}}
.strategy-card:hover {{
    border-left-color: {OAK_GOLD};
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.28);
}}
.strategy-card h3 {{
    color: {OAK_CREAM} !important;
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-size: 26px !important; font-weight: 500 !important;
    letter-spacing: -0.01em; margin: 0 0 6px 0;
    text-transform: none; border: none; padding: 0;
}}
.strategy-card .strat-tag {{
    color: {OAK_SAGE}; font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.14em; font-weight: 600; margin-bottom: 18px;
    display: block;
}}
.strategy-card p {{
    color: {OAK_CREAM_DIM} !important; font-size: 14px;
    line-height: 1.6; margin-bottom: 16px;
}}
.strategy-card .strat-meta {{
    color: {OAK_SAGE_DIM}; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.1em;
    padding-top: 16px; margin-top: auto;
    border-top: 1px solid {OAK_GREEN_3};
}}
.strategy-card .strat-meta strong {{ color: {OAK_CREAM}; }}

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

/* ---- Visibility fixes (scrollbar fallback; arrow handled by config.toml theme) ---- */
::-webkit-scrollbar {{ width: 11px; height: 11px; }}
::-webkit-scrollbar-track {{ background: {OAK_GREEN_2}; }}
::-webkit-scrollbar-thumb {{
    background: {OAK_SAGE_DIM}; border-radius: 8px;
    border: 2px solid {OAK_GREEN_2};
}}
::-webkit-scrollbar-thumb:hover {{ background: {OAK_GOLD}; }}
html, body {{ scrollbar-color: {OAK_SAGE_DIM} {OAK_GREEN_2}; scrollbar-width: thin; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Brand bar
if logo_b64:
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="Oakwood Capital"/>'
else:
    logo_html = f'<span style="color:{OAK_CREAM}; font-family:Cormorant Garamond, serif; font-size:28px;">Oakwood Capital</span>'

st.markdown(f"""
<div class="oak-bar">
    <div class="oak-logo">{logo_html}</div>
    <div class="oak-tagline">
        Strategie-Research-Plattform
        <span class="stamp">Intern · Vertraulich</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero
st.markdown(
    f"<h1 style='color:{OAK_CREAM}; font-family:\"Cormorant Garamond\", Georgia, serif; "
    f"font-weight:500; font-size:48px; letter-spacing:-0.01em; margin:8px 0 8px 0; "
    f"line-height:1.1;'>Quantitativer Strategie-Backtester</h1>",
    unsafe_allow_html=True
)
st.markdown(
    f"<p style='color:{OAK_CREAM_DIM}; font-size:16px; margin-top:0; max-width: 760px;'>"
    "Backtest-Werkzeuge auf institutionellem Niveau für systematische Strategien, "
    "die Schweizer Ertragswerte — Blue-Chip-Aktien und Wohnimmobilien — mit einer "
    "strukturellen Digital-Asset-Allokation verbinden. "
    "Wählen Sie unten eine Strategie oder nutzen Sie die Navigation in der Seitenleiste."
    "</p>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ---- The framework: one method, several applications -----------------------
st.markdown(
    f"<h3 style='color:{OAK_CREAM} !important; font-family:Inter, sans-serif; "
    f"font-size:13px; text-transform:uppercase; letter-spacing:0.12em; "
    f"font-weight:600; padding-bottom:6px; "
    f"border-bottom:1px solid {OAK_GREEN_3}; margin-bottom:20px;'>Das Framework</h3>",
    unsafe_allow_html=True
)
st.markdown(
    f"""
<div style="border:1px solid {OAK_BORDER}; border-left:3px solid {OAK_GOLD};
            border-radius:6px; padding:22px 26px; margin-bottom:34px;
            background:{OAK_GREEN_2};">
  <div style="font-family:'Cormorant Garamond', Georgia, serif; color:{OAK_CREAM};
              font-size:30px; font-weight:500; letter-spacing:-0.01em;
              margin-bottom:6px;">OAK Yield Bridge</div>
  <div style="color:{OAK_GOLD}; font-family:Inter, sans-serif; font-size:11px;
              text-transform:uppercase; letter-spacing:0.12em; font-weight:600;
              margin-bottom:14px;">Yield-Funded Allocation</div>
  <p style="color:{OAK_CREAM_DIM}; font-size:15px; line-height:1.65;
            max-width:820px; margin:0 0 16px 0;">
    Ein regelbasiertes, prognosefreies Allokations-Framework. Ein Schweizer
    Substanz- oder Blue-Chip-<strong style="color:{OAK_CREAM};">Kern</strong> bleibt
    unangetastet und produziert einen
    <strong style="color:{OAK_CREAM};">Ertragsstrom</strong> — Mieten oder Dividenden.
    Nur dieser Ertrag finanziert eine
    <strong style="color:{OAK_CREAM};">Satelliten</strong>-Allokation, die über
    antizyklische Bänder mit hartem Cap gesteuert wird. Keine Prognosen, kein
    Market-Timing, keine Optimierung gefitteter Parameter.
  </p>
  <div style="color:{OAK_SAGE}; font-family:Inter, sans-serif; font-size:13px;
              line-height:1.9;">
    <strong style="color:{OAK_CREAM};">Kern</strong> → erzeugt Ertrag ·
    <strong style="color:{OAK_CREAM};">Ertrag</strong> → finanziert den Satelliten ·
    <strong style="color:{OAK_CREAM};">Bänder</strong> → steuern das Tempo und begrenzen das Risiko<br>
    <span style="color:{OAK_CREAM_DIM};">Das Kernkapital wird nie verkauft.
    Die beiden Strategien unten sind zwei Anwendungen derselben Methode.</span>
  </div>
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    f"<h3 style='color:{OAK_CREAM} !important; font-family:Inter, sans-serif; "
    f"font-size:13px; text-transform:uppercase; letter-spacing:0.12em; "
    f"font-weight:600; padding-bottom:6px; "
    f"border-bottom:1px solid {OAK_GREEN_3}; margin-bottom:20px;'>Strategien</h3>",
    unsafe_allow_html=True
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(f"""
<div class="strategy-card">
    <span class="strat-tag">Strategie 01 · Kern: Schweizer Blue Chips</span>
    <h3>OAK Swiss Blue Chip / Bitcoin</h3>
    <p>Vollständige SMI-Replikation mit 20 Schweizer Blue Chips. Die Dividenden
    werden vereinnahmt und über einen 12-Monats-DCA in Bitcoin investiert. Eine
    Threshold-Regel begrenzt die BTC-Quote zur Risikokontrolle.</p>
    <div class="strat-meta">
        Universum: <strong>SMI 20</strong> ·
        Gewichtung: <strong>Marktkapitalisierung</strong> ·
        BTC: <strong>15 % → Cap 25 %</strong>
    </div>
</div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if hasattr(st, "page_link"):
        st.page_link("pages/1_SMI_Strategy.py", label="Swiss Blue Chip / Bitcoin öffnen →",
                     use_container_width=True)

with col2:
    st.markdown(f"""
<div class="strategy-card">
    <span class="strat-tag">Strategie 02 · Kern: Schweizer Wohnimmobilien</span>
    <h3>OAK Swiss Residential / Bitcoin</h3>
    <p>Schweizer Wohnimmobilien, deren Kapitalwerte dem SNB-Wohnimmobilien-
    preisindex folgen. Die Nettomieterträge fliessen über Bandregeln in Bitcoin;
    ein wachsender CHF-Cash-Puffer dämpft die Volatilität.
    Parametrische Simulation.</p>
    <div class="strat-meta">
        Universum: <strong>CH Wohnimmobilien (SNB)</strong> ·
        Mietallokation: <strong>Bandregeln</strong> ·
        BTC: <strong>Band 10–25 %</strong>
    </div>
</div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if hasattr(st, "page_link"):
        st.page_link("pages/2_OAK_RE_BTC.py", label="Swiss Residential / Bitcoin öffnen →",
                     use_container_width=True)

# Footer
st.markdown(
    f"""<div class='oak-footer'>
    Zu illustrativen Zwecken · Keine Anlageberatung · Vergangene Wertentwicklung ist kein Indikator für zukünftige Ergebnisse
    <span class='oak-mark'>Oakwood Capital · Quantitatives Research</span>
    </div>""", unsafe_allow_html=True
)
