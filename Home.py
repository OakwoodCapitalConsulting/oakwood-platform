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
    height: 100%;
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
    padding-top: 16px; margin-top: 16px;
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
        Strategy Research Platform
        <span class="stamp">Internal · Confidential</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero
st.markdown(
    f"<h1 style='color:{OAK_CREAM}; font-family:\"Cormorant Garamond\", Georgia, serif; "
    f"font-weight:500; font-size:48px; letter-spacing:-0.01em; margin:8px 0 8px 0; "
    f"line-height:1.1;'>Quantitative Strategy Backtester</h1>",
    unsafe_allow_html=True
)
st.markdown(
    f"<p style='color:{OAK_CREAM_DIM}; font-size:16px; margin-top:0; max-width: 760px;'>"
    "Institutional-grade backtest tooling for a systematic strategy combining "
    "Swiss equity income with structural digital-asset exposure. "
    "Open the strategy below or use the sidebar navigation."
    "</p>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f"<h3 style='color:{OAK_CREAM} !important; font-family:Inter, sans-serif; "
    f"font-size:13px; text-transform:uppercase; letter-spacing:0.12em; "
    f"font-weight:600; padding-bottom:6px; "
    f"border-bottom:1px solid {OAK_GREEN_3}; margin-bottom:20px;'>Strategy</h3>",
    unsafe_allow_html=True
)

col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown(f"""
<div class="strategy-card">
    <span class="strat-tag">Strategy 01 · Broad Index</span>
    <h3>SMI Income meets Digital Assets</h3>
    <p>Full SMI replication with 20 Swiss blue-chip constituents. Dividends harvested
    and reinvested via 12-month DCA into Bitcoin exposure. Threshold-based rebalancing
    caps BTC weight to control risk.</p>
    <div class="strat-meta">
        Universe: <strong>SMI 20</strong> ·
        Weighting: <strong>Market Cap</strong> ·
        BTC: <strong>15 % → 25 % cap</strong>
    </div>
</div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if hasattr(st, "page_link"):
        st.page_link("pages/1_SMI_Strategy.py", label="Open SMI Strategy →",
                     use_container_width=True)

# Footer
st.markdown(
    f"""<div class='oak-footer'>
    For Illustrative Purposes · Not Investment Advice · Past Performance is no Guarantee of Future Results
    <span class='oak-mark'>Oakwood Capital · Quantitative Research</span>
    </div>""", unsafe_allow_html=True
)
