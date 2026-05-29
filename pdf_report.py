"""
Oakwood Capital — PDF Tearsheet Generator
==========================================
Builds a multi-page institutional-style PDF report from backtest results.
Used by the SMI strategy page. Returns PDF bytes for st.download_button.

Charts are rendered to PNG via matplotlib (headless, no browser needed).
Requires: reportlab, matplotlib.
"""

import io
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless backend — no browser, stable on any server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, CondPageBreak, HRFlowable, KeepTogether,
)

# Matplotlib palette matching Oakwood CI
MPL_GREEN = "#293624"
MPL_SAGE  = "#99A796"
MPL_GOLD  = "#C9A961"
MPL_CREAM = "#F5F5F1"
MPL_BTC   = "#F7931A"
MPL_RED   = "#B85042"
MPL_GRID  = "#D8DCD3"


# ---------------------------------------------------------------------------
# Font registration — embed Crimson Pro (serif display) + Work Sans (grotesk).
# Falls back to the built-in Times/Helvetica if the TTFs aren't found, so the
# PDF never fails to build.
# ---------------------------------------------------------------------------
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Logical font names used throughout the report, mapped to either the embedded
# families or the safe built-in fallbacks.
F_SERIF = "Times-Roman"
F_SERIF_BOLD = "Times-Bold"
F_SERIF_ITALIC = "Times-Italic"
F_SANS = "Helvetica"
F_SANS_BOLD = "Helvetica-Bold"
F_SANS_ITALIC = "Helvetica-Oblique"

_FONTS_REGISTERED = False


def _register_fonts():
    global _FONTS_REGISTERED, F_SERIF, F_SERIF_BOLD, F_SERIF_ITALIC
    global F_SANS, F_SANS_BOLD, F_SANS_ITALIC
    if _FONTS_REGISTERED:
        return
    fdir = os.path.join(os.path.dirname(__file__), "assets", "fonts")
    mapping = [
        ("CrimsonPro", "CrimsonPro-Regular.ttf", "CrimsonPro-Bold.ttf", "CrimsonPro-Italic.ttf"),
        ("WorkSans", "WorkSans-Regular.ttf", "WorkSans-Bold.ttf", "WorkSans-Italic.ttf"),
    ]
    try:
        ok = {}
        for family, reg, bold, ital in mapping:
            rp = os.path.join(fdir, reg)
            bp = os.path.join(fdir, bold)
            ip = os.path.join(fdir, ital)
            if os.path.exists(rp):
                pdfmetrics.registerFont(TTFont(family, rp))
                if os.path.exists(bp):
                    pdfmetrics.registerFont(TTFont(family + "-Bold", bp))
                if os.path.exists(ip):
                    pdfmetrics.registerFont(TTFont(family + "-Italic", ip))
                ok[family] = True
        if ok.get("CrimsonPro"):
            F_SERIF = "CrimsonPro"
            F_SERIF_BOLD = "CrimsonPro-Bold"
            F_SERIF_ITALIC = "CrimsonPro-Italic"
        if ok.get("WorkSans"):
            F_SANS = "WorkSans"
            F_SANS_BOLD = "WorkSans-Bold"
            F_SANS_ITALIC = "WorkSans-Italic"
    except Exception:
        pass  # keep built-in fallbacks
    _FONTS_REGISTERED = True


def render_line_chart(series_dict, title="", ylabel="", percent=False, fill_first=False,
                      annotate_end=False, crisis_phases=None):
    """Render a line chart to PNG bytes using matplotlib (no browser needed).
    series_dict: ordered dict-like list of (label, pandas Series, color, style).
    annotate_end: if True, label the final value at the end of each line.
    crisis_phases: optional list of (start_date, end_date, label) to shade."""
    try:
        plt.rcParams["font.family"] = "DejaVu Sans"
        PANEL = "#FBFBF8"   # matches C_PAGE_BG — chart blends into the light page
        fig, ax = plt.subplots(figsize=(9.5, 4.0), dpi=170)
        fig.patch.set_facecolor(PANEL)
        ax.set_facecolor(PANEL)
        # Crisis-phase shading (drawn first, behind the lines)
        if crisis_phases:
            import pandas as _pd
            for cp in crisis_phases:
                try:
                    cs, ce, clabel = cp
                    ax.axvspan(_pd.Timestamp(cs), _pd.Timestamp(ce),
                               color="#B85042", alpha=0.07, zorder=0)
                    ax.annotate(clabel, xy=(_pd.Timestamp(cs), 0),
                                xytext=(2, 6), textcoords="offset points",
                                fontsize=6, color="#B07A6E", va="bottom", ha="left",
                                rotation=0, zorder=1)
                except Exception:
                    pass
        for i, (label, s, color, style) in enumerate(series_dict):
            if s is None or len(s) == 0:
                continue
            vals = s.values * 100 if percent else s.values
            ax.plot(s.index, vals, label=label, color=color,
                    linewidth=style.get("lw", 1.8),
                    linestyle=style.get("ls", "-"),
                    alpha=style.get("alpha", 1.0),
                    solid_capstyle="round")
            if fill_first and i == 0:
                ax.fill_between(s.index, vals, alpha=0.10, color=color, linewidth=0)
            # Endpoint value annotation (IB-style)
            if annotate_end and len(vals) > 0:
                end_val = vals[-1]
                if percent:
                    txt = f"{end_val:.1f}%"
                elif end_val >= 1e6:
                    txt = f"{end_val/1e6:.2f}M"
                else:
                    txt = f"{end_val:,.0f}"
                # Stagger labels vertically so series with very close end values
                # (e.g. Strategy vs Benchmark) don't overprint each other.
                _n = len(series_dict)
                if _n > 1:
                    _y_off = (i - (_n - 1) / 2.0) * 9
                else:
                    _y_off = 0
                ax.annotate(txt, xy=(s.index[-1], end_val),
                            xytext=(6, _y_off), textcoords="offset points",
                            fontsize=7.5, color=color, va="center", ha="left",
                            fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8.5, color="#6B7868", labelpad=8)
        ax.tick_params(labelsize=8, colors="#6B7868", length=0)
        ax.tick_params(axis="both", labelcolor="#6B7868")
        ax.grid(True, color="#E2E4DD", linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#D2D5CC")
        # Add right margin so endpoint labels aren't clipped
        if annotate_end:
            ax.margins(x=0.06)
        if percent:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        leg = ax.legend(fontsize=8, loc="lower left", frameon=False,
                        bbox_to_anchor=(0, 1.01), handlelength=1.6,
                        handletextpad=0.6, columnspacing=1.4,
                        ncol=len(series_dict) if len(series_dict) <= 3 else 2)
        if leg:
            for text in leg.get_texts():
                text.set_color("#2A2A26")
        try:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        except Exception:
            pass
        fig.tight_layout(pad=1.0)
        bio = io.BytesIO()
        fig.savefig(bio, format="png", facecolor=PANEL, bbox_inches="tight", dpi=170)
        plt.close(fig)
        bio.seek(0)
        return bio.getvalue()
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def render_bar_chart(x_labels, values, title="", ylabel="", hurdle=None):
    """Render a bar chart (e.g. yearly returns) to PNG bytes."""
    try:
        plt.rcParams["font.family"] = "DejaVu Sans"
        PANEL = "#FBFBF8"   # matches C_PAGE_BG
        fig, ax = plt.subplots(figsize=(9.5, 4.7), dpi=170)
        fig.patch.set_facecolor(PANEL)
        ax.set_facecolor(PANEL)
        bar_colors = [MPL_SAGE if v >= 0 else "#B85042" for v in values]
        bars = ax.bar(range(len(values)), values, color=bar_colors,
                      width=0.58, edgecolor="none", zorder=3)
        # Value labels above/below each bar
        for rect, v in zip(bars, values):
            ax.annotate(f"{v:+.1f}%", xy=(rect.get_x() + rect.get_width()/2,
                        rect.get_height()),
                        xytext=(0, 5 if v >= 0 else -13), textcoords="offset points",
                        ha="center", fontsize=8, color="#2A2A26", zorder=4)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=8.5, color="#6B7868")
        ax.set_ylabel(ylabel, fontsize=8.5, color="#6B7868", labelpad=8)
        ax.tick_params(labelsize=8, colors="#6B7868", length=0)
        ax.grid(True, axis="y", color="#E2E4DD", linewidth=0.6, alpha=0.9, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#D2D5CC")
        ax.axhline(0, color="#8A9584", linewidth=0.8, zorder=2)
        # Add headroom so value labels don't collide with the top
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - abs(ymin)*0.14, ymax + abs(ymax)*0.18)
        if hurdle is not None:
            ax.axhline(hurdle, color=MPL_GOLD, linewidth=1.3, linestyle="--",
                       label=f"Year-1 Performance-Fee Hurdle ({hurdle:.0f}%)", zorder=2)
            leg = ax.legend(fontsize=8, frameon=False, loc="lower left",
                            bbox_to_anchor=(0, 1.01))
            if leg:
                for t in leg.get_texts():
                    t.set_color("#2A2A26")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        # Footnote if any partial-year labels are present
        if any("*" in str(lbl) for lbl in x_labels):
            ax.annotate("* Partial year — backtest does not span the full calendar year.",
                        xy=(0, 0), xytext=(0, -34), textcoords="offset points",
                        xycoords="axes fraction", fontsize=6.5, color="#9AA595",
                        ha="left", va="top")
        fig.tight_layout(pad=1.0)
        bio = io.BytesIO()
        fig.savefig(bio, format="png", facecolor=PANEL, bbox_inches="tight", dpi=170)
        plt.close(fig)
        bio.seek(0)
        return bio.getvalue()
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def render_scatter_chart(points, xlabel="Volatility (ann.)", ylabel="CAGR (ann.)"):
    """Risk/return scatter: each point is (label, vol%, ret%, color, marker).
    Shows where the strategy sits versus benchmarks on a risk/return plane."""
    try:
        plt.rcParams["font.family"] = "DejaVu Sans"
        PANEL = "#FBFBF8"
        fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=170)
        fig.patch.set_facecolor(PANEL)
        ax.set_facecolor(PANEL)
        vols = [p[1] for p in points]
        rets = [p[2] for p in points]
        # Axis ranges: padded window around the data (not forced to zero) so
        # closely-clustered points are still distinguishable.
        xlo, xhi = min(vols), max(vols)
        ylo, yhi = min(rets), max(rets)
        xpad = max((xhi - xlo) * 0.45, 1.5)
        ypad = max((yhi - ylo) * 0.45, 1.5)
        x0, x1 = max(0, xlo - xpad), xhi + xpad
        y0, y1 = max(0, ylo - ypad), yhi + ypad
        # Iso-Sharpe reference lines through the window — only draw those
        # whose endpoint actually fits inside the view, and anchor the label
        # just inside the right edge so it never gets clipped.
        for sharpe in (0.5, 0.75, 1.0):
            y_at_x1 = sharpe * x1
            if y_at_x1 < y0 or sharpe * x0 > y1:
                continue
            ax.plot([x0, x1], [sharpe * x0, sharpe * x1],
                    color="#D2D5CC", linewidth=0.7, linestyle=":", zorder=1)
            # Place the label where the line intersects either the right edge
            # or the top edge — whichever is reached first inside the plot.
            if y_at_x1 <= y1:
                lx, ly = x1, y_at_x1
                ha, va = "right", "bottom"
                offset = (-4, 2)
            else:
                lx, ly = (y1 / sharpe), y1
                ha, va = "left", "top"
                offset = (4, -2)
            ax.annotate(f"Sharpe {sharpe:g}", xy=(lx, ly),
                        xytext=offset, textcoords="offset points",
                        fontsize=6.5, color="#9AA595", va=va, ha=ha,
                        zorder=2)
        # Alternate label placement to avoid overlap
        for idx, (label, vol, ret, color, marker) in enumerate(points):
            ax.scatter([vol], [ret], s=210, c=color, marker=marker,
                       edgecolors="white", linewidths=1.3, zorder=4)
            dy = 13 if idx % 2 == 0 else -16
            ax.annotate(label, xy=(vol, ret), xytext=(0, dy),
                        textcoords="offset points", fontsize=8.5,
                        color="#2A2A26", va="center", ha="center", zorder=5,
                        fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8.5, color="#6B7868", labelpad=8)
        ax.set_ylabel(ylabel, fontsize=8.5, color="#6B7868", labelpad=8)
        ax.tick_params(labelsize=8, colors="#6B7868", length=0)
        ax.grid(True, color="#E2E4DD", linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#D2D5CC")
        ax.spines["bottom"].set_color("#D2D5CC")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        fig.tight_layout(pad=1.0)
        bio = io.BytesIO()
        fig.savefig(bio, format="png", facecolor=PANEL, bbox_inches="tight", dpi=170)
        plt.close(fig)
        bio.seek(0)
        return bio.getvalue()
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Data-prep utilities — pure functions on pandas Series. Live in pdf_report
# so that 1_SMI_Strategy.py can `from pdf_report import compute_period_returns,
# identify_top_drawdowns` without duplicating the logic.
# ---------------------------------------------------------------------------

def compute_period_returns(strat, bench=None):
    """Compute trailing returns over standard horizons (1M, 3M, 6M, YTD, 1Y,
    3Y ann., ITD ann.) for a NAV series and an optional benchmark NAV series.

    Returns a list of (label_key, strategy_pct, benchmark_pct, excess_pct)
    tuples. label_key matches STRINGS keys ('pr_1m', 'pr_3m', ...).
    Missing periods (e.g. no full 3Y of data yet) are silently skipped.
    """
    import pandas as _pd
    import numpy as _np

    if strat is None or len(strat) < 2:
        return []
    strat = strat.dropna()
    if bench is not None:
        bench = bench.dropna()
        # Align on the union of timestamps actually shared
        common = strat.index.intersection(bench.index)
        strat = strat.loc[common]
        bench = bench.loc[common]
    asof = strat.index[-1]
    inception = strat.index[0]
    years_total = (asof - inception).days / 365.25

    def _ret(series, lookback_days=None, ytd=False, ann_years=None):
        """Return percent gain over the lookback window ending at asof."""
        if series is None or series.empty:
            return None
        if ytd:
            start_of_year = _pd.Timestamp(year=asof.year, month=1, day=1)
            window = series.loc[series.index >= start_of_year]
            if len(window) < 2:
                return None
            return (window.iloc[-1] / window.iloc[0] - 1) * 100.0
        if lookback_days is not None:
            cutoff = asof - _pd.Timedelta(days=lookback_days)
            if series.index[0] > cutoff:
                return None  # not enough history
            window = series.loc[series.index >= cutoff]
            if len(window) < 2:
                return None
            total = window.iloc[-1] / window.iloc[0] - 1
            if ann_years is not None and ann_years > 0:
                if (1 + total) <= 0:
                    return None
                return ((1 + total) ** (1.0 / ann_years) - 1) * 100.0
            return total * 100.0
        return None

    # Definitions: (label_key, lookback_days, ytd, ann_years_for_annualizing)
    periods = [
        ("pr_1m",  30,    False, None),
        ("pr_3m",  91,    False, None),
        ("pr_6m",  182,   False, None),
        ("pr_ytd", None,  True,  None),
        ("pr_1y",  365,   False, None),
        ("pr_3y",  365*3, False, 3.0),
    ]
    rows = []
    for key, lb, ytd, ann in periods:
        s = _ret(strat, lookback_days=lb, ytd=ytd, ann_years=ann)
        if s is None:
            continue
        b = _ret(bench, lookback_days=lb, ytd=ytd, ann_years=ann) if bench is not None else None
        excess = (s - b) if (b is not None) else None
        rows.append((key, s, b, excess))
    # Since inception — annualized
    if years_total > 0.05:
        s_itd_total = strat.iloc[-1] / strat.iloc[0] - 1
        s_itd = ((1 + s_itd_total) ** (1.0 / years_total) - 1) * 100.0 if (1 + s_itd_total) > 0 else None
        b_itd = None
        if bench is not None and len(bench) >= 2:
            b_itd_total = bench.iloc[-1] / bench.iloc[0] - 1
            b_itd = ((1 + b_itd_total) ** (1.0 / years_total) - 1) * 100.0 if (1 + b_itd_total) > 0 else None
        excess = (s_itd - b_itd) if (s_itd is not None and b_itd is not None) else None
        rows.append(("pr_itd", s_itd, b_itd, excess))
    return rows


def identify_top_drawdowns(nav, n=5, min_depth_pct=2.0):
    """Identify the top-N drawdown episodes by depth, with start, trough,
    end and recovery characteristics.

    A drawdown episode runs from a new running-maximum until the NAV
    recovers back to that maximum. The final, still-open episode is
    flagged via end=None and recovery_days=None.

    Returns a list of dicts:
        [{"start": Timestamp, "trough": Timestamp, "end": Timestamp|None,
          "depth_pct": float (negative), "duration_days": int,
          "recovery_days": int|None}, ...]
    sorted by depth (deepest first), truncated to n.
    """
    import pandas as _pd

    if nav is None or len(nav) < 3:
        return []
    s = nav.dropna()
    if s.empty:
        return []
    peak = s.cummax()
    dd = (s / peak - 1.0)  # values in [-1, 0]

    episodes = []
    in_dd = False
    start_idx = None
    trough_idx = None
    trough_val = 0.0
    peak_val = None
    for i, (t, v) in enumerate(dd.items()):
        if not in_dd:
            if v < 0:
                in_dd = True
                start_idx = t
                trough_idx = t
                trough_val = v
                peak_val = peak.iloc[i]
        else:
            if v < trough_val:
                trough_val = v
                trough_idx = t
            if v >= 0 - 1e-12:  # recovered
                episodes.append({
                    "start": start_idx, "trough": trough_idx, "end": t,
                    "depth_pct": trough_val * 100.0,
                    "duration_days": (t - start_idx).days,
                    "recovery_days": (t - trough_idx).days,
                    "peak_value": peak_val,
                })
                in_dd = False
                start_idx = trough_idx = None
                trough_val = 0.0
    # Final unresolved episode
    if in_dd and start_idx is not None:
        episodes.append({
            "start": start_idx, "trough": trough_idx, "end": None,
            "depth_pct": trough_val * 100.0,
            "duration_days": (s.index[-1] - start_idx).days,
            "recovery_days": None,
            "peak_value": peak_val,
        })

    # Filter trivial, sort by depth (most negative first)
    episodes = [e for e in episodes if abs(e["depth_pct"]) >= min_depth_pct]
    episodes.sort(key=lambda e: e["depth_pct"])
    return episodes[:n]


# ---------------------------------------------------------------------------
C_PAGE_BG = colors.HexColor("#FBFBF8")    # warm off-white — page background
C_PANEL   = colors.HexColor("#F4F4EF")    # very light panel for cards
C_PANEL_2 = colors.HexColor("#293624")    # deep green for panel/table headers
C_GREEN   = colors.HexColor("#293624")    # deep green (headings, header band)
C_GREEN2  = colors.HexColor("#1F2A1B")    # darker green
C_SAGE    = colors.HexColor("#7C8978")    # muted sage for labels on light
C_SAGE_DIM= colors.HexColor("#9AA595")    # lighter sage for fine print
C_CREAM   = colors.HexColor("#F5F5F1")    # light text (on green bands)
C_CREAMD  = colors.HexColor("#E8E8E2")    # secondary light text
C_GOLD    = colors.HexColor("#B8954A")    # gold accent (slightly deeper for light bg)
C_BORDER  = colors.HexColor("#DCDDD6")    # subtle borders on light
C_TEXT    = colors.HexColor("#2A2A26")    # primary dark body text
C_MUTED   = colors.HexColor("#6B7868")    # muted dark labels
C_RED     = colors.HexColor("#B85042")    # red for negatives
C_GREEN_POS = colors.HexColor("#5F7C4F")  # muted positive green (for + values)
C_TABLE_ALT = colors.HexColor("#EFEFE7")  # warmer table row stripe (vs C_PAGE_BG)
C_WHITE   = colors.white

# ---------------------------------------------------------------------------
# Bilingual string table. All static UI text the tearsheet renders flows
# through this dict so we can build either an EN or DE version of the same
# PDF. German values can be filled in a follow-up — the EN keys are the
# canonical set and act as a fallback.
STRINGS = {
    "en": {
        # Header / footer
        "platform":              "STRATEGY RESEARCH PLATFORM",
        "confidential":          "INTERNAL · CONFIDENTIAL",
        "illustrative":          "For Illustrative Purposes · Not Investment Advice",
        "cover_tagline":         "Strategy Research Platform · Internal · Confidential",
        "cover_byline":          "Oakwood Capital · Quantitative Research",
        "cover_period":          "Backtest Period",
        "cover_generated":       "Generated",
        # Section titles
        "exec_summary":          "Executive Summary",
        "key_takeaways":         "KEY TAKEAWAYS",
        "perf_summary":          "Performance Summary",
        "perf_summary_sub":      "Net of fees, transaction costs and 35% dividend withholding tax",
        "risk_metrics":          "Risk &amp; Risk-Adjusted Metrics",
        "fee_summary":           "Fee Summary",
        "snapshot":              "Strategy Snapshot",
        "period_returns":        "Performance per Period",
        "period_returns_sub":    "Cumulative net returns over standard reporting horizons",
        "charts":                "Portfolio Evolution &amp; Risk Charts",
        "evolution":             "Portfolio Evolution vs. Benchmarks",
        "drawdown":              "Drawdown Analysis",
        "yearly":                "Yearly Net Performance",
        "scatter":               "Risk / Return Positioning",
        "monthly_returns":       "Monthly Returns (Net, %)",
        "monthly_returns_sub":   "Green = positive, red = negative; intensity scales with magnitude. FY = compounded full-year return.",
        "detailed_risk":         "Detailed Risk Metrics",
        "top_drawdowns":         "Top 5 Drawdowns",
        "top_drawdowns_sub":     "Largest peak-to-trough declines over the backtest period",
        "perf_fee_crystal":      "Performance Fee Crystallization Detail",
        "methodology":           "Methodology &amp; Parameters",
        "universe":              "Investment Universe",
        "disclosures":           "Important Disclosures",
        # Snapshot labels
        "sn_inception":          "Inception",
        "sn_currency":           "Currency",
        "sn_benchmark":          "Benchmark",
        "sn_style":              "Investment Style",
        "sn_domicile":           "Domicile",
        "sn_frequency":          "Reporting Frequency",
        "sn_riskprofile":        "Risk Profile",
        "sn_strategyaum":        "Strategy AUM",
        # Period returns labels
        "pr_period":             "Period",
        "pr_strategy":           "Strategy (Net)",
        "pr_benchmark":          "SMI Total Return",
        "pr_excess":             "Excess",
        "pr_1m":                 "1 Month",
        "pr_3m":                 "3 Months",
        "pr_6m":                 "6 Months",
        "pr_ytd":                "Year-to-Date",
        "pr_1y":                 "1 Year",
        "pr_3y":                 "3 Years (ann.)",
        "pr_itd":                "Since Inception (ann.)",
        # Top drawdowns labels
        "dd_start":              "Start",
        "dd_trough":             "Trough",
        "dd_end":                "End",
        "dd_duration":           "Duration",
        "dd_depth":              "Depth",
        "dd_recovery":           "Recovery",
        "dd_days":               "days",
        "dd_ongoing":            "ongoing",
        # Universe labels
        "uni_constituent":       "Constituent",
        "uni_ticker":            "Ticker",
        "uni_sector":            "Sector",
        "uni_weight":            "Weight",
        # Methodology table headers + perf-fee fallback
        "param_label":           "Parameter",
        "param_value":           "Value",
        "no_perf_fee":           "No performance fees were crystallized in this period.",
        # Cover italic tagline (between hero KPIs and date block)
        "cover_strapline":       "Where Swiss discipline meets digital innovation.",
        # Editorial section eyebrows — small-caps numbered chapter markers
        "eyebrow_01":            "OVERVIEW",
        "eyebrow_02":            "PERFORMANCE",
        "eyebrow_03":            "EVOLUTION",
        "eyebrow_04":            "RISK",
        "eyebrow_05":            "FEES",
        "eyebrow_06":            "UNIVERSE",
        "eyebrow_07":            "DISCLOSURES",
    },
    # German values are intentionally left empty — Phase 2 fills them.
    # During Phase 1, lang="de" falls back to the EN values automatically.
    "de": {
        # Header / footer
        "platform":              "STRATEGIE-RESEARCH PLATTFORM",
        "confidential":          "INTERN · VERTRAULICH",
        "illustrative":          "Zu illustrativen Zwecken · Keine Anlageberatung",
        "cover_tagline":         "Strategie-Research Plattform · Intern · Vertraulich",
        "cover_byline":          "Oakwood Capital · Quantitatives Research",
        "cover_period":          "Backtest-Zeitraum",
        "cover_generated":       "Erstellt",
        # Section titles
        "exec_summary":          "Zusammenfassung",
        "key_takeaways":         "KERNAUSSAGEN",
        "perf_summary":          "Performance-Übersicht",
        "perf_summary_sub":      "Nach Gebühren, Transaktionskosten und 35% Dividenden-Quellensteuer",
        "risk_metrics":          "Risiko und risikoadjustierte Kennzahlen",
        "fee_summary":           "Gebührenübersicht",
        "snapshot":              "Strategie-Eckdaten",
        "period_returns":        "Performance nach Zeitraum",
        "period_returns_sub":    "Kumulierte Nettorenditen über Standard-Reporting-Zeiträume",
        "charts":                "Portfolioentwicklung &amp; Risikoanalyse",
        "evolution":             "Portfolioentwicklung vs. Benchmarks",
        "drawdown":              "Drawdown-Analyse",
        "yearly":                "Jährliche Netto-Performance",
        "scatter":               "Risiko-Rendite-Positionierung",
        "monthly_returns":       "Monatsrenditen (Netto, %)",
        "monthly_returns_sub":   "Grün = positiv, rot = negativ; Intensität skaliert mit Magnitude. GJ = kumulierte Jahresrendite.",
        "detailed_risk":         "Detaillierte Risikokennzahlen",
        "top_drawdowns":         "Top 5 Drawdowns",
        "top_drawdowns_sub":     "Grösste Peak-to-Trough-Verluste im Backtest-Zeitraum",
        "perf_fee_crystal":      "Performance-Fee-Kristallisationsdetails",
        "methodology":           "Methodik &amp; Parameter",
        "universe":              "Anlageuniversum",
        "disclosures":           "Wichtige Hinweise",
        # Snapshot labels
        "sn_inception":          "Auflagedatum",
        "sn_currency":           "Währung",
        "sn_benchmark":          "Benchmark",
        "sn_style":              "Anlagestil",
        "sn_domicile":           "Domizil",
        "sn_frequency":          "Reporting-Frequenz",
        "sn_riskprofile":        "Risikoprofil",
        "sn_strategyaum":        "Strategie-Volumen",
        # Period returns labels
        "pr_period":             "Zeitraum",
        "pr_strategy":           "Strategie (Netto)",
        "pr_benchmark":          "SMI Total Return",
        "pr_excess":             "Excess",
        "pr_1m":                 "1 Monat",
        "pr_3m":                 "3 Monate",
        "pr_6m":                 "6 Monate",
        "pr_ytd":                "Lfd. Jahr",
        "pr_1y":                 "1 Jahr",
        "pr_3y":                 "3 Jahre (ann.)",
        "pr_itd":                "Seit Auflage (ann.)",
        # Top drawdowns labels
        "dd_start":              "Beginn",
        "dd_trough":             "Tiefpunkt",
        "dd_end":                "Ende",
        "dd_duration":           "Dauer",
        "dd_depth":              "Tiefe",
        "dd_recovery":           "Erholung",
        "dd_days":               "Tage",
        "dd_ongoing":            "laufend",
        # Universe labels
        "uni_constituent":       "Titel",
        "uni_ticker":            "Ticker",
        "uni_sector":            "Sektor",
        "uni_weight":            "Gewicht",
        # Methodology table headers + perf-fee fallback
        "param_label":           "Parameter",
        "param_value":           "Wert",
        "no_perf_fee":           "In diesem Zeitraum wurden keine Performance-Gebühren kristallisiert.",
        # Cover italic tagline (between hero KPIs and date block)
        "cover_strapline":       "Wo Schweizer Disziplin auf digitale Innovation trifft.",
        # Editorial section eyebrows — small-caps numbered chapter markers
        "eyebrow_01":            "ÜBERSICHT",
        "eyebrow_02":            "PERFORMANCE",
        "eyebrow_03":            "ENTWICKLUNG",
        "eyebrow_04":            "RISIKO",
        "eyebrow_05":            "GEBÜHREN",
        "eyebrow_06":            "UNIVERSUM",
        "eyebrow_07":            "HINWEISE",
    },
}


# Long-form disclaimer text — kept separate from STRINGS so the dict stays
# tidy. Each language is a list of paragraphs.
DISCLAIMER_PARAGRAPHS = {
    "en": [
        "This document has been prepared by Oakwood Capital for illustrative and "
        "informational purposes only. It does not constitute investment advice, a "
        "recommendation, an offer, or a solicitation to buy or sell any security "
        "or financial instrument, nor a basis for any investment decision.",

        "All figures shown are derived from a historical backtest using publicly "
        "available market data. Backtested performance is hypothetical, does not "
        "represent actual trading, and is subject to the benefit of hindsight. "
        "Past performance and simulated past performance are not reliable indicators "
        "of future results. Actual results may differ materially.",

        "The simulation models transaction costs (a configurable per-trade cost in "
        "basis points) and the 35% Swiss withholding tax on dividends, which is "
        "treated as non-reclaimable within the AMC (Actively Managed Certificate) "
        "wrapper — only the net 65% of each gross dividend is reinvested. The same "
        "after-tax dividend basis is applied to the SMI Total Return benchmark for "
        "a consistent comparison. The simulation does not account for market impact, "
        "slippage, bid-ask spreads, or liquidity constraints. The investment universe "
        "is applied on a current-constituent basis and may be subject to survivorship "
        "bias. Performance figures are shown net of the stated management and "
        "performance fees.",

        "Digital assets such as Bitcoin are highly volatile and may result in the "
        "total loss of capital. Any allocation to digital assets carries substantial "
        "risk and may not be suitable for all investors.",

        "Market data is sourced from third-party providers believed to be reliable "
        "but is not guaranteed as to accuracy or completeness. This material is "
        "strictly confidential and intended solely for the recipient. It may not be "
        "reproduced or distributed, in whole or in part, without the prior written "
        "consent of Oakwood Capital.",
    ],
    "de": [
        "Dieses Dokument wurde von Oakwood Capital ausschliesslich zu illustrativen "
        "und informativen Zwecken erstellt. Es stellt weder eine Anlageberatung, eine "
        "Empfehlung, ein Angebot noch eine Aufforderung zum Kauf oder Verkauf eines "
        "Wertpapiers oder Finanzinstruments dar, noch eine Grundlage für eine "
        "Anlageentscheidung.",

        "Alle dargestellten Zahlen basieren auf einem historischen Backtest unter "
        "Verwendung öffentlich verfügbarer Marktdaten. Die backtestete Performance "
        "ist hypothetisch, stellt keinen tatsächlichen Handel dar und unterliegt dem "
        "Vorteil der Rückschau. Vergangene und simulierte vergangene Wertentwicklungen "
        "sind keine verlässlichen Indikatoren für zukünftige Ergebnisse. Tatsächliche "
        "Resultate können erheblich abweichen.",

        "Die Simulation modelliert Transaktionskosten (eine konfigurierbare Gebühr pro "
        "Trade in Basispunkten) sowie die 35% Schweizer Verrechnungssteuer auf "
        "Dividenden, die innerhalb des AMC-Mantels (Actively Managed Certificate) als "
        "nicht rückforderbar behandelt wird — nur die Netto-65% jeder Bruttodividende "
        "werden reinvestiert. Dieselbe Nach-Steuer-Dividendenbasis wird für einen "
        "konsistenten Vergleich auf den SMI Total Return Benchmark angewendet. Die "
        "Simulation berücksichtigt keine Marktauswirkungen, Slippage, Geld-Brief-"
        "Spannen oder Liquiditätsbeschränkungen. Das Anlageuniversum wird auf Basis "
        "der aktuellen Indexmitglieder angewendet und kann einem Survivorship Bias "
        "unterliegen. Die Performance-Zahlen werden nach Abzug der angegebenen "
        "Management- und Performance-Gebühren ausgewiesen.",

        "Digitale Vermögenswerte wie Bitcoin sind hochvolatil und können zum "
        "Totalverlust des eingesetzten Kapitals führen. Jede Allokation in digitale "
        "Vermögenswerte beinhaltet erhebliche Risiken und ist möglicherweise nicht "
        "für alle Anleger geeignet.",

        "Marktdaten stammen von Drittanbietern, die als verlässlich gelten, deren "
        "Genauigkeit oder Vollständigkeit jedoch nicht garantiert ist. Dieses "
        "Material ist streng vertraulich und ausschliesslich für den Empfänger "
        "bestimmt. Es darf weder ganz noch teilweise ohne vorherige schriftliche "
        "Zustimmung von Oakwood Capital reproduziert oder verbreitet werden.",
    ],
}


def _S(lang):
    """Return a string-lookup function with EN fallback."""
    table_l = STRINGS.get(lang, {})
    table_en = STRINGS["en"]
    def lookup(key):
        return table_l.get(key) or table_en.get(key, key)
    return lookup


def _styles():
    _register_fonts()
    ss = getSampleStyleSheet()
    styles = {}
    styles["title"] = ParagraphStyle(
        "OakTitle", parent=ss["Title"], fontName=F_SERIF,
        fontSize=26, textColor=C_GREEN, spaceAfter=2, leading=30, alignment=TA_LEFT,
    )
    styles["subtitle"] = ParagraphStyle(
        "OakSubtitle", parent=ss["Normal"], fontName=F_SANS,
        fontSize=10.5, textColor=C_MUTED, spaceAfter=14, leading=15, alignment=TA_LEFT,
    )
    styles["h2"] = ParagraphStyle(
        "OakH2", parent=ss["Heading2"], fontName=F_SERIF,
        fontSize=16, textColor=C_GREEN, spaceBefore=11, spaceAfter=7, leading=19,
    )
    styles["h3"] = ParagraphStyle(
        "OakH3", parent=ss["Heading3"], fontName=F_SANS_BOLD,
        fontSize=8, textColor=C_MUTED, spaceBefore=10, spaceAfter=4,
        leading=11, alignment=TA_LEFT,
    )
    styles["body"] = ParagraphStyle(
        "OakBody", parent=ss["Normal"], fontName=F_SANS,
        fontSize=9.5, textColor=C_TEXT, spaceAfter=7, leading=14, alignment=TA_JUSTIFY,
    )
    styles["small"] = ParagraphStyle(
        "OakSmall", parent=ss["Normal"], fontName=F_SANS,
        fontSize=8, textColor=C_MUTED, spaceAfter=4, leading=11,
    )
    styles["disclaimer"] = ParagraphStyle(
        "OakDisc", parent=ss["Normal"], fontName=F_SANS,
        fontSize=7.5, textColor=C_MUTED, spaceAfter=4, leading=10, alignment=TA_JUSTIFY,
    )
    styles["kpi_label"] = ParagraphStyle(
        "OakKpiLabel", parent=ss["Normal"], fontName=F_SANS,
        fontSize=7, textColor=C_MUTED, leading=9, alignment=TA_CENTER,
    )
    styles["kpi_value"] = ParagraphStyle(
        "OakKpiValue", parent=ss["Normal"], fontName=F_SERIF,
        fontSize=16, textColor=C_GREEN, leading=19, alignment=TA_CENTER,
    )
    styles["kpi_label_light"] = ParagraphStyle(
        "OakKpiLabelLight", parent=ss["Normal"], fontName=F_SANS,
        fontSize=7, textColor=C_CREAMD, leading=9, alignment=TA_CENTER,
    )
    styles["kpi_value_light"] = ParagraphStyle(
        "OakKpiValueLight", parent=ss["Normal"], fontName=F_SERIF,
        fontSize=16, textColor=C_GOLD, leading=19, alignment=TA_CENTER,
    )
    styles["foot"] = ParagraphStyle(
        "OakFoot", parent=ss["Normal"], fontName=F_SANS,
        fontSize=7, textColor=C_MUTED, leading=9, alignment=TA_CENTER,
    )
    return styles


def _png_to_image(png_bytes, width_mm, height_mm):
    """Wrap pre-rendered PNG bytes into a reportlab Image flowable."""
    if not png_bytes:
        return None
    try:
        bio = io.BytesIO(png_bytes)
        return Image(bio, width=width_mm * mm, height=height_mm * mm)
    except Exception:
        return None


def _kpi_grid(kpis, styles, cols=4, accent=False):
    """kpis: list of (label, value) tuples. Renders as a grid of cells.
    accent=True gives dark cards with a gold top accent (for highlight rows)."""
    cells = []
    row = []
    label_style = styles["kpi_label_light"] if accent else styles["kpi_label"]
    value_style = styles["kpi_value_light"] if accent else styles["kpi_value"]
    for i, (label, value) in enumerate(kpis):
        cell = [
            Paragraph(label.upper(), label_style),
            Spacer(1, 3),
            Paragraph(str(value), value_style),
        ]
        row.append(cell)
        if len(row) == cols:
            cells.append(row)
            row = []
    if row:
        while len(row) < cols:
            row.append([Paragraph("", label_style)])
        cells.append(row)

    tbl = Table(cells, colWidths=[(170 * mm) / cols] * cols)
    if accent:
        style = [
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("BACKGROUND", (0, 0), (-1, -1), C_PANEL_2),
            ("LINEABOVE", (0, 0), (-1, 0), 2, C_GOLD),
            ("LINEAFTER", (0, 0), (-2, -1), 0.5, C_GREEN2),
        ]
    else:
        style = [
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("BACKGROUND", (0, 0), (-1, -1), C_PANEL),
            ("LINEABOVE", (0, 0), (-1, 0), 1.2, C_GOLD),
            ("LINEAFTER", (0, 0), (-2, -1), 0.5, C_BORDER),
        ]
    tbl.setStyle(TableStyle(style))
    return tbl


def _section_heading(eyebrow_num, title, styles, lang):
    """Editorial section opener: small-caps numbered eyebrow above the H2.
    eyebrow_num: '01' through '07' — looked up in STRINGS as eyebrow_NN.
    title: the regular H2 string (pre-translated).
    Returns a list of flowables (caller can wrap in KeepTogether if needed).
    """
    S = _S(lang)
    eyebrow_text = S(f"eyebrow_{eyebrow_num}")
    # Tracked small-caps via hair-spaces between glyphs
    spaced_label = "&#8202;".join(eyebrow_text)
    eyebrow_html = (
        f"<font color='#B8954A'><b>{eyebrow_num}</b></font>"
        f"&nbsp;&nbsp;<font color='#9AA595'>·</font>&nbsp;&nbsp;"
        f"<font color='#7C8978'><b>{spaced_label}</b></font>"
    )
    eyebrow_style = ParagraphStyle(
        f"eyebrow_{eyebrow_num}", fontName=F_SANS_BOLD, fontSize=7,
        textColor=C_SAGE, leading=10, spaceBefore=14, spaceAfter=2)
    h2_style = ParagraphStyle(
        "h2_after_eyebrow", parent=styles["h2"], spaceBefore=0, spaceAfter=8)
    return [
        Paragraph(eyebrow_html, eyebrow_style),
        Paragraph(title, h2_style),
    ]


def _data_table(headers, rows, styles, col_widths=None, highlight_first_col=True):
    """Generic styled table. First column left-aligned (labels), all subsequent
    columns right-aligned (numbers) — the professional finance convention.
    Cells may be plain strings/numbers or pre-built Paragraph instances
    (the latter enables per-cell colour coding, e.g. excess return tinting)."""
    header_cells = []
    for j, h in enumerate(headers):
        align = TA_LEFT if j == 0 else TA_RIGHT
        header_cells.append(Paragraph(f"<b>{h}</b>", ParagraphStyle(
            f"th{j}", fontName=F_SANS_BOLD, fontSize=8, textColor=C_CREAM,
            leading=10, alignment=align)))
    data = [header_cells]
    for r in rows:
        cells = []
        for j, val in enumerate(r):
            if isinstance(val, Paragraph):
                cells.append(val)
                continue
            align = TA_LEFT if j == 0 else TA_RIGHT
            style = ParagraphStyle(
                f"td{j}", fontName=F_SANS if j > 0 or not highlight_first_col else F_SANS_BOLD,
                fontSize=8, textColor=C_TEXT, leading=11, alignment=align)
            cells.append(Paragraph(str(val), style))
        data.append(cells)

    n = len(headers)
    if col_widths is None:
        col_widths = [170 * mm / n] * n
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_PANEL_2),
        ("LINEABOVE", (0, 0), (-1, 0), 1, C_GOLD),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_PAGE_BG, C_TABLE_ALT]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, C_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def _monthly_returns_table(monthly_returns, styles):
    """Year × Month grid of monthly returns with green/red colour coding —
    the signature element of professional fund factsheets.
    monthly_returns: dict {year(int): [12 floats-or-None]}, plus optional 'YTD'.
    """
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    header = ["Year"] + months + ["FY"]
    head_cells = [Paragraph(f"<b>{h}</b>", ParagraphStyle(
        "mh", fontName=F_SANS_BOLD, fontSize=6.5, textColor=C_CREAM,
        leading=8, alignment=TA_CENTER)) for h in header]
    data = [head_cells]

    # Colour helper: blend from a light cream tone toward sage (positive) or
    # red (negative); intensity scales with magnitude.
    def _bg(v):
        if v is None:
            return C_PANEL
        mag = min(abs(v) / 10.0, 1.0)
        base_r, base_g, base_b = 0xF4, 0xF4, 0xEF  # light panel tone
        if v >= 0:
            r = int(base_r + (0x7C - base_r) * mag)
            g = int(base_g + (0x89 - base_g) * mag)
            b = int(base_b + (0x78 - base_b) * mag)
        else:
            r = int(base_r + (0xB8 - base_r) * mag)
            g = int(base_g + (0x50 - base_g) * mag)
            b = int(base_b + (0x42 - base_b) * mag)
        return colors.Color(r / 255, g / 255, b / 255)

    bg_cmds = []
    sorted_years = sorted(monthly_returns.keys())
    for ri, year in enumerate(sorted_years, start=1):
        vals = monthly_returns[year]
        row = [Paragraph(str(year), ParagraphStyle(
            "myr", fontName=F_SANS_BOLD, fontSize=6.8, textColor=C_CREAM,
            leading=8, alignment=TA_CENTER))]
        fy_product = 1.0
        has_any = False
        for ci, v in enumerate(vals[:12], start=1):
            if v is None:
                row.append(Paragraph("", styles["small"]))
            else:
                has_any = True
                fy_product *= (1 + v / 100.0)
                txt = f"{v:.1f}"
                # Dark text on faint cells, white on intense cells for contrast
                tcol = C_CREAM if abs(v) > 5 else C_TEXT
                row.append(Paragraph(txt, ParagraphStyle(
                    "mv", fontName=F_SANS, fontSize=6.3,
                    textColor=tcol, leading=8, alignment=TA_CENTER)))
                bg_cmds.append(("BACKGROUND", (ci, ri), (ci, ri), _bg(v)))
        # Full-year column
        fy = (fy_product - 1) * 100 if has_any else None
        if fy is not None:
            row.append(Paragraph(f"<b>{fy:.1f}</b>", ParagraphStyle(
                "mfy", fontName=F_SANS_BOLD, fontSize=6.3,
                textColor=C_GREEN, leading=8, alignment=TA_CENTER)))
            bg_cmds.append(("BACKGROUND", (13, ri), (13, ri), C_PANEL))
        else:
            row.append(Paragraph("", styles["small"]))
        data.append(row)

    col_widths = [12 * mm] + [11.7 * mm] * 12 + [13 * mm]
    tbl = Table(data, colWidths=col_widths)
    base_style = [
        ("BACKGROUND", (0, 0), (-1, 0), C_PANEL_2),
        ("LINEABOVE", (0, 0), (-1, 0), 1, C_GOLD),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 1), (-1, -1), 0.4, C_WHITE),
        ("BACKGROUND", (0, 1), (0, -1), C_PANEL_2),
    ]
    tbl.setStyle(TableStyle(base_style + bg_cmds))
    return tbl


def _two_col_universe(rows, styles):
    """Render the investment universe as two side-by-side compact tables so all
    constituents fit on a single page. rows: list of [name, ticker, sector]."""
    half = (len(rows) + 1) // 2
    left_rows = rows[:half]
    right_rows = rows[half:]

    def _mini(sub):
        header = [Paragraph("<b>Constituent</b>", ParagraphStyle(
                    "uh", fontName=F_SANS_BOLD, fontSize=7.5, textColor=C_CREAM, leading=9)),
                  Paragraph("<b>Ticker</b>", ParagraphStyle(
                    "uh2", fontName=F_SANS_BOLD, fontSize=7.5, textColor=C_CREAM, leading=9))]
        data = [header]
        for r in sub:
            name_p = Paragraph(str(r[0]), ParagraphStyle(
                "un", fontName=F_SANS, fontSize=7.5, textColor=C_TEXT, leading=10))
            tick_p = Paragraph(str(r[1]), ParagraphStyle(
                "ut", fontName=F_SANS, fontSize=7.5, textColor=C_MUTED, leading=10))
            data.append([name_p, tick_p])
        t = Table(data, colWidths=[52 * mm, 28 * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_PANEL_2),
            ("LINEABOVE", (0, 0), (-1, 0), 1, C_GOLD),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_PANEL, C_PAGE_BG]),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, C_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return t

    left_tbl = _mini(left_rows)
    right_tbl = _mini(right_rows) if right_rows else Spacer(1, 1)
    outer = Table([[left_tbl, right_tbl]], colWidths=[83 * mm, 83 * mm])
    outer.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 4),
    ]))
    return outer


# ---------------------------------------------------------------------------
# IB-style fact sheet elements
# ---------------------------------------------------------------------------

def _strategy_snapshot_panel(snapshot_data, lang, styles):
    """Render a compact 3-column "Strategy Snapshot" panel — the canonical
    IB-factsheet fact box. snapshot_data is a list of (label_key, value)
    tuples; label_keys are STRINGS keys ('sn_inception', 'sn_currency', ...)
    or raw label strings if no key matches.
    """
    S = _S(lang)
    if not snapshot_data:
        return Spacer(1, 1)

    label_style = ParagraphStyle(
        "snap_label", fontName=F_SANS, fontSize=7, textColor=C_SAGE,
        leading=9, alignment=0)
    value_style = ParagraphStyle(
        "snap_value", fontName=F_SERIF, fontSize=10.5, textColor=C_TEXT,
        leading=13, alignment=0)

    # Build mini-cells: each card is a small two-row stack (label / value).
    cards = []
    for key, value in snapshot_data:
        label = S(key) if key in STRINGS["en"] else key
        cell = [
            Paragraph(label.upper(), label_style),
            Paragraph(str(value), value_style),
        ]
        cards.append(cell)

    # Arrange 3 per row
    rows = []
    for i in range(0, len(cards), 3):
        chunk = cards[i:i + 3]
        # Pad to length 3 so all rows have the same number of cols
        while len(chunk) < 3:
            chunk.append([Spacer(1, 1)])
        rows.append(chunk)

    col_w = 170 / 3.0 * mm  # 3 cols on the content width
    t = Table(rows, colWidths=[col_w] * 3)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, -1), C_PANEL),
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, C_GOLD),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    # Subtle vertical dividers between the three columns
    for c in (1, 2):
        style_cmds.append(("LINEBEFORE", (c, 0), (c, -1), 0.4, C_BORDER))
    t.setStyle(TableStyle(style_cmds))
    return t


def _period_returns_table(period_returns_data, lang, styles):
    """Render the 'Performance per Period' table.
    period_returns_data is a list of (label_key, strategy_pct, bench_pct,
    excess_pct) tuples from compute_period_returns(). Any of bench/excess
    can be None and will render as a dash.
    Excess column is colour-coded (green for positive, red for negative).
    """
    S = _S(lang)
    if not period_returns_data:
        return Spacer(1, 1)

    headers = [S("pr_period"), S("pr_strategy"), S("pr_benchmark"), S("pr_excess")]

    def _fmt(x):
        if x is None:
            return "—"
        sign = "+" if x >= 0 else ""
        return f"{sign}{x:.2f}%"

    def _excess_para(x):
        if x is None:
            return "—"
        sign = "+" if x >= 0 else ""
        text = f"{sign}{x:.2f}%"
        color = C_GREEN_POS if x >= 0 else C_RED
        return Paragraph(text, ParagraphStyle(
            "exc", fontName=F_SANS_BOLD, fontSize=8, textColor=color,
            leading=11, alignment=TA_RIGHT))

    rows = []
    for key, s, b, e in period_returns_data:
        rows.append([S(key), _fmt(s), _fmt(b), _excess_para(e)])

    return _data_table(headers, rows, styles,
                       col_widths=[50 * mm, 40 * mm, 40 * mm, 40 * mm])


def _top_drawdowns_table(drawdowns, lang, styles):
    """Render the 'Top 5 Drawdowns' table.
    drawdowns is a list of dicts from identify_top_drawdowns().
    """
    S = _S(lang)
    if not drawdowns:
        return Spacer(1, 1)

    headers = [S("dd_start"), S("dd_trough"), S("dd_end"),
               S("dd_duration"), S("dd_depth"), S("dd_recovery")]

    def _dfmt(ts):
        if ts is None:
            return "—"
        try:
            return ts.strftime("%Y-%m-%d")
        except Exception:
            return str(ts)

    rows = []
    for d in drawdowns:
        end_str = _dfmt(d.get("end")) if d.get("end") is not None else S("dd_ongoing")
        rec_str = (f"{d['recovery_days']} {S('dd_days')}"
                   if d.get("recovery_days") is not None else S("dd_ongoing"))
        rows.append([
            _dfmt(d["start"]),
            _dfmt(d["trough"]),
            end_str,
            f"{d['duration_days']} {S('dd_days')}",
            f"{d['depth_pct']:.2f}%",
            rec_str,
        ])

    return _data_table(headers, rows, styles,
                       col_widths=[27*mm, 27*mm, 27*mm, 27*mm, 27*mm, 35*mm])


def _universe_sector_table(rows, lang, styles):
    """Render the investment universe with Sector + Weight — single full-width
    table. rows: list of [name, ticker, sector, weight_pct].
    If rows have only 3 entries (no weight), the weight column is omitted.
    If rows have only 2 entries (no sector either), fall back to the
    two-column compact layout.
    """
    S = _S(lang)
    if not rows:
        return Spacer(1, 1)

    # Detect schema. We render the four-col layout only if every row has a
    # non-empty sector AND a weight is present somewhere — otherwise we keep
    # the legacy two-col layout so old callers don't break.
    have_sector = any(len(r) >= 3 and r[2] for r in rows)
    have_weight = any(len(r) >= 4 and r[3] is not None for r in rows)

    if not have_sector and not have_weight:
        return _two_col_universe(rows, styles)

    headers = [S("uni_constituent"), S("uni_ticker"), S("uni_sector")]
    col_widths = [55 * mm, 30 * mm, 50 * mm]
    if have_weight:
        headers.append(S("uni_weight"))
        col_widths.append(35 * mm)

    body = []
    for r in rows:
        row = [str(r[0]) if len(r) > 0 else "",
               str(r[1]) if len(r) > 1 else "",
               str(r[2]) if len(r) > 2 else ""]
        if have_weight:
            w = r[3] if len(r) > 3 else None
            row.append(f"{w:.2f}%" if isinstance(w, (int, float)) else (str(w) if w else "—"))
        body.append(row)

    return _data_table(headers, body, styles, col_widths=col_widths)


def _draw_cover(canvas, doc, strategy_name, strategy_subtitle, period_str,
                highlight_kpis, logo_path, lang="en"):
    """Full-bleed dark-green cover page drawn directly on the canvas.
    highlight_kpis: list of up to 3 (label, value) tuples for the hero band."""
    S = _S(lang)
    canvas.saveState()
    W, H = A4
    # Cover stays deep green (dark cover + light interior = classic factsheet).
    # Use a light sage locally since the global COVER_SAGE is dark for the light theme.
    COVER_SAGE = colors.HexColor("#99A796")

    # Full-page deep green background
    canvas.setFillColor(C_GREEN)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)

    # Subtle darker band at the very top and bottom for depth
    canvas.setFillColor(C_GREEN2)
    canvas.rect(0, H - 6 * mm, W, 6 * mm, fill=1, stroke=0)
    canvas.rect(0, 0, W, 6 * mm, fill=1, stroke=0)

    # Logo (cream-on-transparent) centered in the upper third
    if logo_path and os.path.exists(logo_path):
        try:
            from reportlab.lib.utils import ImageReader
            img = ImageReader(logo_path)
            iw, ih = img.getSize()
            disp_w = 70 * mm
            disp_h = disp_w * ih / iw
            canvas.drawImage(img, (W - disp_w) / 2, H - 78 * mm,
                             width=disp_w, height=disp_h,
                             mask="auto", preserveAspectRatio=True)
        except Exception:
            pass

    # Thin gold rule under the logo area
    canvas.setStrokeColor(C_GOLD)
    canvas.setLineWidth(1)
    canvas.line(W / 2 - 30 * mm, H - 92 * mm, W / 2 + 30 * mm, H - 92 * mm)

    # Strategy title (centered, serif, cream)
    canvas.setFillColor(C_CREAM)
    canvas.setFont(F_SERIF, 32)
    canvas.drawCentredString(W / 2, H - 112 * mm, strategy_name)

    # Subtitle (wrapped, centered, sage) — simple word wrap
    canvas.setFont(F_SANS, 10.5)
    canvas.setFillColor(COVER_SAGE)
    words = strategy_subtitle.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if canvas.stringWidth(test, F_SANS, 10.5) < 150 * mm:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    y = H - 122 * mm
    for ln in lines[:3]:
        canvas.drawCentredString(W / 2, y, ln)
        y -= 6 * mm

    # Hero highlight metrics band (up to 3), centered.
    # Separated by whitespace only — no divider lines (cleaner, more premium).
    if highlight_kpis:
        n = len(highlight_kpis)
        band_w = 170 * mm
        cell_w = band_w / n
        x0 = (W - band_w) / 2
        y_band = 95 * mm
        for i, (label, value) in enumerate(highlight_kpis):
            cx = x0 + i * cell_w + cell_w / 2
            canvas.setFillColor(C_GOLD)
            canvas.setFont(F_SERIF, 27)
            canvas.drawCentredString(cx, y_band + 8 * mm, str(value))
            canvas.setFillColor(COVER_SAGE)
            canvas.setFont(F_SANS, 7.5)
            canvas.drawCentredString(cx, y_band, label.upper())

    # Italic tagline + gold ornament — fills the empty zone between hero KPIs
    # (95mm) and the date block (52mm). Editorial flourish.
    canvas.setFillColor(C_CREAMD)
    canvas.setFont(F_SERIF_ITALIC, 12)
    canvas.drawCentredString(W / 2, 78 * mm, S("cover_strapline"))
    # Gold ornament: thin line · diamond · thin line, centered
    canvas.setStrokeColor(C_GOLD)
    canvas.setLineWidth(0.6)
    orn_y = 70 * mm
    canvas.line(W / 2 - 22 * mm, orn_y, W / 2 - 4 * mm, orn_y)
    canvas.line(W / 2 + 4 * mm, orn_y, W / 2 + 22 * mm, orn_y)
    canvas.setFillColor(C_GOLD)
    canvas.setFont(F_SANS, 8)
    canvas.drawCentredString(W / 2, orn_y - 1.5, "◆")

    # Period + generation block near the bottom
    canvas.setFillColor(C_CREAMD)
    canvas.setFont(F_SANS, 9)
    canvas.drawCentredString(W / 2, 52 * mm, f"{S('cover_period')}   {period_str}")
    canvas.setFillColor(COVER_SAGE)
    canvas.setFont(F_SANS, 8)
    canvas.drawCentredString(W / 2, 45 * mm,
                             f"{S('cover_generated')} {datetime.now().strftime('%d %B %Y, %H:%M')}")

    # Contact + confidential footer mark
    canvas.setFillColor(COVER_SAGE)
    canvas.setFont(F_SANS, 7.5)
    canvas.drawCentredString(W / 2, 26 * mm,
                             "Oakwood Capital Consulting AG  ·  Gotthardstrasse 14  ·  6300 Zug")
    canvas.drawCentredString(W / 2, 21 * mm,
                             "+41 79 250 72 31  ·  info@oakwood-capital.ch  ·  www.oakwood-capital.ch")
    canvas.setFillColor(colors.HexColor("#7C8978"))
    canvas.setFont(F_SANS, 7)
    canvas.drawCentredString(W / 2, 15 * mm, S("cover_tagline").upper())
    canvas.setFillColor(C_GOLD)
    canvas.setFont(F_SERIF_ITALIC, 12)
    canvas.drawCentredString(W / 2, 9 * mm, S("cover_byline"))

    canvas.restoreState()


def _header_footer(canvas, doc, strategy_name, logo_path=None, lang="en"):
    S = _S(lang)
    canvas.saveState()
    W, H = A4
    # Full-page light background (ReportLab doesn't fill it automatically)
    canvas.setFillColor(C_PAGE_BG)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)

    # Subtle centered logo watermark (very faint) on interior pages
    if logo_path and os.path.exists(logo_path):
        try:
            from reportlab.lib.utils import ImageReader
            img = ImageReader(logo_path)
            iw, ih = img.getSize()
            wm_w = 120 * mm
            wm_h = wm_w * ih / iw
            canvas.saveState()
            canvas.setFillAlpha(0.022)
            canvas.drawImage(img, (W - wm_w) / 2, (H - wm_h) / 2,
                             width=wm_w, height=wm_h,
                             mask="auto", preserveAspectRatio=True)
            canvas.restoreState()
        except Exception:
            pass

    # Header band — deep green, mirroring the website's header
    canvas.setFillColor(C_GREEN)
    canvas.rect(0, H - 22 * mm, W, 22 * mm, fill=1, stroke=0)
    # Thin gold hairline under the header band
    canvas.setStrokeColor(C_GOLD)
    canvas.setLineWidth(0.75)
    canvas.line(0, H - 22 * mm, W, H - 22 * mm)

    canvas.setFillColor(C_CREAM)
    canvas.setFont(F_SERIF, 15)
    canvas.drawString(20 * mm, H - 14 * mm, "Oakwood Capital")
    canvas.setFillColor(C_SAGE_DIM)
    canvas.setFont(F_SANS, 7)
    canvas.drawRightString(W - 20 * mm, H - 11 * mm, S("platform"))
    canvas.drawRightString(W - 20 * mm, H - 15 * mm, S("confidential"))

    # Footer
    canvas.setFillColor(C_MUTED)
    canvas.setFont(F_SANS, 7)
    canvas.drawString(20 * mm, 12 * mm, S("illustrative"))
    canvas.drawCentredString(W / 2, 12 * mm, strategy_name)
    total = getattr(doc, "_total_pages", None)
    if total:
        canvas.drawRightString(W - 20 * mm, 12 * mm, f"{doc.page:02d} / {total:02d}")
    else:
        canvas.drawRightString(W - 20 * mm, 12 * mm, f"{doc.page:02d}")
    canvas.setStrokeColor(C_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 15 * mm, W - 20 * mm, 15 * mm)
    canvas.restoreState()


def build_tearsheet(
    strategy_name,
    strategy_subtitle,
    period_str,
    kpis_performance,      # list of (label, value)
    kpis_risk,             # list of (label, value)
    fee_summary,           # list of (label, value)
    risk_table_headers,    # list
    risk_table_rows,       # list of lists
    fee_table_headers,     # list
    fee_table_rows,        # list of lists
    figures,               # list of (title, plotly_fig) tuples
    params_summary,        # list of (label, value) for the methodology page
    universe_rows,         # list of [name, ticker, sector] for holdings page
    monthly_returns=None,  # optional: dict {year: [12 monthly % values or None]}
    exec_summary=None,     # optional: short prose string for an executive summary
    key_takeaways=None,    # optional: list of short bullet strings
    scatter_png=None,      # optional: pre-rendered risk/return scatter PNG bytes
    # --- IB-style add-ons (all optional, all skip silently when None/empty) ---
    snapshot_data=None,    # optional: list of (label_key, value) for the Strategy Snapshot box
    period_returns=None,   # optional: list of (label_key, strat, bench, excess) from compute_period_returns()
    top_drawdowns=None,    # optional: list of dicts from identify_top_drawdowns()
    lang="en",             # "en" or "de" (de falls back to en strings in Phase 1)
):
    """Build the PDF and return raw bytes."""
    S = _S(lang)
    styles = _styles()
    buf = io.BytesIO()

    # Locate the logo for the cover
    logo_path = None
    for cand in (
        os.path.join(os.path.dirname(__file__), "assets", "oakwood_logo.png"),
        os.path.join(os.path.dirname(__file__), "assets", "logo.png"),
    ):
        if os.path.exists(cand):
            logo_path = cand
            break

    # Pick up to 3 hero metrics from the performance KPIs for the cover band
    highlight_kpis = kpis_performance[:3] if kpis_performance else []

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=28 * mm, bottomMargin=18 * mm,
        leftMargin=20 * mm, rightMargin=20 * mm,
        title=f"Oakwood Capital — {strategy_name}",
        author="Oakwood Capital",
    )

    story = []

    # ===== PAGE 1: COVER =====
    # The cover is drawn entirely by the onFirstPage canvas callback. We just
    # need one flowable so the first page exists, then break to content.
    story.append(Spacer(1, 1))
    story.append(PageBreak())

    # ===== PAGE 2: KPIs / Summary =====
    if exec_summary:
        for fl in _section_heading("01", S("exec_summary"), styles, lang):
            story.append(fl)
        story.append(Paragraph(exec_summary, styles["body"]))
        story.append(Spacer(1, 8))

    # Key Takeaways box — crisp bullets in a gold-accented panel
    if key_takeaways:
        kt_style = ParagraphStyle(
            "kt", fontName=F_SANS, fontSize=8.5, textColor=C_TEXT, leading=13,
            leftIndent=10, bulletIndent=0, spaceAfter=4)
        kt_flow = [Paragraph(S("key_takeaways"), ParagraphStyle(
            "kth", fontName=F_SANS_BOLD, fontSize=7.5, textColor=C_GOLD,
            leading=10, spaceAfter=6))]
        for tk in key_takeaways[:4]:
            kt_flow.append(Paragraph(f"<font color='#B8954A'>&#9642;</font>&nbsp;&nbsp;{tk}", kt_style))
        kt_tbl = Table([[kt_flow]], colWidths=[170 * mm])
        kt_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_PANEL),
            ("LINEABOVE", (0, 0), (-1, 0), 1.5, C_GOLD),
            ("TOPPADDING", (0, 0), (-1, -1), 11),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ]))
        story.append(kt_tbl)
        story.append(Spacer(1, 12))

    story.append(Paragraph(S("perf_summary"), styles["h2"]))
    story.append(Paragraph(S("perf_summary_sub"), styles["h3"]))
    story.append(Spacer(1, 2))
    story.append(_kpi_grid(kpis_performance, styles, cols=4, accent=True))
    story.append(Spacer(1, 9))

    story.append(Paragraph(S("risk_metrics"), styles["h2"]))
    story.append(_kpi_grid(kpis_risk, styles, cols=4))
    story.append(Spacer(1, 9))

    story.append(Paragraph(S("fee_summary"), styles["h2"]))
    story.append(_kpi_grid(fee_summary, styles, cols=4))

    # IB-style Strategy Snapshot panel — 3x2 fact box, fills the lower
    # half of P2 with the canonical "fact box" investors expect.
    if snapshot_data:
        story.append(Spacer(1, 10))
        story.append(Paragraph(S("snapshot"), styles["h2"]))
        story.append(_strategy_snapshot_panel(snapshot_data, lang, styles))

    # IB-style Performance per Period table — trailing returns over standard
    # horizons. Sits on P2 below the Snapshot, giving the page real density.
    if period_returns:
        story.append(Spacer(1, 10))
        story.append(KeepTogether(
            _section_heading("02", S("period_returns"), styles, lang) + [
                Paragraph(S("period_returns_sub"), styles["h3"]),
                Spacer(1, 2),
                _period_returns_table(period_returns, lang, styles),
            ]))

    # Monthly Returns heatmap — kept together so it never splits, and
    # placed thematically with Period Returns as 'Performance Detail'
    # (rather than after the charts).
    if monthly_returns:
        story.append(Spacer(1, 12))
        story.append(KeepTogether([
            Paragraph(S("monthly_returns"), styles["h2"]),
            Paragraph(S("monthly_returns_sub"), styles["h3"]),
            Spacer(1, 3),
            _monthly_returns_table(monthly_returns, styles),
        ]))

    # ===== PAGE 3 + 4: Charts (Evolution+Drawdown, then Yearly+Scatter) =====
    # Charts come immediately after the headline KPIs — classic factsheet
    # flow. Each chart sits in its own KeepTogether so a partial-fit never
    # splits an image across pages.
    story.append(PageBreak())
    for fl in _section_heading("03", S("charts"), styles, lang):
        story.append(fl)
    any_chart = False
    n_fig = len(figures)
    for idx, (title, png_bytes) in enumerate(figures):
        is_last = (idx == n_fig - 1)
        if is_last and n_fig >= 3:
            # Yearly perf opens a fresh page paired with the scatter
            story.append(PageBreak())
            img = _png_to_image(png_bytes, width_mm=170, height_mm=100)
        else:
            # Evolution + drawdown share one page — height tuned so both fit
            img = _png_to_image(png_bytes, width_mm=170, height_mm=95)
        if img is not None:
            any_chart = True
            block = KeepTogether([
                Paragraph(title, styles["h3"]),
                Spacer(1, 2),
                img,
                Spacer(1, 12),
            ])
            story.append(block)
    if not any_chart:
        story.append(Paragraph(
            "Charts could not be embedded in this PDF (chart rendering engine "
            "unavailable in the current environment). All numerical data is "
            "provided in full in the tables on the following pages.",
            styles["body"]))

    # Risk/Return positioning scatter — sits below the yearly chart on page 4
    if scatter_png:
        sc_img = _png_to_image(scatter_png, width_mm=160, height_mm=90)
        if sc_img is not None:
            story.append(Spacer(1, 6))
            story.append(KeepTogether([
                Paragraph(S("scatter"), styles["h3"]),
                Spacer(1, 2),
                sc_img,
            ]))

    # ===== PAGE 5: Detailed Risk Metrics + Top Drawdowns =====
    story.append(KeepTogether(
        _section_heading("04", S("detailed_risk"), styles, lang) + [
            _data_table(risk_table_headers, risk_table_rows, styles,
                        col_widths=[55 * mm, 40 * mm, 40 * mm, 35 * mm])
            if risk_table_rows else Spacer(1, 1),
            Spacer(1, 12),
        ]))

    # IB-style Top 5 Drawdowns — gives the Max-Drawdown KPI real substance
    # by showing the actual peak-to-trough episodes with duration and
    # recovery time.
    if top_drawdowns:
        story.append(KeepTogether([
            Paragraph(S("top_drawdowns"), styles["h2"]),
            Paragraph(S("top_drawdowns_sub"), styles["h3"]),
            Spacer(1, 2),
            _top_drawdowns_table(top_drawdowns, lang, styles),
            Spacer(1, 12),
        ]))

    # ===== PAGE 6: Perf Fee Crystallization + Methodology + Universe =====
    # Perf fee in KeepTogether so it never splits mid-table. Methodology
    # and Universe flow directly after — no forced PageBreak — so they
    # fill the page elegantly when there's room.
    if fee_table_rows:
        story.append(KeepTogether(
            _section_heading("05", S("perf_fee_crystal"), styles, lang) + [
                _data_table(fee_table_headers, fee_table_rows, styles),
            ]))
    else:
        for fl in _section_heading("05", S("perf_fee_crystal"), styles, lang):
            story.append(fl)
        story.append(Paragraph(S("no_perf_fee"),
                               styles["body"]))

    # Methodology + Universe each in their own KeepTogether — they flow
    # after Perf Fees and land on the next page if there's no room. The
    # H2 styles already provide their own spaceBefore so no free-standing
    # Spacer is needed (a top-of-frame Spacer would trigger a LayoutError).
    if params_summary:
        story.append(KeepTogether([
            Paragraph(S("methodology"), styles["h2"]),
            _data_table([S("param_label"), S("param_value")],
                        [[k, v] for k, v in params_summary], styles,
                        col_widths=[85 * mm, 85 * mm]),
        ]))

    if universe_rows:
        story.append(KeepTogether(
            _section_heading("06", S("universe"), styles, lang) + [
                _universe_sector_table(universe_rows, lang, styles),
        ]))

    # Keep the entire disclosures section together on a fresh page
    story.append(PageBreak())
    disc_block = list(_section_heading("07", S("disclosures"), styles, lang))
    disclaimer_paragraphs = (DISCLAIMER_PARAGRAPHS.get(lang)
                             or DISCLAIMER_PARAGRAPHS["en"])
    for p in disclaimer_paragraphs:
        disc_block.append(Paragraph(p, styles["disclaimer"]))

    # Contact block — a panel with a gold top rule, in the brand style
    disc_block.append(Spacer(1, 16))
    contact_name = Paragraph(
        "Oakwood Capital Consulting AG", ParagraphStyle(
            "cn", fontName=F_SERIF, fontSize=13, textColor=C_CREAM, leading=16))
    contact_lines = Paragraph(
        "Gotthardstrasse 14&nbsp;&nbsp;·&nbsp;&nbsp;6300 Zug&nbsp;&nbsp;·&nbsp;&nbsp;Switzerland<br/>"
        "+41 79 250 72 31&nbsp;&nbsp;·&nbsp;&nbsp;info@oakwood-capital.ch&nbsp;&nbsp;·&nbsp;&nbsp;www.oakwood-capital.ch",
        ParagraphStyle("cl", fontName=F_SANS, fontSize=8.5, textColor=C_CREAMD,
                       leading=14))
    contact_tbl = Table([[contact_name], [contact_lines]], colWidths=[170 * mm])
    contact_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_GREEN),
        ("LINEABOVE", (0, 0), (-1, 0), 1.5, C_GOLD),
        ("TOPPADDING", (0, 0), (-1, 0), 12),
        ("TOPPADDING", (0, 1), (-1, 1), 2),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 12),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
    ]))
    disc_block.append(contact_tbl)
    story.append(KeepTogether(disc_block))

    # First page = full cover art; all later pages = header/footer band.
    def on_first(canvas, doc_):
        _draw_cover(canvas, doc_, strategy_name, strategy_subtitle, period_str,
                    highlight_kpis, logo_path, lang=lang)

    def on_later(canvas, doc_):
        _header_footer(canvas, doc_, strategy_name, logo_path=logo_path, lang=lang)

    # Pass 1: build once to count total pages (no footer total yet)
    try:
        import copy
        probe_buf = io.BytesIO()
        probe_doc = SimpleDocTemplate(
            probe_buf, pagesize=A4,
            topMargin=28 * mm, bottomMargin=18 * mm,
            leftMargin=20 * mm, rightMargin=20 * mm)
        probe_story = copy.copy(story)
        probe_doc.build(probe_story, onFirstPage=on_first, onLaterPages=on_later)
        total_pages = probe_doc.page
        doc._total_pages = total_pages
    except Exception:
        doc._total_pages = None

    # Pass 2: real build with the total page count available to the footer
    doc.build(story, onFirstPage=on_first, onLaterPages=on_later)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Bilingual wrapper — builds DE and EN versions sequentially, concatenated
# into a single PDF via pypdf.
# ---------------------------------------------------------------------------

def build_bilingual_tearsheet(
    strategy_name,
    strategy_subtitle_de,
    strategy_subtitle_en,
    period_str,
    kpis_performance,
    kpis_risk,
    fee_summary,
    risk_table_headers,
    risk_table_rows,
    fee_table_headers,
    fee_table_rows,
    figures,
    params_summary,
    universe_rows,
    monthly_returns=None,
    exec_summary_de=None,
    exec_summary_en=None,
    key_takeaways_de=None,
    key_takeaways_en=None,
    scatter_png=None,
    snapshot_data=None,
    period_returns=None,
    top_drawdowns=None,
):
    """Build a single PDF with both DE and EN versions concatenated.

    Sprach-spezifische Parameter (subtitle, exec_summary, key_takeaways)
    werden pro Sprache übergeben; alle anderen Parameter sind sprach-neutral
    und werden durchgereicht. Section-Titles, Labels und Disclaimer kommen
    aus dem STRINGS-Dict und werden automatisch je Sprache gerendert.

    Reihenfolge im finalen PDF:
      Seiten 1-9: Deutsche Version (Cover + Inhalt)
      Seiten 10-18: Englische Version (Cover + Inhalt)
    """
    # Tolerant PDF-merge import: prefer modern pypdf, fall back to the
    # legacy PyPDF2 fork (still common on hosted Python environments).
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfReader, PdfWriter  # PyPDF2 ≥ 3.0
        except ImportError:
            raise ImportError(
                "build_bilingual_tearsheet needs either 'pypdf' (preferred) "
                "or 'PyPDF2' installed to merge the two language versions. "
                "Add 'pypdf' to requirements.txt."
            )

    common_kwargs = dict(
        strategy_name=strategy_name,
        period_str=period_str,
        kpis_performance=kpis_performance,
        kpis_risk=kpis_risk,
        fee_summary=fee_summary,
        risk_table_headers=risk_table_headers,
        risk_table_rows=risk_table_rows,
        fee_table_headers=fee_table_headers,
        fee_table_rows=fee_table_rows,
        figures=figures,
        params_summary=params_summary,
        universe_rows=universe_rows,
        monthly_returns=monthly_returns,
        scatter_png=scatter_png,
        snapshot_data=snapshot_data,
        period_returns=period_returns,
        top_drawdowns=top_drawdowns,
    )

    de_pdf = build_tearsheet(
        strategy_subtitle=strategy_subtitle_de,
        exec_summary=exec_summary_de,
        key_takeaways=key_takeaways_de,
        lang="de",
        **common_kwargs,
    )
    en_pdf = build_tearsheet(
        strategy_subtitle=strategy_subtitle_en,
        exec_summary=exec_summary_en,
        key_takeaways=key_takeaways_en,
        lang="en",
        **common_kwargs,
    )

    writer = PdfWriter()
    for src in (de_pdf, en_pdf):
        reader = PdfReader(io.BytesIO(src))
        for page in reader.pages:
            writer.add_page(page)

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()
