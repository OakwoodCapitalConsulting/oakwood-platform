"""
Oakwood Capital — PDF Tearsheet Generator
==========================================
Builds a multi-page institutional-style PDF report from backtest results.
Used by both strategy pages. Returns PDF bytes for st.download_button.

Requires: reportlab, kaleido (for Plotly chart -> PNG export).
"""

import io
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
    Image, PageBreak, HRFlowable, KeepTogether
)

# Matplotlib palette matching Oakwood CI
MPL_GREEN = "#293624"
MPL_SAGE  = "#99A796"
MPL_GOLD  = "#C9A961"
MPL_CREAM = "#F5F5F1"
MPL_BTC   = "#F7931A"
MPL_RED   = "#B85042"
MPL_GRID  = "#D8DCD3"


def render_line_chart(series_dict, title="", ylabel="", percent=False, fill_first=False):
    """Render a line chart to PNG bytes using matplotlib (no browser needed).
    series_dict: ordered dict-like list of (label, pandas Series, color, style)."""
    try:
        fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=150)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        for i, (label, s, color, style) in enumerate(series_dict):
            if s is None or len(s) == 0:
                continue
            vals = s.values * 100 if percent else s.values
            ax.plot(s.index, vals, label=label, color=color,
                    linewidth=style.get("lw", 1.8),
                    linestyle=style.get("ls", "-"),
                    alpha=style.get("alpha", 1.0))
            if fill_first and i == 0:
                ax.fill_between(s.index, vals, alpha=0.08, color=color)
        ax.set_ylabel(ylabel, fontsize=9, color="#2A2A26")
        ax.tick_params(labelsize=8, colors="#5C6B57")
        ax.grid(True, color=MPL_GRID, linewidth=0.6, alpha=0.8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(MPL_GRID)
        if percent:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        leg = ax.legend(fontsize=7.5, loc="best", frameon=False)
        if leg:
            for text in leg.get_texts():
                text.set_color("#2A2A26")
        try:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        except Exception:
            pass
        fig.tight_layout(pad=1.2)
        bio = io.BytesIO()
        fig.savefig(bio, format="png", facecolor="white", bbox_inches="tight")
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
        fig, ax = plt.subplots(figsize=(9.5, 4.0), dpi=150)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        bar_colors = [MPL_SAGE if v >= 0 else MPL_RED for v in values]
        ax.bar(range(len(values)), values, color=bar_colors,
               edgecolor=MPL_GREEN, linewidth=0.5)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=8, color="#5C6B57")
        ax.set_ylabel(ylabel, fontsize=9, color="#2A2A26")
        ax.tick_params(labelsize=8, colors="#5C6B57")
        ax.grid(True, axis="y", color=MPL_GRID, linewidth=0.6, alpha=0.8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(MPL_GRID)
        ax.axhline(0, color="#5C6B57", linewidth=0.8)
        if hurdle is not None:
            ax.axhline(hurdle, color=MPL_GOLD, linewidth=1.3, linestyle="--",
                       label=f"Year-1 Hurdle {hurdle:.0f}%")
            leg = ax.legend(fontsize=7.5, frameon=False)
            if leg:
                for t in leg.get_texts():
                    t.set_color("#2A2A26")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        fig.tight_layout(pad=1.2)
        bio = io.BytesIO()
        fig.savefig(bio, format="png", facecolor="white", bbox_inches="tight")
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
# Brand palette (reportlab Color objects)
# ---------------------------------------------------------------------------
C_GREEN   = colors.HexColor("#293624")
C_GREEN2  = colors.HexColor("#1F2A1B")
C_SAGE    = colors.HexColor("#99A796")
C_CREAM   = colors.HexColor("#F5F5F1")
C_CREAMD  = colors.HexColor("#E8E8E2")
C_GOLD    = colors.HexColor("#C9A961")
C_BORDER  = colors.HexColor("#C8CCC2")
C_TEXT    = colors.HexColor("#2A2A26")
C_MUTED   = colors.HexColor("#5C6B57")
C_RED     = colors.HexColor("#B85042")
C_WHITE   = colors.white


def _styles():
    ss = getSampleStyleSheet()
    styles = {}
    styles["title"] = ParagraphStyle(
        "OakTitle", parent=ss["Title"], fontName="Times-Roman",
        fontSize=26, textColor=C_GREEN, spaceAfter=2, leading=30, alignment=TA_LEFT,
    )
    styles["subtitle"] = ParagraphStyle(
        "OakSubtitle", parent=ss["Normal"], fontName="Helvetica",
        fontSize=10.5, textColor=C_MUTED, spaceAfter=14, leading=15, alignment=TA_LEFT,
    )
    styles["h2"] = ParagraphStyle(
        "OakH2", parent=ss["Heading2"], fontName="Times-Roman",
        fontSize=15, textColor=C_GREEN, spaceBefore=16, spaceAfter=8, leading=18,
    )
    styles["h3"] = ParagraphStyle(
        "OakH3", parent=ss["Heading3"], fontName="Helvetica-Bold",
        fontSize=8.5, textColor=C_MUTED, spaceBefore=10, spaceAfter=4,
        leading=11, alignment=TA_LEFT,
    )
    styles["body"] = ParagraphStyle(
        "OakBody", parent=ss["Normal"], fontName="Helvetica",
        fontSize=9.5, textColor=C_TEXT, spaceAfter=7, leading=14, alignment=TA_JUSTIFY,
    )
    styles["small"] = ParagraphStyle(
        "OakSmall", parent=ss["Normal"], fontName="Helvetica",
        fontSize=8, textColor=C_MUTED, spaceAfter=4, leading=11,
    )
    styles["disclaimer"] = ParagraphStyle(
        "OakDisc", parent=ss["Normal"], fontName="Helvetica",
        fontSize=7.5, textColor=C_MUTED, spaceAfter=4, leading=10, alignment=TA_JUSTIFY,
    )
    styles["kpi_label"] = ParagraphStyle(
        "OakKpiLabel", parent=ss["Normal"], fontName="Helvetica",
        fontSize=7, textColor=C_MUTED, leading=9, alignment=TA_CENTER,
    )
    styles["kpi_value"] = ParagraphStyle(
        "OakKpiValue", parent=ss["Normal"], fontName="Times-Roman",
        fontSize=15, textColor=C_GREEN, leading=18, alignment=TA_CENTER,
    )
    styles["foot"] = ParagraphStyle(
        "OakFoot", parent=ss["Normal"], fontName="Helvetica",
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


def _kpi_grid(kpis, styles, cols=4):
    """kpis: list of (label, value) tuples. Renders as a grid of cells."""
    cells = []
    row = []
    for i, (label, value) in enumerate(kpis):
        cell = [
            Paragraph(label.upper(), styles["kpi_label"]),
            Spacer(1, 2),
            Paragraph(str(value), styles["kpi_value"]),
        ]
        row.append(cell)
        if len(row) == cols:
            cells.append(row)
            row = []
    if row:
        while len(row) < cols:
            row.append([Paragraph("", styles["kpi_label"])])
        cells.append(row)

    tbl = Table(cells, colWidths=[(170 * mm) / cols] * cols)
    tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, C_BORDER),
        ("LINEAFTER", (0, 0), (-2, -1), 0.5, C_BORDER),
        ("BACKGROUND", (0, 0), (-1, -1), C_CREAM),
    ]))
    return tbl


def _data_table(headers, rows, styles, col_widths=None, highlight_first_col=True):
    """Generic styled table."""
    header_cells = [Paragraph(f"<b>{h}</b>", ParagraphStyle(
        "th", fontName="Helvetica-Bold", fontSize=8, textColor=C_CREAM, leading=10))
        for h in headers]
    data = [header_cells]
    for r in rows:
        cells = []
        for j, val in enumerate(r):
            style = ParagraphStyle(
                "td", fontName="Helvetica" if j > 0 or not highlight_first_col else "Helvetica-Bold",
                fontSize=8, textColor=C_TEXT, leading=11)
            cells.append(Paragraph(str(val), style))
        data.append(cells)

    n = len(headers)
    if col_widths is None:
        col_widths = [170 * mm / n] * n
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_GREEN),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_CREAM]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, C_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def _header_footer(canvas, doc, strategy_name):
    canvas.saveState()
    # Header band
    canvas.setFillColor(C_GREEN)
    canvas.rect(0, A4[1] - 22 * mm, A4[0], 22 * mm, fill=1, stroke=0)
    canvas.setFillColor(C_CREAM)
    canvas.setFont("Times-Roman", 14)
    canvas.drawString(20 * mm, A4[1] - 14 * mm, "Oakwood Capital")
    canvas.setFillColor(C_SAGE)
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(A4[0] - 20 * mm, A4[1] - 11 * mm, "STRATEGY RESEARCH PLATFORM")
    canvas.drawRightString(A4[0] - 20 * mm, A4[1] - 15 * mm, "INTERNAL · CONFIDENTIAL")
    # Footer
    canvas.setFillColor(C_MUTED)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(20 * mm, 12 * mm,
                      "For Illustrative Purposes · Not Investment Advice")
    canvas.drawCentredString(A4[0] / 2, 12 * mm, strategy_name)
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(C_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 15 * mm, A4[0] - 20 * mm, 15 * mm)
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
):
    """Build the PDF and return raw bytes."""
    styles = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=30 * mm, bottomMargin=20 * mm,
        leftMargin=20 * mm, rightMargin=20 * mm,
        title=f"Oakwood Capital — {strategy_name}",
        author="Oakwood Capital",
    )
    story = []

    # ===== PAGE 1: Cover + KPIs =====
    story.append(Paragraph(strategy_name, styles["title"]))
    story.append(Paragraph(strategy_subtitle, styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_GOLD, spaceAfter=4))
    story.append(Paragraph(
        f"Backtest Period: {period_str} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        styles["small"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Performance Summary (Net of Fees)", styles["h2"]))
    story.append(_kpi_grid(kpis_performance, styles, cols=4))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Risk &amp; Risk-Adjusted Metrics", styles["h2"]))
    story.append(_kpi_grid(kpis_risk, styles, cols=4))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Fee Summary", styles["h2"]))
    story.append(_kpi_grid(fee_summary, styles, cols=4))

    story.append(PageBreak())

    # ===== PAGE 2: Charts =====
    story.append(Paragraph("Portfolio Evolution &amp; Risk Charts", styles["h2"]))
    any_chart = False
    for title, png_bytes in figures:
        img = _png_to_image(png_bytes, width_mm=170, height_mm=80)
        if img is not None:
            any_chart = True
            story.append(Paragraph(title, styles["h3"]))
            story.append(img)
            story.append(Spacer(1, 8))
    if not any_chart:
        story.append(Paragraph(
            "Charts could not be embedded in this PDF (chart rendering engine "
            "unavailable in the current environment). All numerical data is "
            "provided in full in the tables on the following pages.",
            styles["body"]))

    story.append(PageBreak())

    # ===== PAGE 3: Detailed tables =====
    story.append(Paragraph("Detailed Risk Metrics", styles["h2"]))
    if risk_table_rows:
        story.append(_data_table(risk_table_headers, risk_table_rows, styles,
                                 col_widths=[55 * mm, 40 * mm, 40 * mm, 35 * mm]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Performance Fee Crystallization Detail", styles["h2"]))
    if fee_table_rows:
        story.append(_data_table(fee_table_headers, fee_table_rows, styles))
    else:
        story.append(Paragraph("No performance fees were crystallized in this period.",
                               styles["body"]))

    story.append(PageBreak())

    # ===== PAGE 4: Methodology, Holdings, Disclaimer =====
    story.append(Paragraph("Methodology &amp; Parameters", styles["h2"]))
    if params_summary:
        prows = [[k, v] for k, v in params_summary]
        story.append(_data_table(["Parameter", "Value"], prows, styles,
                                 col_widths=[85 * mm, 85 * mm]))
    story.append(Spacer(1, 12))

    if universe_rows:
        story.append(Paragraph("Investment Universe", styles["h2"]))
        story.append(_data_table(["Constituent", "Ticker", "Sector"], universe_rows, styles,
                                 col_widths=[70 * mm, 45 * mm, 55 * mm]))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Important Disclosures", styles["h2"]))
    disclaimer_paragraphs = [
        "This document has been prepared by Oakwood Capital for illustrative and "
        "informational purposes only. It does not constitute investment advice, a "
        "recommendation, an offer, or a solicitation to buy or sell any security or "
        "financial instrument, nor a basis for any investment decision.",

        "All figures shown are derived from a historical backtest using publicly "
        "available market data. Backtested performance is hypothetical, does not "
        "represent actual trading, and is subject to the benefit of hindsight. "
        "Past performance and simulated past performance are not reliable indicators "
        "of future results. Actual results may differ materially.",

        "The simulation models transaction costs (a configurable per-trade cost in "
        "basis points) and the 35% Swiss withholding tax on dividends, which is "
        "treated as non-reclaimable within the AMC (Actively Managed Certificate) "
        "wrapper — only the net 65% of each gross dividend is reinvested. The same "
        "after-tax dividend basis is applied to the SMI Total Return benchmark for a "
        "consistent comparison. The simulation does not account for market impact, "
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
    ]
    for p in disclaimer_paragraphs:
        story.append(Paragraph(p, styles["disclaimer"]))

    # Build with header/footer on every page
    def on_page(canvas, doc_):
        _header_footer(canvas, doc_, strategy_name)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    buf.seek(0)
    return buf.getvalue()
