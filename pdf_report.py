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
    Image, PageBreak, HRFlowable, KeepTogether,
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


def render_line_chart(series_dict, title="", ylabel="", percent=False, fill_first=False):
    """Render a line chart to PNG bytes using matplotlib (no browser needed).
    series_dict: ordered dict-like list of (label, pandas Series, color, style)."""
    try:
        plt.rcParams["font.family"] = "DejaVu Sans"
        PANEL = "#FBFBF8"   # matches C_PAGE_BG — chart blends into the light page
        fig, ax = plt.subplots(figsize=(9.5, 4.0), dpi=170)
        fig.patch.set_facecolor(PANEL)
        ax.set_facecolor(PANEL)
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
        ax.set_ylabel(ylabel, fontsize=8.5, color="#6B7868", labelpad=8)
        ax.tick_params(labelsize=8, colors="#6B7868", length=0)
        ax.tick_params(axis="both", labelcolor="#6B7868")
        ax.grid(True, color="#E2E4DD", linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#D2D5CC")
        if percent:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        leg = ax.legend(fontsize=8, loc="upper left", frameon=False,
                        handlelength=1.6, handletextpad=0.6, columnspacing=1.2,
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
        ax.set_ylim(ymin - abs(ymin)*0.12, ymax + abs(ymax)*0.15)
        if hurdle is not None:
            ax.axhline(hurdle, color=MPL_GOLD, linewidth=1.3, linestyle="--",
                       label=f"Year-1 Hurdle {hurdle:.0f}%", zorder=2)
            leg = ax.legend(fontsize=8, frameon=False, loc="upper right")
            if leg:
                for t in leg.get_texts():
                    t.set_color("#2A2A26")
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
# Brand palette — LIGHT CLASSIC THEME (reportlab Color objects)
# Warm off-white page, deep green for headers/text, gold accent. Mirrors the
# Oakwood website's clean, light institutional look. Optimised for both screen
# and print.
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
C_WHITE   = colors.white


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
        fontSize=16, textColor=C_GREEN, spaceBefore=16, spaceAfter=8, leading=19,
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
            ("LINEABOVE", (0, 0), (-1, 0), 1.5, C_SAGE),
            ("LINEAFTER", (0, 0), (-2, -1), 0.5, C_BORDER),
        ]
    tbl.setStyle(TableStyle(style))
    return tbl


def _data_table(headers, rows, styles, col_widths=None, highlight_first_col=True):
    """Generic styled table."""
    header_cells = [Paragraph(f"<b>{h}</b>", ParagraphStyle(
        "th", fontName=F_SANS_BOLD, fontSize=8, textColor=C_CREAM, leading=10))
        for h in headers]
    data = [header_cells]
    for r in rows:
        cells = []
        for j, val in enumerate(r):
            style = ParagraphStyle(
                "td", fontName=F_SANS if j > 0 or not highlight_first_col else F_SANS_BOLD,
                fontSize=8, textColor=C_TEXT, leading=11)
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
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_PANEL, C_PAGE_BG]),
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


def _draw_cover(canvas, doc, strategy_name, strategy_subtitle, period_str,
                highlight_kpis, logo_path):
    """Full-bleed dark-green cover page drawn directly on the canvas.
    highlight_kpis: list of up to 3 (label, value) tuples for the hero band."""
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

    # Hero highlight metrics band (up to 3), centered
    if highlight_kpis:
        n = len(highlight_kpis)
        band_w = 170 * mm
        cell_w = band_w / n
        x0 = (W - band_w) / 2
        y_band = 95 * mm
        # divider lines between cells — subtle, semi-transparent sage hairlines
        canvas.setStrokeColor(COVER_SAGE)
        canvas.setStrokeAlpha(0.25)
        canvas.setLineWidth(0.5)
        for i in range(1, n):
            xd = x0 + i * cell_w
            canvas.line(xd, y_band - 2 * mm, xd, y_band + 16 * mm)
        canvas.setStrokeAlpha(1.0)
        for i, (label, value) in enumerate(highlight_kpis):
            cx = x0 + i * cell_w + cell_w / 2
            canvas.setFillColor(C_GOLD)
            canvas.setFont(F_SERIF, 27)
            canvas.drawCentredString(cx, y_band + 8 * mm, str(value))
            canvas.setFillColor(COVER_SAGE)
            canvas.setFont(F_SANS, 7.5)
            canvas.drawCentredString(cx, y_band, label.upper())

    # Period + generation block near the bottom
    canvas.setFillColor(C_CREAMD)
    canvas.setFont(F_SANS, 9)
    canvas.drawCentredString(W / 2, 52 * mm, f"Backtest Period   {period_str}")
    canvas.setFillColor(COVER_SAGE)
    canvas.setFont(F_SANS, 8)
    canvas.drawCentredString(W / 2, 45 * mm,
                             f"Generated {datetime.now().strftime('%d %B %Y, %H:%M')}")

    # Contact + confidential footer mark
    canvas.setFillColor(COVER_SAGE)
    canvas.setFont(F_SANS, 7.5)
    canvas.drawCentredString(W / 2, 26 * mm,
                             "Oakwood Capital Consulting AG  ·  Gotthardstrasse 14  ·  6300 Zug")
    canvas.drawCentredString(W / 2, 21 * mm,
                             "+41 79 250 72 31  ·  info@oakwood-capital.ch  ·  www.oakwood-capital.ch")
    canvas.setFillColor(colors.HexColor("#7C8978"))
    canvas.setFont(F_SANS, 7)
    canvas.drawCentredString(W / 2, 15 * mm,
                             "STRATEGY RESEARCH PLATFORM   ·   INTERNAL · CONFIDENTIAL")
    canvas.setFillColor(C_GOLD)
    canvas.setFont(F_SERIF_ITALIC, 12)
    canvas.drawCentredString(W / 2, 9 * mm, "Oakwood Capital · Quantitative Research")

    canvas.restoreState()


def _header_footer(canvas, doc, strategy_name):
    canvas.saveState()
    W, H = A4
    # Full-page light background (ReportLab doesn't fill it automatically)
    canvas.setFillColor(C_PAGE_BG)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)

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
    canvas.drawRightString(W - 20 * mm, H - 11 * mm, "STRATEGY RESEARCH PLATFORM")
    canvas.drawRightString(W - 20 * mm, H - 15 * mm, "INTERNAL · CONFIDENTIAL")

    # Footer
    canvas.setFillColor(C_MUTED)
    canvas.setFont(F_SANS, 7)
    canvas.drawString(20 * mm, 12 * mm,
                      "For Illustrative Purposes · Not Investment Advice")
    canvas.drawCentredString(W / 2, 12 * mm, strategy_name)
    canvas.drawRightString(W - 20 * mm, 12 * mm, f"{doc.page}")
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
):
    """Build the PDF and return raw bytes."""
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
        story.append(Paragraph("Executive Summary", styles["h2"]))
        story.append(Paragraph(exec_summary, styles["body"]))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Performance Summary", styles["h2"]))
    story.append(Paragraph("Net of fees, transaction costs and 35% dividend withholding tax",
                           styles["h3"]))
    story.append(Spacer(1, 2))
    story.append(_kpi_grid(kpis_performance, styles, cols=4, accent=True))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Risk &amp; Risk-Adjusted Metrics", styles["h2"]))
    story.append(_kpi_grid(kpis_risk, styles, cols=4))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Fee Summary", styles["h2"]))
    story.append(_kpi_grid(fee_summary, styles, cols=4))

    # Monthly returns heatmap — the signature factsheet element
    if monthly_returns:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Monthly Returns (Net, %)", styles["h2"]))
        story.append(Paragraph(
            "Green = positive, red = negative; intensity scales with magnitude. "
            "FY = compounded full-year return.", styles["h3"]))
        story.append(Spacer(1, 3))
        story.append(_monthly_returns_table(monthly_returns, styles))

    story.append(PageBreak())

    # ===== Charts =====
    story.append(Paragraph("Portfolio Evolution &amp; Risk Charts", styles["h2"]))
    any_chart = False
    for title, png_bytes in figures:
        img = _png_to_image(png_bytes, width_mm=168, height_mm=71)
        if img is not None:
            any_chart = True
            block = KeepTogether([
                Paragraph(title, styles["h3"]),
                img,
                Spacer(1, 10),
            ])
            story.append(block)
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
        story.append(_two_col_universe(universe_rows, styles))
        story.append(Spacer(1, 12))

    # Keep the entire disclosures section together on a fresh page
    story.append(PageBreak())
    disc_block = [Paragraph("Important Disclosures", styles["h2"])]
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
                    highlight_kpis, logo_path)

    def on_later(canvas, doc_):
        _header_footer(canvas, doc_, strategy_name)

    doc.build(story, onFirstPage=on_first, onLaterPages=on_later)
    buf.seek(0)
    return buf.getvalue()
