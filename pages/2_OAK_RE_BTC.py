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
      Verkaufserlöse amortisieren die Hypothek (falls vorhanden), sonst
      zusätzliche Immobilien-Exposure.
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
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from pdf_report import (build_bilingual_tearsheet,
                        render_line_chart, render_bar_chart,
                        render_scatter_chart,
                        compute_period_returns, identify_top_drawdowns,
                        get_font_status)

st.set_page_config(page_title="OAK RE/BTC — AMC Backtesting",
                   page_icon="🏠", layout="wide")

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

    # Per-period hurdle rate (annual hurdle pro-rated to the crystallization period)
    period_hurdle = hwm_hurdle / periods_per_year if hwm_hurdle > 0 else 0.0

    daily_mgmt = mgmt_fee_annual / 252.0
    net = pd.Series(index=gross_values.index, dtype=float)
    net.iloc[0] = float(initial_capital)
    hwm = float(initial_capital)            # plain high water mark (post-fee highs)
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

            # The hurdle-grown threshold the NAV must clear this period
            hurdle_threshold = hwm * (1.0 + period_hurdle)

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
    years = n_days / 252.0  # for annualizing volatility (trading days)
    # CAGR must use CALENDAR time so it matches the KPI boxes exactly.
    cal_years = (values.index[-1] - values.index[0]).days / 365.25

    start_val = float(base_value) if base_value else float(values.iloc[0])
    total_return = float(values.iloc[-1] / start_val - 1)
    cagr = float((values.iloc[-1] / start_val) ** (1 / cal_years) - 1) if cal_years > 0 else 0.0
    vol_ann = float(returns.std() * np.sqrt(252))

    sharpe = (cagr - risk_free_rate) / vol_ann if vol_ann > 0 else 0.0

    downside = returns[returns < 0]
    downside_vol = float(downside.std() * np.sqrt(252)) if not downside.empty else 0.0
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
      net_yield            net rental yield p.a. on CURRENT property value
                           (pre-computed externally: vacancy, operating costs
                           and any financing are already netted out)
      lower_threshold      BTC weight below which the boosted rate applies
      upper_threshold      BTC weight cap
      base_invest_rate     share of accumulated net rent invested into BTC
      boost_invest_rate    share invested while BTC weight < lower_threshold
      rent_to_btc_freq     "M" (monthly) or "Q" (quarterly) — DCA dates
      btc_to_cash_freq     "M" or "Q" — sell-rule check dates
      sell_on_upper        bool — sell BTC down to the upper threshold into
                           CHF cash on check dates when the weight exceeds it
      tx_cost_bps          transaction cost on BTC trades (bps)

    Zone rule applied on each DCA date (pre-trade BTC weight w):
        w < lower_threshold   ->  boost_invest_rate
        w > upper_threshold   ->  0%  (rent stays in cash)
        otherwise             ->  base_invest_rate
    Non-invested rent and all sale proceeds accumulate as CHF cash (0%
    interest), which is never re-invested — a one-way volatility buffer.

    Returns a daily DataFrame with columns: total_value, re_value, btc_value,
    cash, property_value, net_cf_monthly (month-end rows), btc_buys, btc_sells.
    """
    idx = btc_chf.index.intersection(prop_index_daily.index).sort_values()
    if len(idx) < 30:
        return pd.DataFrame()
    btc = btc_chf.reindex(idx).ffill()
    pidx = prop_index_daily.reindex(idx).ffill()

    cap = float(params["initial_capital"])
    tx = float(params.get("tx_cost_bps", 0.0)) / 10000.0

    btc_chf0 = cap * float(params["initial_btc_pct"])
    btc_units = (btc_chf0 * (1 - tx)) / btc.iloc[0] if btc_chf0 > 0 else 0.0
    prop_units = (cap - btc_chf0) / pidx.iloc[0]

    cash = 0.0        # one-way buffer: rent remainders + sale proceeds
    rent_pool = 0.0   # rent accumulated since the last DCA date

    ny = float(params["net_yield"])
    lo = float(params["lower_threshold"])
    up = float(params["upper_threshold"])
    base_r = float(params["base_invest_rate"])
    boost_r = float(params["boost_invest_rate"])
    f_dca = params.get("rent_to_btc_freq", "M")
    f_sell = params.get("btc_to_cash_freq", "Q")
    sell_on = bool(params.get("sell_on_upper", True))

    rows = []
    for i, d in enumerate(idx):
        p_val = prop_units * pidx.loc[d]
        b_val = btc_units * btc.loc[d]
        net_cf = np.nan
        buys = 0.0
        sells = 0.0

        is_me = (i == len(idx) - 1) or (idx[i + 1].to_period("M") != d.to_period("M"))
        is_qe = is_me and d.month in (3, 6, 9, 12)

        # ---- month-end: net rent accrues into the rent pool ----------------
        if is_me:
            net_cf = ny / 12.0 * p_val
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
        })

    return pd.DataFrame(rows).set_index("date")


def run_re_only(prop_index_daily, ref_index, params):
    """Benchmark: identical unlevered property model WITHOUT BTC — the full
    net rental income is reinvested into additional property units each
    month. Isolates the contribution of the BTC/cash rebalancing sleeve."""
    idx = ref_index.intersection(prop_index_daily.index).sort_values()
    pidx = prop_index_daily.reindex(idx).ffill()
    cap = float(params["initial_capital"])
    ny = float(params["net_yield"])
    prop_units = cap / pidx.iloc[0]
    vals = []
    for i, d in enumerate(idx):
        p_val = prop_units * pidx.loc[d]
        is_me = (i == len(idx) - 1) or (idx[i + 1].to_period("M") != d.to_period("M"))
        if is_me:
            prop_units += (ny / 12.0 * p_val) / pidx.loc[d]
            p_val = prop_units * pidx.loc[d]
        vals.append(p_val)
    return pd.Series(vals, index=idx)
# ===========================================================================
# UI
# ===========================================================================
st.title("OAK RE/BTC — AMC Backtesting")
st.caption("Schweizer Wohnimmobilien mit struktureller Bitcoin-Allokation — "
           "die Kapitalwerte folgen dem SNB-Wohnimmobilienpreisindex, die "
           "Nettoerträge einer eigenen Parametrik. Parametrische Simulation; "
           "Details unter Methodik & Hinweise.")

with st.sidebar:
    st.markdown("### Portfolio-Setup")
    initial_capital = st.number_input("Startkapital (CHF)", 100_000, 100_000_000,
                                      1_000_000, step=100_000)
    initial_btc_pct = st.slider("Initial BTC Allokation (%)", 0, 50, 15, 1) / 100.0
    net_yield = st.slider("Nettomietrendite (% p.a.)", 0.5, 6.0, 3.0, 0.1,
                          help="Extern vorberechnet — nach Leerstand, "
                               "Bewirtschaftung, Unterhalt und Finanzierung. "
                               "Bezieht sich auf den aktuellen "
                               "Liegenschaftswert.") / 100.0

    st.markdown("### Rebalancing-Regeln (Nettomieteinnahmen)")
    lower_threshold = st.slider("Lower BTC Threshold (%)", 0, 40, 10, 1) / 100.0
    upper_threshold = st.slider("Upper BTC Threshold (%)", 5, 75, 25, 1) / 100.0
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
        help="Verkaufserlöse bleiben als CHF-Cash liegen und werden nicht "
             "reinvestiert — der wachsende Cash-Bestand dämpft die "
             "Volatilität.")
    if lower_threshold >= upper_threshold:
        st.error("Lower Threshold muss kleiner als Upper Threshold sein.")

    st.markdown("### Kosten & Gebühren (AMC)")
    tx_cost_bps = st.slider("Transaktionskosten BTC (bps)", 0, 50, 10, 1)
    mgmt_fee = st.slider("Management Fee (% p.a.)", 0.0, 3.0, 1.5, 0.05) / 100.0
    perf_fee = st.slider("Performance Fee (%)", 0, 30, 15, 1) / 100.0
    hurdle = st.slider("Hurdle (Jahr 1, %)", 0.0, 10.0, 5.0, 0.5) / 100.0

    st.markdown("### Zeitraum")
    start_date = st.date_input("Start", date(2018, 1, 3))
    end_date = st.date_input("Ende", date.today())

# --------------------------------------------------------------------------
# Data: SNB index (with manual-CSV fallback) + BTC in CHF
# --------------------------------------------------------------------------
snb_catalog, snb_error = {}, None
try:
    snb_catalog = fetch_snb_catalog()
except Exception as e:  # network blocked / API change → manual fallback
    snb_error = str(e)

if snb_error:
    st.warning(f"SNB-API nicht erreichbar ({snb_error}). "
               "Lade die CSV manuell von data.snb.ch (Cube *plimoinchq*) hoch.")
up = st.file_uploader("Optional: SNB-CSV manuell (data.snb.ch → Immobilienpreisindizes → Download CSV)",
                      type=["csv"])
if up is not None:
    try:
        snb_catalog = snb_series_catalog(_parse_snb_csv(up.getvalue().decode("utf-8-sig")))
        st.success(f"{len(snb_catalog)} Serien aus Upload geladen.")
    except Exception as e:
        st.error(f"CSV konnte nicht geparst werden: {e}")

if not snb_catalog:
    st.info("Keine SNB-Daten verfügbar — Backtest kann nicht starten.")
    st.stop()

labels = sorted(snb_catalog.keys())
default_i = next((i for i, l in enumerate(labels)
                  if "wohnliegenschaft" in l.lower() or "mehrfamilien" in l.lower()), 0)
series_label = st.selectbox("SNB-Indexserie (Kapitalwert-Entwicklung)", labels, index=default_i)
snb_q = snb_catalog[series_label]
st.caption(f"Serie: {series_label} · {snb_q.index[0]:%Y-%m} bis {snb_q.index[-1]:%Y-%m} "
           f"({len(snb_q)} Quartale, linear auf Tagesbasis interpoliert)")

run = st.button("Backtest starten", type="primary",
                disabled=(lower_threshold >= upper_threshold))
if not run:
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

    prop_daily = interpolate_quarterly_to_daily(snb_q, btc_chf.index)

    params = dict(initial_capital=initial_capital, initial_btc_pct=initial_btc_pct,
                  net_yield=net_yield,
                  lower_threshold=lower_threshold, upper_threshold=upper_threshold,
                  base_invest_rate=base_invest_rate, boost_invest_rate=boost_invest_rate,
                  rent_to_btc_freq=("M" if rent_to_btc_freq == "monatlich" else "Q"),
                  btc_to_cash_freq=("M" if btc_to_cash_freq == "monatlich" else "Q"),
                  sell_on_upper=sell_on_upper,
                  tx_cost_bps=tx_cost_bps)

    ts = run_re_btc(prop_daily, btc_chf, params)
    if ts.empty:
        st.error("Simulation lieferte keine Daten (zu kurzer Überlappungszeitraum?).")
        st.stop()

    bench_re = run_re_only(prop_daily, ts.index, params)
    bench_index = (snb_q / snb_q.reindex(
        [ts.index[0]], method="ffill").iloc[0] * initial_capital)
    bench_index_daily = interpolate_quarterly_to_daily(bench_index, ts.index)

    net, total_mgmt, total_perf, fee_events = apply_fees(
        ts["total_value"], initial_capital, mgmt_fee_annual=mgmt_fee,
        perf_fee_rate=perf_fee, hwm_hurdle=hurdle,
        crystallization_freq="Quarterly", hurdle_type="Hard Hurdle")

# --------------------------------------------------------------------------
# KPIs & charts
# --------------------------------------------------------------------------
years = max((net.index[-1] - net.index[0]).days / 365.25, 1e-9)
net_cagr = (net.iloc[-1] / initial_capital) ** (1 / years) - 1
re_cagr = (bench_re.iloc[-1] / initial_capital) ** (1 / years) - 1
m = compute_risk_metrics(net, base_value=initial_capital)
w_btc = ts["btc_value"].iloc[-1] / ts["total_value"].iloc[-1]
w_cash = ts["cash"].iloc[-1] / ts["total_value"].iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Strategie (Netto)", f"CHF {net.iloc[-1]:,.0f}", f"{net_cagr*100:.2f}% CAGR")
c2.metric("RE only (Benchmark)", f"CHF {bench_re.iloc[-1]:,.0f}", f"{re_cagr*100:.2f}% CAGR")
c3.metric("Max Drawdown (Netto)", f"{m['max_drawdown']*100:.2f}%")
c4.metric("BTC / Cash Gewicht aktuell", f"{w_btc*100:.1f}% / {w_cash*100:.1f}%")

st.line_chart(pd.DataFrame({
    "OAK RE/BTC (Netto)": net,
    "RE only (gleiches Modell, ohne BTC)": bench_re,
    "SNB-Index (preis-only, skaliert)": bench_index_daily,
}))

st.area_chart(pd.DataFrame({
    "Wohnimmobilien": ts["re_value"],
    "BTC": ts["btc_value"],
    "CHF Cash": ts["cash"],
}))

with st.expander("Monatliche Netto-Cashflows (Mieterträge → BTC-DCA)"):
    cf = ts["net_cf_monthly"].dropna()
    st.bar_chart(cf)

with st.expander("Gebühren-Aufstellung je Periode"):
    if fee_events is not None and not fee_events.empty:
        st.dataframe(fee_events)
    st.write(f"Mgmt: CHF {total_mgmt:,.0f} · Perf: CHF {total_perf:,.0f} · "
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

        line = render_line_chart([
            ("OAK RE/BTC (Net of Fees)", net, "#B8954A", "-"),
            ("RE only (same model, no BTC)", bench_re, "#7C8978", "--"),
            ("SNB Residential Index (price only)", bench_index_daily, "#999999", ":"),
        ], annotate_end=True, fill_first=True)

        dd = compute_drawdown(net)
        dd_b = compute_drawdown(bench_re)
        dd_chart = render_line_chart([
            ("Strategy (Net)", dd, "#B8954A", "-"),
            ("RE only", dd_b, "#7C8978", "--"),
        ])

        yearly = net.resample("YE").last()
        yearly_ret = yearly.pct_change()
        first_year_ret = yearly.iloc[0] / initial_capital - 1
        yearly_ret.iloc[0] = first_year_ret
        ylabels = [str(y.year) + ("*" if i == len(yearly_ret) - 1 else "")
                   for i, y in enumerate(yearly_ret.index)]
        bar = render_bar_chart(ylabels, [v * 100 for v in yearly_ret.values],
                               hurdle=hurdle * 100)

        mb = compute_risk_metrics(bench_re, base_value=initial_capital)
        scatter = render_scatter_chart([
            ("Strategy", m["vol_ann"] * 100, net_cagr * 100, "#B8954A", "o"),
            ("RE only", mb["vol_ann"] * 100, re_cagr * 100, "#7C8978", "s"),
        ])

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
            ("Net Rental Yield", f"{net_yield*100:.1f}% p.a. (pre-computed, on current value)"),
            ("Lower BTC Threshold", f"{lower_threshold*100:.0f}%"),
            ("Upper BTC Threshold", f"{upper_threshold*100:.0f}%"),
            ("Base Investment Rate (net rent)", f"{base_invest_rate*100:.0f}%"),
            ("Boosted Rate below Lower Threshold", f"{boost_invest_rate*100:.0f}%"),
            ("Rent → BTC Allocation", "Monthly" if rent_to_btc_freq == "monatlich" else "Quarterly"),
            ("BTC → Cash Allocation", "Monthly" if btc_to_cash_freq == "monatlich" else "Quarterly"),
            ("Sell Rule (above Upper Threshold)", "Sell down to upper threshold into CHF cash" if sell_on_upper else "Disabled"),
            ("Cash Treatment", "Uninvested CHF, 0% interest (one-way buffer)"),
            ("Transaction Cost (BTC)", f"{tx_cost_bps} bps per trade"),
            ("Management Fee", f"{mgmt_fee*100:.2f}% p.a."),
            ("Performance Fee", f"{perf_fee*100:.0f}% (Quarterly, Hard Hurdle {hurdle*100:.1f}% Yr 1)"),
        ]
        universe_rows = [
            ["Bitcoin", "BTC", "Digital Assets",
             f"Band {lower_threshold*100:.0f}–{upper_threshold*100:.0f}%"],
            ["CH Wohnliegenschaften (parametrisch)", "SNB plimoinchq", "Residential Real Estate",
             f"{(1-initial_btc_pct)*100:.0f}% initial"],
            ["CHF Cash", "—", "Cash",
             "Rebalancing-Puffer · 0%"],
        ]
        disc_de = [
            "Dieses Dokument wurde von Oakwood Capital ausschliesslich zu illustrativen und informativen Zwecken erstellt. Es stellt weder eine Anlageberatung, eine Empfehlung, ein Angebot noch eine Aufforderung zum Kauf oder Verkauf eines Finanzinstruments dar.",
            "OAK RE/BTC ist eine parametrische Simulation und kein marktdatenbasierter Backtest. Die Wertentwicklung des Immobilienteils folgt einem quartalsweise erhobenen Bewertungsindex der SNB-Datenplattform, der für die Simulation linear auf Tagesbasis interpoliert wird. Bewertungsindizes sind geglättet und unterzeichnen die tatsächliche Volatilität und die Drawdowns von Immobilienanlagen erheblich; Volatilität, Sharpe Ratio und Drawdown-Kennzahlen sind deshalb nicht mit marktbasierten Strategien vergleichbar. Die Nettomietrendite ist eine extern vorberechnete Annahme — nach Leerstand, Bewirtschaftung, Unterhalt und Finanzierung — und kein realisierter Wert; eine allfällige Fremdfinanzierung der Liegenschaften ist im Modell nicht abgebildet.",
            "Der Bitcoin-Anteil basiert auf historischen Marktpreisen (BTC/USD, in Schweizer Franken umgerechnet). Digitale Vermögenswerte sind hochvolatil und können zum Totalverlust des eingesetzten Kapitals führen. Der CHF-Cash-Puffer wird unverzinst gehalten; seine dämpfende Wirkung auf die Volatilität geht zulasten der erwarteten Rendite.",
            "Die simulierte Performance ist hypothetisch, unterliegt dem Vorteil der Rückschau und ist kein verlässlicher Indikator für zukünftige Ergebnisse. Die ausgewiesenen Zahlen verstehen sich nach Abzug der angegebenen Management- und Performance-Gebühren; Steuern — insbesondere Grundstückgewinn-, Liegenschafts- und Einkommenssteuern — sind nicht modelliert.",
            "Dieses Material ist streng vertraulich und ausschliesslich für den Empfänger bestimmt. Es darf ohne vorherige schriftliche Zustimmung von Oakwood Capital weder reproduziert noch verbreitet werden.",
        ]
        disc_en = [
            "This document has been prepared by Oakwood Capital for illustrative and informational purposes only. It does not constitute investment advice, a recommendation, an offer, or a solicitation to buy or sell any financial instrument.",
            "OAK RE/BTC is a parametric simulation, not a market-data backtest. Capital values of the property sleeve follow a quarterly valuation index from the SNB data portal, linearly interpolated to daily frequency for the simulation. Valuation indices are smoothed and materially understate the true volatility and drawdowns of real estate investments; volatility, Sharpe ratio and drawdown figures are therefore not comparable to market-priced strategies. The net rental yield is an externally pre-computed assumption — after vacancy, operating costs, maintenance and financing — not a realized figure; any debt financing of the properties is not modelled.",
            "The Bitcoin sleeve is based on historical market prices (BTC/USD converted to CHF). Digital assets are highly volatile and may result in total loss. The CHF cash buffer is held uninvested; its dampening effect on volatility comes at the cost of expected return.",
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
            risk_table_headers=["Metric", "Strategy (Net)", "RE only"],
            risk_table_rows=[
                ["Total Return", f"{(net.iloc[-1]/initial_capital-1)*100:.2f}%",
                 f"{(bench_re.iloc[-1]/initial_capital-1)*100:.2f}%"],
                ["CAGR", f"{net_cagr*100:.2f}%", f"{re_cagr*100:.2f}%"],
                ["Volatility*", f"{m['vol_ann']*100:.2f}%", f"{mb['vol_ann']*100:.2f}%"],
                ["Max Drawdown*", f"{m['max_drawdown']*100:.2f}%", f"{mb['max_drawdown']*100:.2f}%"],
                ["Sharpe Ratio*", f"{m['sharpe']:.2f}", f"{mb['sharpe']:.2f}"],
                ["Sortino Ratio*", f"{m['sortino']:.2f}", f"{mb['sortino']:.2f}"],
            ],
            fee_table_headers=["Period", "Mgmt Fee", "Perf Fee", "Total Cost"],
            fee_table_rows=fee_rows,
            figures=[("Portfolio Evolution vs. RE-only & SNB Index", line),
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
