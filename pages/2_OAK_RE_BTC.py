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
SNB_CSV_URL = "https://data.snb.ch/api/cube/plimoinchq/data/csv/de"


def _parse_snb_csv(text):
    """Parse an SNB data-portal CSV (semicolon-separated, metadata header
    lines before the actual 'Date;...' header). Returns a DataFrame with
    columns: Date (quarter string like '2020-Q3'), one column per dimension,
    and 'Value'. Robust against the exact number of dimension columns."""
    lines = text.splitlines()
    header_i = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Date;"):
            header_i = i
            break
    if header_i is None:
        raise ValueError("SNB-CSV: keine 'Date;'-Headerzeile gefunden")
    body = "\n".join(lines[header_i:])
    df = pd.read_csv(io.StringIO(body), sep=";")
    df.columns = [c.strip() for c in df.columns]
    if "Value" not in df.columns:
        raise ValueError("SNB-CSV: 'Value'-Spalte fehlt")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])
    return df


def _quarter_to_timestamp(qstr):
    """'2020-Q3' -> Timestamp of quarter END (valuation effective date)."""
    qstr = str(qstr).strip()
    per = pd.Period(qstr.replace("-Q", "Q"), freq="Q")
    return per.to_timestamp(how="end").normalize()


def snb_series_catalog(df):
    """From a parsed SNB frame, build {series_label: quarterly pd.Series}.
    The label is the join of all dimension columns (everything that is not
    Date/Value), so the user can pick the exact index variant."""
    dim_cols = [c for c in df.columns if c not in ("Date", "Value")]
    out = {}
    if dim_cols:
        for key, sub in df.groupby(dim_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            label = " · ".join(str(k) for k in key if pd.notna(k) and str(k).strip())
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
    """Fetch the full SNB residential price-index cube and return the
    series catalog. Cached for a day (quarterly data)."""
    r = requests.get(SNB_CSV_URL, timeout=30)
    r.raise_for_status()
    return snb_series_catalog(_parse_snb_csv(r.text))


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
    """Daily simulation.

    params (dict):
      initial_capital      total CHF at t0
      initial_btc_pct      fraction of capital in BTC at t0
      upper_threshold      BTC sell-down trigger (fraction of NAV)
      target_btc_pct       BTC target after sell-down
      ltv                  mortgage loan-to-value on the property at t0
      mortgage_rate        annual interest rate on the mortgage
      amort_rate           annual amortization, as fraction of INITIAL mortgage
      gross_yield          gross rental yield p.a. on CURRENT property value
      vacancy              vacancy / loss-of-rent rate (fraction of gross rent)
      opex_pct             operating costs (fraction of gross rent)
      tx_cost_bps          transaction cost on BTC trades (bps)

    Mechanics:
      * Property is held as UNITS of the SNB index: value = units * index.
      * Month-end: net cash flow = gross rent*(1-vacancy) - opex - interest
        - amortization. Positive cash (incl. carried balance) buys BTC (DCA).
        Negative cash flow is carried as a (interest-free) cash deficit.
      * Daily: if BTC share of NAV > upper_threshold -> sell down to target;
        proceeds first repay the mortgage, any remainder buys property units.

    Returns DataFrame indexed daily with columns:
      total_value, re_value (property - mortgage), btc_value, cash,
      property_value, mortgage, net_cf_monthly (on month-end rows),
      btc_buys, btc_sells
    """
    idx = btc_chf.index.intersection(prop_index_daily.index)
    idx = idx.sort_values()
    if len(idx) < 30:
        return pd.DataFrame()
    btc = btc_chf.reindex(idx).ffill()
    pidx = prop_index_daily.reindex(idx).ffill()

    cap = float(params["initial_capital"])
    tx = float(params.get("tx_cost_bps", 0.0)) / 10000.0

    # --- t0 setup -----------------------------------------------------------
    btc_chf0 = cap * float(params["initial_btc_pct"])
    btc_units = (btc_chf0 * (1 - tx)) / btc.iloc[0] if btc_chf0 > 0 else 0.0

    re_equity0 = cap - btc_chf0
    ltv = float(params.get("ltv", 0.0))
    prop_value0 = re_equity0 / (1.0 - ltv) if ltv < 1.0 else re_equity0
    mortgage = prop_value0 * ltv
    mortgage0 = mortgage
    prop_units = prop_value0 / pidx.iloc[0]

    cash = 0.0

    g_yield = float(params["gross_yield"])
    vac = float(params["vacancy"])
    opex = float(params["opex_pct"])
    m_rate = float(params["mortgage_rate"])
    a_rate = float(params.get("amort_rate", 0.0))
    upper = float(params["upper_threshold"])
    target = float(params["target_btc_pct"])

    rows = []
    month_marker = idx[0].to_period("M")

    for i, d in enumerate(idx):
        p_val = prop_units * pidx.loc[d]
        b_val = btc_units * btc.loc[d]
        net_cf = np.nan
        buys = 0.0
        sells = 0.0

        # ---- month-end: rental cash flow + DCA into BTC --------------------
        is_month_end = (i == len(idx) - 1) or (idx[i + 1].to_period("M") != d.to_period("M"))
        if is_month_end:
            gross_rent = g_yield / 12.0 * p_val
            eff_rent = gross_rent * (1.0 - vac)
            cost = gross_rent * opex
            interest = m_rate / 12.0 * mortgage
            amort = min(a_rate / 12.0 * mortgage0, mortgage)
            net_cf = eff_rent - cost - interest - amort
            mortgage -= amort
            cash += net_cf
            if cash > 0:
                spend = cash
                btc_units += (spend * (1 - tx)) / btc.loc[d]
                buys = spend
                cash = 0.0
            b_val = btc_units * btc.loc[d]

        # ---- daily threshold check on BTC share -----------------------------
        nav = p_val - mortgage + b_val + cash
        if nav > 0 and b_val / nav > upper:
            target_btc_chf = nav * target
            sell_chf = b_val - target_btc_chf
            btc_units -= sell_chf / btc.loc[d]
            proceeds = sell_chf * (1 - tx)
            sells = sell_chf
            repay = min(proceeds, mortgage)
            mortgage -= repay
            proceeds -= repay
            if proceeds > 0:  # unlevered remainder -> additional property
                prop_units += proceeds / pidx.loc[d]
            p_val = prop_units * pidx.loc[d]
            b_val = btc_units * btc.loc[d]
            nav = p_val - mortgage + b_val + cash

        rows.append({
            "date": d,
            "total_value": nav,
            "re_value": p_val - mortgage,
            "btc_value": b_val,
            "cash": cash,
            "property_value": p_val,
            "mortgage": mortgage,
            "net_cf_monthly": net_cf,
            "btc_buys": buys,
            "btc_sells": sells,
        })

    out = pd.DataFrame(rows).set_index("date")
    return out


def run_re_only(prop_index_daily, ref_index, params):
    """Benchmark: identical parametric property model WITHOUT BTC — net rental
    cash flows are reinvested into additional property units (after interest
    and amortization), same leverage. Isolates the BTC contribution."""
    p = dict(params)
    idx = ref_index.intersection(prop_index_daily.index).sort_values()
    pidx = prop_index_daily.reindex(idx).ffill()

    cap = float(p["initial_capital"])
    ltv = float(p.get("ltv", 0.0))
    prop_value0 = cap / (1.0 - ltv) if ltv < 1.0 else cap
    mortgage = prop_value0 * ltv
    mortgage0 = mortgage
    prop_units = prop_value0 / pidx.iloc[0]
    cash = 0.0

    g_yield, vac, opex = float(p["gross_yield"]), float(p["vacancy"]), float(p["opex_pct"])
    m_rate, a_rate = float(p["mortgage_rate"]), float(p.get("amort_rate", 0.0))

    vals = []
    for i, d in enumerate(idx):
        p_val = prop_units * pidx.loc[d]
        is_month_end = (i == len(idx) - 1) or (idx[i + 1].to_period("M") != d.to_period("M"))
        if is_month_end:
            gross_rent = g_yield / 12.0 * p_val
            net_cf = gross_rent * (1.0 - vac) - gross_rent * opex \
                - m_rate / 12.0 * mortgage - min(a_rate / 12.0 * mortgage0, mortgage)
            mortgage -= min(a_rate / 12.0 * mortgage0, mortgage)
            cash += net_cf
            if cash > 0:
                prop_units += cash / pidx.loc[d]
                cash = 0.0
            p_val = prop_units * pidx.loc[d]
        vals.append(p_val - mortgage + cash)
    return pd.Series(vals, index=idx)


# ===========================================================================
# UI
# ===========================================================================
st.title("OAK RE/BTC — AMC Backtesting")
st.caption("Wohnimmobilien Schweiz (SNB-Index + eigene Netto-Rendite-Parametrik) "
           "mit struktureller Bitcoin-Allokation. Parametrische Simulation — "
           "siehe Methodik.")

with st.sidebar:
    st.markdown("### Kapital & BTC-Sleeve")
    initial_capital = st.number_input("Startkapital (CHF)", 100_000, 100_000_000,
                                      1_000_000, step=100_000)
    initial_btc_pct = st.slider("Initial BTC Allokation (%)", 0, 50, 15, 1) / 100.0
    upper_threshold = st.slider("Upper Threshold — Sell-Down Trigger (%)", 15, 75, 25, 1) / 100.0
    target_btc_pct = st.slider("Target nach Sell-Down (%)", 0, 50, 15, 1) / 100.0
    if target_btc_pct >= upper_threshold:
        st.error("Target muss kleiner als Upper Threshold sein.")

    st.markdown("### Wohnimmobilien — Netto-Rendite-Parametrik")
    gross_yield = st.slider("Bruttomietrendite (% p.a.)", 2.0, 7.0, 4.5, 0.1) / 100.0
    vacancy = st.slider("Leerstand / Mietausfall (%)", 0.0, 10.0, 4.0, 0.5) / 100.0
    opex_pct = st.slider("Bewirtschaftung & Unterhalt (% der Sollmiete)", 5.0, 40.0, 20.0, 1.0) / 100.0

    st.markdown("### Finanzierung")
    ltv = st.slider("Hypothek LTV (%)", 0, 80, 60, 5) / 100.0
    mortgage_rate = st.slider("Hypothekarzins (% p.a.)", 0.5, 5.0, 1.8, 0.1) / 100.0
    amort_rate = st.slider("Amortisation (% der Anfangshypothek p.a.)", 0.0, 3.0, 0.0, 0.25) / 100.0

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
                disabled=(target_btc_pct >= upper_threshold))
if not run:
    st.stop()

with st.spinner("Lade BTC/FX-Daten und simuliere…"):
    btc_usd = fetch_series("BTC-USD", str(start_date), str(end_date))
    fx = fetch_series("CHFUSD=X", str(start_date), str(end_date))
    if btc_usd.empty or fx.empty:
        st.error("BTC/FX-Daten konnten nicht geladen werden.")
        st.stop()
    btc_chf = (btc_usd / fx.reindex(btc_usd.index).ffill()).dropna()
    btc_chf = btc_chf[(btc_chf.index >= pd.Timestamp(start_date))
                      & (btc_chf.index <= pd.Timestamp(end_date))]

    prop_daily = interpolate_quarterly_to_daily(snb_q, btc_chf.index)

    params = dict(initial_capital=initial_capital, initial_btc_pct=initial_btc_pct,
                  upper_threshold=upper_threshold, target_btc_pct=target_btc_pct,
                  ltv=ltv, mortgage_rate=mortgage_rate, amort_rate=amort_rate,
                  gross_yield=gross_yield, vacancy=vacancy, opex_pct=opex_pct,
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
ann_cf = ts["net_cf_monthly"].dropna()
net_equity_yield = (ann_cf.tail(12).sum() / ts["re_value"].iloc[-1]
                    if len(ann_cf) >= 12 and ts["re_value"].iloc[-1] > 0 else np.nan)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Strategie (Netto)", f"CHF {net.iloc[-1]:,.0f}", f"{net_cagr*100:.2f}% CAGR")
c2.metric("RE only (Benchmark)", f"CHF {bench_re.iloc[-1]:,.0f}", f"{re_cagr*100:.2f}% CAGR")
c3.metric("Max Drawdown (Netto)", f"{m['max_drawdown']*100:.2f}%")
c4.metric("Netto-Rendite auf EK (letzte 12M)",
          f"{net_equity_yield*100:.2f}%" if pd.notna(net_equity_yield) else "n/a")

st.line_chart(pd.DataFrame({
    "OAK RE/BTC (Netto)": net,
    "RE only (gleiches Modell, ohne BTC)": bench_re,
    "SNB-Index (preis-only, skaliert)": bench_index_daily,
}))

st.area_chart(pd.DataFrame({
    "RE-Eigenkapital": ts["re_value"],
    "BTC": ts["btc_value"],
}))

with st.expander("Monatliche Netto-Cashflows (Mieterträge → BTC-DCA)"):
    cf = ts["net_cf_monthly"].dropna()
    st.bar_chart(cf)
    if (cf < 0).any():
        st.warning(f"{int((cf < 0).sum())} Monate mit negativem Netto-Cashflow "
                   "(Zins+Kosten > Mieten) — werden als Cash-Defizit vorgetragen.")

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
        exec_de = (f"OAK RE/BTC kombiniert ein parametrisches Schweizer Wohnimmobilien-"
                   f"Portfolio (Kapitalwert gemäss SNB-Immobilienpreisindex, Netto-"
                   f"Eigenkapitalrendite gemäss eigener Parametrik mit {ltv*100:.0f}% "
                   f"Belehnung) mit einer strukturellen Bitcoin-Allokation. Die "
                   f"monatlichen Netto-Mieterträge fliessen via Dollar-Cost-Averaging "
                   f"in Bitcoin; eine Schwellenwert-Regel begrenzt die BTC-Quote auf "
                   f"{upper_threshold*100:.0f}%. Im Simulationszeitraum resultierte "
                   f"ein Netto-CAGR von {net_cagr*100:.1f}% gegenüber "
                   f"{re_cagr*100:.1f}% für das identische Immobilienmodell ohne "
                   f"Bitcoin. Achtung: parametrische Simulation auf Basis eines "
                   f"geglätteten Bewertungsindex — nicht mit kotierten Strategien "
                   f"vergleichbar.")
        exec_en = (f"OAK RE/BTC combines a parametric Swiss residential property "
                   f"portfolio (capital values per the SNB residential price index, "
                   f"net equity yield per our own parametrization at {ltv*100:.0f}% "
                   f"LTV) with a structural Bitcoin allocation. Monthly net rental "
                   f"cash flows are dollar-cost-averaged into Bitcoin; a threshold "
                   f"rule caps the BTC share at {upper_threshold*100:.0f}%. Over the "
                   f"simulation period the strategy delivered a net CAGR of "
                   f"{net_cagr*100:.1f}% versus {re_cagr*100:.1f}% for the identical "
                   f"property model without Bitcoin. Note: parametric simulation on "
                   f"a smoothed valuation index — not comparable to listed strategies.")

        kt_de = [
            f"Netto-CAGR {net_cagr*100:.1f}% vs. {re_cagr*100:.1f}% für RE-only — BTC-Beitrag {excess*100:+.1f}% p.a.",
            f"Netto-Eigenkapitalrendite (letzte 12M) {net_equity_yield*100:.1f}% speist das monatliche BTC-DCA." if pd.notna(net_equity_yield) else "Monatliche Netto-Mieterträge speisen das BTC-DCA.",
            "Parametrische Simulation: geglätteter Bewertungsindex, Risikokennzahlen nicht mit Marktdaten vergleichbar.",
        ]
        kt_en = [
            f"Net CAGR {net_cagr*100:.1f}% vs. {re_cagr*100:.1f}% for RE-only — BTC contribution {excess*100:+.1f}% p.a.",
            f"Net equity yield (last 12M) of {net_equity_yield*100:.1f}% funds the monthly BTC DCA." if pd.notna(net_equity_yield) else "Monthly net rental cash flows fund the BTC DCA.",
            "Parametric simulation: smoothed valuation index — risk metrics not comparable to market-priced strategies.",
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
            ("Gross Rental Yield", f"{gross_yield*100:.1f}% p.a. (on current value)"),
            ("Vacancy / Loss of Rent", f"{vacancy*100:.1f}%"),
            ("Opex & Maintenance", f"{opex_pct*100:.0f}% of gross rent"),
            ("Mortgage LTV", f"{ltv*100:.0f}%"),
            ("Mortgage Rate", f"{mortgage_rate*100:.2f}% p.a."),
            ("Amortization", f"{amort_rate*100:.2f}% of initial mortgage p.a."),
            ("BTC Upper Threshold", f"{upper_threshold*100:.0f}%"),
            ("BTC Target after Rebalance", f"{target_btc_pct*100:.0f}%"),
            ("BTC Sell-Down Proceeds", "Mortgage repayment first, then property"),
            ("Transaction Cost (BTC)", f"{tx_cost_bps} bps per trade"),
            ("Management Fee", f"{mgmt_fee*100:.2f}% p.a."),
            ("Performance Fee", f"{perf_fee*100:.0f}% (Quarterly, Hard Hurdle {hurdle*100:.1f}% Yr 1)"),
        ]

        universe_rows = [
            ["Bitcoin", "BTC", "Digital Assets",
             f"{target_btc_pct*100:.0f}% · cap {upper_threshold*100:.0f}%"],
            ["CH Wohnliegenschaften (parametrisch)", "SNB plimoinchq", "Residential Real Estate",
             f"{(1-initial_btc_pct)*100:.0f}% · LTV {ltv*100:.0f}%"],
        ]

        disc_de = [
            "Dieses Dokument wurde von Oakwood Capital ausschliesslich zu illustrativen und informativen Zwecken erstellt. Es stellt weder eine Anlageberatung, eine Empfehlung, ein Angebot noch eine Aufforderung zum Kauf oder Verkauf eines Finanzinstruments dar.",
            "OAK RE/BTC ist eine PARAMETRISCHE SIMULATION, kein marktdatenbasierter Backtest. Die Kapitalwert-Entwicklung des Immobilienteils folgt einem quartalsweisen Bewertungsindex der SNB-Datenplattform (linear auf Tagesbasis interpoliert); Bewertungsindizes sind geglättet und unterzeichnen die tatsächliche Volatilität und die Drawdowns von Immobilienanlagen erheblich. Volatilität, Sharpe Ratio und Drawdown-Kennzahlen sind daher NICHT mit marktbasierten Strategien vergleichbar. Mieterträge, Kosten, Leerstand, Hypothekarzins und Amortisation sind Annahmen (eigene Parametrik) und keine realisierten Werte.",
            "Der Bitcoin-Anteil basiert auf historischen Marktpreisen (BTC/USD, in CHF konvertiert). Digitale Vermögenswerte sind hochvolatil und können zum Totalverlust führen. Eine Belehnung (Hypothek) hebelt sowohl Gewinne als auch Verluste auf das Eigenkapital; bei negativen Netto-Cashflows entsteht ein Liquiditätsbedarf.",
            "Die simulierte Performance ist hypothetisch, unterliegt dem Vorteil der Rückschau und ist kein verlässlicher Indikator für zukünftige Ergebnisse. Die Performance-Zahlen werden nach Abzug der angegebenen Management- und Performance-Gebühren ausgewiesen. Steuern (insb. Grundstückgewinn-, Liegenschafts- und Einkommenssteuern) sind nicht modelliert.",
            "Dieses Material ist streng vertraulich und ausschliesslich für den Empfänger bestimmt. Es darf ohne vorherige schriftliche Zustimmung von Oakwood Capital weder reproduziert noch verbreitet werden.",
        ]
        disc_en = [
            "This document has been prepared by Oakwood Capital for illustrative and informational purposes only. It does not constitute investment advice, a recommendation, an offer, or a solicitation to buy or sell any financial instrument.",
            "OAK RE/BTC is a PARAMETRIC SIMULATION, not a market-data backtest. Capital values of the property sleeve follow a quarterly valuation index from the SNB data portal (linearly interpolated to daily); valuation indices are smoothed and materially understate the true volatility and drawdowns of real estate investments. Volatility, Sharpe ratio and drawdown figures are therefore NOT comparable to market-priced strategies. Rental income, costs, vacancy, mortgage rate and amortization are assumptions (own parametrization), not realized figures.",
            "The Bitcoin sleeve is based on historical market prices (BTC/USD converted to CHF). Digital assets are highly volatile and may result in total loss. Mortgage leverage amplifies both gains and losses on equity; negative net cash flows create liquidity needs.",
            "Simulated performance is hypothetical, benefits from hindsight, and is not a reliable indicator of future results. Performance figures are shown net of the stated management and performance fees. Taxes (in particular property-gains, property and income taxes) are not modelled.",
            "This material is strictly confidential and intended solely for the recipient. It may not be reproduced or distributed without the prior written consent of Oakwood Capital.",
        ]

        pdf_bytes = build_bilingual_tearsheet(
            strategy_name="OAK RE/BTC",
            strategy_subtitle_de="Schweizer Wohnimmobilien mit struktureller BTC-Allokation — Netto-Mieterträge finanzieren ein diszipliniertes Bitcoin-DCA.",
            strategy_subtitle_en="Swiss residential real estate with a structural BTC allocation — net rental income funds a disciplined Bitcoin DCA.",
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
