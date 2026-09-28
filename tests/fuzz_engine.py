"""Zufallslaeufe gegen die gepatchte Engine. Prueft Invarianten, nicht Werte."""
import numpy as np, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pruefung_engine import lade, testdaten, jahresenden, quartalsenden, SEITE

neu = lade(SEITE)
rng = random.Random(4711)
fehler = []

for i in range(150):
    n_titel = rng.choice([1, 2, 5, 20])
    prices, divs, btc, fx, gew = testdaten(n_titel=n_titel, jahre=rng.choice([2,4,6]),
                                           saat=rng.randrange(10**6))
    kap = rng.choice([25_000.0, 250_000.0, 1_000_000.0, 50_000_000.0])
    minfee = rng.choice([0.0, 25.0, 75.0, 500.0, 5_000.0])
    cap = rng.choice([None, 0.18, 0.30])
    modus = rng.choice(["dividend", "withdrawal"])
    satz = rng.choice([0.0, 0.00161, 0.0025, 0.005, 0.05])
    entn_n = rng.choice([1, 3, 6, 12])
    fluss = rng.choice([0.0, 0.005, 0.01, 0.03, -0.005, -0.02])
    fluss_chf = rng.choice([0.0, 0.0, 25_000.0, -10_000.0])
    netting = rng.choice([True, False])
    if cap and n_titel*cap < 1.0:
        cap = None
    try:
        ts, txs, evts = neu["run_strategy"](
            prices, divs, btc, fx, kap, gew,
            rng.choice([0.0, 0.15, 0.45]), rng.choice([0.25, 0.60]),
            rng.choice([0.0, 0.15]),
            jahresenden(prices.index), rng.choice([1, 6, 24]),
            tx_cost_bps=rng.choice([0.0, 8.0, 50.0]),
            cap_dates_set=quartalsenden(prices.index), weight_cap=cap,
            min_fee_chf=minfee, fx_fee_bps=rng.choice([0.0, 30.0]),
            min_order_chf=rng.choice([0.0, 500.0, 25_000.0]),
            harvest_mode=modus,
            withdrawal_pct_monthly=(satz if modus == "withdrawal" else 0.0),
            withdrawal_every_n_months=entn_n,
            monthly_flow_pct=fluss, monthly_flow_chf=fluss_chf,
            netting=netting)
    except Exception as e:
        fehler.append("Lauf %d: Ausnahme %r  (Titel=%d Kapital=%.0f Mindest=%.0f Modus=%s)"
                      % (i, e, n_titel, kap, minfee, modus)); continue
    if ts.empty:
        continue
    etikett = ("Lauf %d (Titel=%d Kapital=%.0f Mindest=%.0f Modus=%s Satz=%.4f "
               "Fluss=%.3f/%.0f Netting=%s)" % (
        i, n_titel, kap, minfee, modus, satz, fluss, fluss_chf, netting))
    v = ts["total_value"]
    if (v < -1e-6).any():
        fehler.append(etikett + ": negativer NAV")
    if not np.isfinite(v).all():
        fehler.append(etikett + ": NaN oder Inf im NAV")
    if ts.attrs["total_tx_costs"] < -1e-9:
        fehler.append(etikett + ": negative Kosten")
    f = ts.attrs["attribution"]["reconciliation_error"]
    _tol = max(1.0, abs(kap)*1e-6, abs(ts.attrs.get("net_flow", 0.0))*1e-6)
    if abs(f) > _tol:
        fehler.append(etikett + ": Zerlegungsfehler %.4f" % f)
    s = ts.attrs["cost_stats"]
    if s["lines_at_min"] > s["lines"]:
        fehler.append(etikett + ": mehr Mindestgebuehren als Orderzeilen")
    if minfee == 0.0 and s["lines_at_min"] > 0:
        fehler.append(etikett + ": Mindestgebuehr gezaehlt, obwohl keine gesetzt")
    if (ts["btc_held"] < -1e-9).any():
        fehler.append(etikett + ": negativer Bitcoinbestand")
    if (ts["dividend_cash"] < -1e-6).any():
        fehler.append(etikett + ": negative Dividendenkasse")
    if (ts["smi_value"] < -1e-6).any():
        fehler.append(etikett + ": negativer Aktienteil")
    if modus == "withdrawal" and satz == 0.0 and ts.attrs["attribution"]["btc_dca_invested"] > 1e-6:
        fehler.append(etikett + ": Entnahme trotz Satz 0")
    if "nav_per_unit" not in ts.columns:
        fehler.append(etikett + ": Anteilswert fehlt")
    elif (ts["nav_per_unit"] < 0).any() or not np.isfinite(ts["nav_per_unit"]).all():
        fehler.append(etikett + ": Anteilswert negativ oder nicht endlich")
    if fluss == 0.0 and fluss_chf == 0.0 and abs(ts.attrs.get("net_flow", 0.0)) > 1e-6:
        fehler.append(etikett + ": Nettofluss trotz Fluss 0")
    if ts.attrs.get("netting") != netting:
        fehler.append(etikett + ": Netting nicht uebernommen")

print("150 Zufallslaeufe abgeschlossen.")
if fehler:
    print("Verletzte Invarianten: %d" % len(fehler))
    for f in fehler[:15]: print("   " + f)
    sys.exit(1)
print("Keine Invariante verletzt.")
