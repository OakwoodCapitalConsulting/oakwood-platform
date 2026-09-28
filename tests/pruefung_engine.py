"""
Prueft die gepatchte Engine gegen die Originalfassung.

Die Streamlit-Seite laesst sich nicht importieren, ohne die ganze Oberflaeche
auszufuehren. Deshalb werden die benoetigten Funktionen ueber den Syntaxbaum
herausgeschnitten und isoliert ausgefuehrt.

Ohne Netzzugang zu Yahoo werden synthetische Kurs- und Dividendenreihen
verwendet. Fuer den Regressionstest genuegt das: beide Fassungen sehen exakt
dieselben Daten.
"""
import ast, sys, math
import numpy as np
import pandas as pd

import os as _os
# Pfade: standardmaessig das Repo eine Ebene ueber diesem Skript (tests/),
# ueberschreibbar mit der Umgebungsvariablen OAK_REPO.
_HIER = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.environ.get("OAK_REPO", _os.path.dirname(_HIER))
SEITE = _os.path.join(REPO, "pages", "1_SMI_Strategy.py")
ORIGINAL = _os.environ.get(
    "OAK_ORIGINAL", _os.path.join(_HIER, "sicherung_1_SMI_Strategy_original.py"))


BRAUCHT = ["_clean_index", "_norm_ts", "_to_series", "run_strategy",
           "synthesize_accumulating"]
KONST = ["WITHHOLDING_TAX", "DIVIDEND_NET_FACTOR"]


def lade(pfad):
    """Schneidet die benoetigten Funktionen und Konstanten heraus."""
    baum = ast.parse(open(pfad, encoding="utf-8").read())
    teile = []
    for knoten in baum.body:
        if isinstance(knoten, ast.FunctionDef) and knoten.name in BRAUCHT:
            teile.append(knoten)
        elif isinstance(knoten, ast.Assign):
            for ziel in knoten.targets:
                if isinstance(ziel, ast.Name) and ziel.id in KONST:
                    teile.append(knoten)
    modul = ast.Module(body=teile, type_ignores=[])
    raum = {"pd": pd, "np": np, "math": math}
    exec(compile(ast.fix_missing_locations(modul), pfad, "exec"), raum)
    return raum


# ------------------------------------------------------------- Testdaten
def testdaten(n_titel=20, jahre=6, saat=7):
    rng = np.random.default_rng(saat)
    tage = pd.bdate_range("2019-01-01", periods=int(jahre*252))
    namen = ["T%02d.SW" % i for i in range(n_titel)]
    # Marktfaktor plus titelspezifisches Rauschen
    markt = rng.normal(0.00030, 0.0090, len(tage))
    kurse = {}
    for i, t in enumerate(namen):
        eigen = rng.normal(0.0, 0.0090, len(tage))
        kurse[t] = 100.0*(1+i*0.1)*np.exp(np.cumsum(markt+eigen))
    prices = pd.DataFrame(kurse, index=tage)

    # Dividenden: je Titel einmal im Jahr, Monat gestreut
    zeilen = []
    for i, t in enumerate(namen):
        monat = 3 + (i % 4)
        for jahr in range(2019, 2019+jahre):
            kand = [d for d in tage if d.year == jahr and d.month == monat]
            if kand:
                d = kand[len(kand)//2]
                zeilen.append({"date": d, "ticker": t,
                               "dividend_per_share": float(prices.loc[d, t])*0.030})
    divs = pd.DataFrame(zeilen)

    btc = pd.Series(30000.0*np.exp(np.cumsum(rng.normal(0.0008, 0.035, len(tage)))),
                    index=tage)
    fx = pd.Series(0.90+np.cumsum(rng.normal(0.0, 0.002, len(tage))), index=tage)
    gew = {t: (16.5 if i == 0 else max(1.0, 14.0-i*0.7)) for i, t in enumerate(namen)}
    return prices, divs, btc, fx, gew


def monatsenden(idx):
    df = pd.DataFrame(index=idx); df["ym"] = df.index.to_period("M")
    return {sub.index[-1] for _, sub in df.groupby("ym")}


def jahresenden(idx):
    return {d for d in monatsenden(idx) if d.month == 9}


def quartalsenden(idx):
    return {d for d in monatsenden(idx) if d.month in (3, 6, 12)}


def lauf(raum, prices, divs, btc, fx, gew, obere_schwelle=0.25,
         tx_cost_bps_override=10.0, **kw):
    return raum["run_strategy"](
        prices, divs, btc, fx, 1_000_000.0, gew,
        0.15, obere_schwelle, 0.15,
        jahresenden(prices.index), 6, tx_cost_bps=tx_cost_bps_override,
        threshold_check_dates_set=None,
        cap_dates_set=quartalsenden(prices.index), weight_cap=0.18, **kw)


# ------------------------------------------------------------- Pruefungen
if __name__ == "__main__":
    if not _os.path.exists(ORIGINAL):
        print("Originalfassung nicht gefunden: %s" % ORIGINAL)
        print("Regressionsteil wird uebersprungen.")
        sys.exit(0)
    alt = lade(ORIGINAL)
    neu = lade(SEITE)
    prices, divs, btc, fx, gew = testdaten()

    bestanden = 0; fehlgeschlagen = 0
    def pruefe(name, bedingung, detail=""):
        global bestanden, fehlgeschlagen
        if bedingung:
            bestanden += 1
            print("   ok    %s" % name)
        else:
            fehlgeschlagen += 1
            print("   FEHLT %s   %s" % (name, detail))

    print("=== 1. Regression: Mindestgebuehr 0 muss das alte Ergebnis liefern ===")
    ts_a, tx_a, ev_a = lauf(alt, prices, divs, btc, fx, gew)
    ts_n, tx_n, ev_n = lauf(neu, prices, divs, btc, fx, gew,
                            min_fee_chf=0.0, fx_fee_bps=0.0, min_order_chf=0.0)

    nav_a = float(ts_a["total_value"].iloc[-1])
    nav_n = float(ts_n["total_value"].iloc[-1])
    k_a = ts_a.attrs["total_tx_costs"]; k_n = ts_n.attrs["total_tx_costs"]
    print("   NAV alt %15.2f   neu %15.2f   Differenz %12.2f" % (nav_a, nav_n, nav_n-nav_a))
    print("   Kosten alt %12.2f   neu %12.2f   Differenz %12.2f" % (k_a, k_n, k_n-k_a))
    # Rebalancing und Kappung wurden bewusst von halbem auf vollen Turnover
    # umgestellt. Der NAV darf sich deshalb unterscheiden, aber nur nach unten
    # und nur in der Groessenordnung der Rebalancingkosten.
    pruefe("NAV weicht um weniger als 1 % ab",
           abs(nav_n-nav_a)/nav_a < 0.01, "%.4f %%" % (abs(nav_n-nav_a)/nav_a*100))
    pruefe("Kosten neu >= Kosten alt (voller statt halber Turnover)", k_n >= k_a - 1e-6)

    print()
    print("=== 2. Abstimmung der Renditezerlegung ===")
    for kennzeichen, ts in (("alt", ts_a), ("neu", ts_n)):
        f = ts.attrs["attribution"]["reconciliation_error"]
        pruefe("Zerlegungsfehler %s ~ 0" % kennzeichen, abs(f) < 1.0, "%.6f" % f)

    print()
    print("=== 3. Mindestgebuehr greift: 20 Einzeltitel gegen 1 ETF ===")
    ts_20, _, _ = lauf(neu, prices, divs, btc, fx, gew,
                       min_fee_chf=75.0, fx_fee_bps=30.0, min_order_chf=500.0)
    s20 = ts_20.attrs["cost_stats"]
    print("   20 Titel: Kosten %10.2f   Orderzeilen %5d   davon Mindestgebuehr %4d"
          % (ts_20.attrs["total_tx_costs"], s20["lines"], s20["lines_at_min"]))

    # Derselbe Marktverlauf, aber als ein einziger Titel abgebildet
    etf_kurs = (prices*pd.Series(gew)).sum(axis=1)/sum(gew.values())
    etf = pd.DataFrame({"ETF.SW": etf_kurs})
    etf_divs = []
    for jahr in sorted({d.year for d in prices.index}):
        kand = [d for d in prices.index if d.year == jahr and d.month == 4]
        if kand:
            d = kand[len(kand)//2]
            etf_divs.append({"date": d, "ticker": "ETF.SW",
                             "dividend_per_share": float(etf_kurs.loc[d])*0.030})
    ts_etf, _, _ = lauf(neu, etf, pd.DataFrame(etf_divs), btc, fx, {"ETF.SW": 100.0},
                        min_fee_chf=75.0, fx_fee_bps=30.0, min_order_chf=500.0)
    se = ts_etf.attrs["cost_stats"]
    print("   1 ETF   : Kosten %10.2f   Orderzeilen %5d   davon Mindestgebuehr %4d"
          % (ts_etf.attrs["total_tx_costs"], se["lines"], se["lines_at_min"]))
    pruefe("ETF erzeugt deutlich weniger Orderzeilen", se["lines"] < s20["lines"]/3,
           "%d gegen %d" % (se["lines"], s20["lines"]))
    pruefe("ETF ist billiger", ts_etf.attrs["total_tx_costs"] < ts_20.attrs["total_tx_costs"])

    print()
    print("=== 4. Wirkung der Mindestgebuehr auf die Einzeltitelvariante ===")
    ohne = lauf(neu, prices, divs, btc, fx, gew, min_fee_chf=0.0)[0]
    print("   ohne Mindestgebuehr %12.2f" % ohne.attrs["total_tx_costs"])
    print("   mit  Mindestgebuehr %12.2f" % ts_20.attrs["total_tx_costs"])
    pruefe("Mindestgebuehr erhoeht die Kosten spuerbar",
           ts_20.attrs["total_tx_costs"] > ohne.attrs["total_tx_costs"]*1.5)

    print()
    print("=== 5. Grenzfaelle ===")
    leer = lauf(neu, prices, divs, btc, fx, gew, min_fee_chf=1e9, min_order_chf=0.0)[0]
    pruefe("absurde Mindestgebuehr fuehrt nicht zu negativem NAV",
           float(leer["total_value"].iloc[-1]) > 0)
    null = lauf(neu, prices, divs, btc, fx, gew, min_fee_chf=0.0, fx_fee_bps=0.0)[0]
    pruefe("Statistik vorhanden und konsistent",
           null.attrs["cost_stats"]["lines"] > 0)
    gross = lauf(neu, prices, divs, btc, fx, gew, min_fee_chf=75.0,
                 min_order_chf=1e9)[0]
    pruefe("Bagatellgrenze unterdrueckt alle Gebuehren",
           gross.attrs["total_tx_costs"] == 0.0,
           "%.2f" % gross.attrs["total_tx_costs"])

    print()
    print("=== 5b. Rekonstruktion der thesaurierenden Anteilsklasse ===")
    syn = neu["synthesize_accumulating"]
    # Exakt nachrechenbarer Fall: Kurs 100, Bruttoausschuettung 10, VSt 35 %.
    # Am Ex-Tag faellt der Kurs um 10 (brutto), wiederangelegt werden 6.50.
    _t = pd.bdate_range("2020-01-01", periods=10)
    _kurs = pd.Series([100.0]*5 + [90.0]*5, index=_t)
    _dv = pd.DataFrame([{"date": _t[5], "ticker": "X", "dividend_per_share": 10.0}])
    _acc = syn(_kurs, _dv, "X")
    print("   vor Ex-Tag %.4f   ab Ex-Tag %.4f   erwartet %.4f"
          % (_acc.iloc[4], _acc.iloc[5], 96.5))
    pruefe("Wert vor dem Ex-Tag unveraendert", abs(_acc.iloc[4] - 100.0) < 1e-9)
    pruefe("Nettoausschuettung exakt wiederangelegt",
           abs(_acc.iloc[5] - 96.5) < 1e-9, "%.6f" % _acc.iloc[5])
    _verlust = (100.0 - float(_acc.iloc[5])) / 10.0
    pruefe("genau 35 Prozent der Ausschuettung gehen verloren",
           abs(_verlust - 0.35) < 1e-9, "%.4f" % _verlust)
    _acc0 = syn(_kurs, _dv, "X", net_factor=1.0)
    pruefe("ohne Steuer bleibt der Wert erhalten",
           abs(float(_acc0.iloc[5]) - 100.0) < 1e-9, "%.6f" % _acc0.iloc[5])
    _leer = syn(_kurs, pd.DataFrame(columns=["date", "ticker", "dividend_per_share"]), "X")
    pruefe("ohne Ausschuettungen identisch zur Ausgangsreihe",
           bool(np.allclose(_leer.values, _kurs.values)))
    _fremd = syn(_kurs, pd.DataFrame([{"date": _t[5], "ticker": "ANDERER",
                                       "dividend_per_share": 10.0}]), "X")
    pruefe("Ausschuettungen fremder Titel werden ignoriert",
           bool(np.allclose(_fremd.values, _kurs.values)))
    # Monotonie: mehr Ausschuettung heisst hoehere thesaurierende Reihe
    _mehr = syn(_kurs, pd.DataFrame([{"date": _t[5], "ticker": "X",
                                      "dividend_per_share": 20.0}]), "X")
    pruefe("groessere Ausschuettung hebt die Reihe staerker",
           float(_mehr.iloc[5]) > float(_acc.iloc[5]))

    print()
    print("=== 6. Entnahmemodus (thesaurierender ETF) ===")
    # Thesaurierend heisst: ein Titel, keine Dividenden.
    thes = pd.DataFrame({"ACC.SW": (prices*pd.Series(gew)).sum(axis=1)/sum(gew.values())})
    leer_divs = pd.DataFrame(columns=["date", "ticker", "dividend_per_share"])
    ts_e, tx_e, _ = lauf(neu, thes, leer_divs, btc, fx, {"ACC.SW": 100.0},
                         min_fee_chf=75.0, fx_fee_bps=30.0, min_order_chf=500.0,
                         harvest_mode="withdrawal", withdrawal_pct_monthly=0.0025)
    a_e = ts_e.attrs["attribution"]
    se = ts_e.attrs["cost_stats"]
    monate = len({(d.year, d.month) for d in ts_e.index})
    print("   Endwert %12.2f   Kosten %9.2f   Orderzeilen %4d   Monate %3d"
          % (float(ts_e["total_value"].iloc[-1]), ts_e.attrs["total_tx_costs"],
             se["lines"], monate))
    print("   in Bitcoin investiert (Entnahmen) %12.2f" % a_e["btc_dca_invested"])
    pruefe("Zerlegung stimmt ab", abs(a_e["reconciliation_error"]) < 1.0,
           "%.4f" % a_e["reconciliation_error"])
    pruefe("Entnahmen haben stattgefunden", a_e["btc_dca_invested"] > 0)
    pruefe("keine Dividendenertraege", abs(a_e["dividend_income"]) < 1e-6)
    pruefe("Dividendenkasse bleibt leer",
           float(ts_e["dividend_cash"].abs().max()) < 1e-6)
    ent = tx_e[tx_e["reason"] == "ENTNAHME"] if not tx_e.empty else tx_e
    pruefe("eine Entnahme je Monat, nicht mehr", len(ent) <= monate,
           "%d Entnahmen, %d Monate" % (len(ent), monate))
    # Zwei Orderzeilen je Entnahme (Verkauf ETF, Kauf Bitcoin), dazu die
    # Startallokation und die Schwellenereignisse. Sinnvoll pruefbar ist der
    # Abstand zur Einzeltitelvariante, nicht eine absolute Schranke.
    print("   zum Vergleich 20 Einzeltitel mit Dividendenernte: %d Zeilen"
          % s20["lines"])
    pruefe("Entnahmevariante braucht ein Vielfaches weniger Orderzeilen",
           se["lines"] < s20["lines"]/1.4,
           "%d gegen %d" % (se["lines"], s20["lines"]))
    pruefe("Zeilenzahl liegt in der Groessenordnung zweier je Monat",
           1.5*monate <= se["lines"] <= 2.6*monate,
           "%d Zeilen bei %d Monaten" % (se["lines"], monate))

    print()
    print("=== 7. Entnahmesatz wirkt wie erwartet ===")
    werte = {}
    for satz in (0.0, 0.00161, 0.0025, 0.005):
        t_, _, _ = lauf(neu, thes, leer_divs, btc, fx, {"ACC.SW": 100.0},
                        min_fee_chf=75.0, fx_fee_bps=30.0, min_order_chf=500.0,
                        harvest_mode="withdrawal", withdrawal_pct_monthly=satz)
        a_ = t_.attrs["attribution"]
        werte[satz] = (float(t_["smi_value"].iloc[-1]), a_["btc_dca_invested"],
                       float(t_["total_value"].iloc[-1]))
        print("   Satz %6.3f %%   Aktienteil %12.2f   in Bitcoin %11.2f   NAV %12.2f"
              % (satz*100, *werte[satz]))
    pruefe("hoeherer Satz verschiebt mehr in Bitcoin",
           werte[0.005][1] > werte[0.0025][1] > werte[0.00161][1] > werte[0.0][1])
    pruefe("Satz 0 entspricht keiner Entnahme", werte[0.0][1] == 0.0)

    # Der Aktienteil faellt NICHT monoton mit dem Entnahmesatz, und das ist
    # richtig so: je hoeher die Entnahme, desto oefter reisst Bitcoin die
    # 25-Prozent-Schwelle, und desto mehr Kapital speist die Schwelle in die
    # Aktien zurueck. Das Band ist ein geschlossener Kreislauf. Monoton ist
    # der Zusammenhang erst, wenn man das Band abschaltet.
    ohne_band = {}
    mit_band = {}
    for satz in (0.0, 0.00161, 0.0025, 0.005):
        t_o, _, _ = lauf(neu, thes, leer_divs, btc, fx, {"ACC.SW": 100.0},
                         min_fee_chf=75.0, fx_fee_bps=30.0, min_order_chf=500.0,
                         harvest_mode="withdrawal", withdrawal_pct_monthly=satz,
                         obere_schwelle=0.99)
        ohne_band[satz] = float(t_o["smi_value"].iloc[-1])
        t_m, _, e_m = lauf(neu, thes, leer_divs, btc, fx, {"ACC.SW": 100.0},
                           min_fee_chf=75.0, fx_fee_bps=30.0, min_order_chf=500.0,
                           harvest_mode="withdrawal", withdrawal_pct_monthly=satz)
        mit_band[satz] = 0 if e_m is None or len(e_m) == 0 else len(e_m)
    print("   ohne Band, Aktienteil je Satz: " + "  ".join(
        "%.3f %%: %.0f" % (k*100, v) for k, v in ohne_band.items()))
    print("   mit Band, Schwellenereignisse je Satz: " + "  ".join(
        "%.3f %%: %d" % (k*100, v) for k, v in mit_band.items()))
    pruefe("ohne Band faellt der Aktienteil streng mit dem Satz",
           ohne_band[0.005] < ohne_band[0.0025] < ohne_band[0.00161] < ohne_band[0.0])
    pruefe("mit Band loest ein hoeherer Satz mehr Schwellenereignisse aus",
           mit_band[0.005] >= mit_band[0.0025] >= mit_band[0.0])

    print()
    print("=== 7b. Entnahmefrequenz ===")
    freq = {}
    for nm in (1, 3, 6, 12):
        t_, _, _ = lauf(neu, thes, leer_divs, btc, fx, {"ACC.SW": 100.0},
                        min_fee_chf=75.0, fx_fee_bps=30.0, min_order_chf=500.0,
                        harvest_mode="withdrawal", withdrawal_pct_monthly=0.0025,
                        withdrawal_every_n_months=nm)
        a_ = t_.attrs["attribution"]; s_ = t_.attrs["cost_stats"]
        freq[nm] = (s_["lines"], t_.attrs["total_tx_costs"], a_["btc_dca_invested"],
                    a_["reconciliation_error"])
        print("   alle %2d Monate   Zeilen %4d   Kosten %9.0f   in Bitcoin %10.0f"
              % (nm, freq[nm][0], freq[nm][1], freq[nm][2]))
    pruefe("seltenere Termine erzeugen weniger Orderzeilen",
           freq[12][0] < freq[6][0] < freq[3][0] < freq[1][0])
    pruefe("seltenere Termine kosten weniger",
           freq[12][1] < freq[6][1] < freq[3][1] < freq[1][1])
    # Der Jahresbetrag soll sich nicht wesentlich aendern: die Entnahme je
    # Termin waechst mit dem Abstand. Abweichungen kommen nur daher, dass
    # zwischen den Terminen Kurse laufen.
    _ref = freq[1][2]
    pruefe("Jahresbetrag bleibt in derselben Groessenordnung",
           all(abs(v[2] - _ref)/_ref < 0.20 for v in freq.values()),
           "  ".join("%d: %.0f" % (k, v[2]) for k, v in freq.items()))
    for nm, v in freq.items():
        pruefe("Zerlegung stimmt bei Abstand %d ab" % nm, abs(v[3]) < 1.0,
               "%.4f" % v[3])

    print()
    print("=== 8. Wasserfall: Kasse vor Abverkauf ===")
    # Ausschuettender Sleeve im Entnahmemodus: die Dividende muss verwendet
    # werden, bevor Anteile verkauft werden, und darf nicht liegenbleiben.
    ts_m, _, _ = lauf(neu, prices, divs, btc, fx, gew,
                      min_fee_chf=75.0, fx_fee_bps=30.0, min_order_chf=500.0,
                      harvest_mode="withdrawal", withdrawal_pct_monthly=0.0025)
    a_m = ts_m.attrs["attribution"]
    print("   Dividendenertrag %11.2f   in Bitcoin %12.2f   Restkasse %9.2f"
          % (a_m["dividend_income"], a_m["btc_dca_invested"],
             float(ts_m["dividend_cash"].iloc[-1])))
    pruefe("Zerlegung stimmt auch im Mischfall ab",
           abs(a_m["reconciliation_error"]) < 1.0,
           "%.4f" % a_m["reconciliation_error"])
    pruefe("Dividenden bleiben nicht unbegrenzt liegen",
           float(ts_m["dividend_cash"].iloc[-1]) < a_m["dividend_income"],
           "Restkasse %.2f gegen Ertrag %.2f" % (
               float(ts_m["dividend_cash"].iloc[-1]), a_m["dividend_income"]))
    pruefe("keine DCA-Tranchen im Entnahmemodus",
           not (not tx_e.empty and (tx_e["reason"] == "DCA").any()))

    print()
    print("=== 9. Kapitalfluss: Zeichnungen und Ruecknahmen ===")
    fluss = {}
    for pct, txt in ((0.0, "kein Fluss"), (0.005, "+0.5 %"), (0.01, "+1.0 %"),
                     (0.02, "+2.0 %"), (-0.005, "-0.5 %")):
        t_, tx_, _ = lauf(neu, prices, divs, btc, fx, gew,
                          min_fee_chf=75.0, fx_fee_bps=30.0, min_order_chf=500.0,
                          monthly_flow_pct=pct)
        a_ = t_.attrs["attribution"]
        fluss[pct] = (float(t_["total_value"].iloc[-1]), t_.attrs["net_flow"],
                      float(t_["nav_per_unit"].iloc[-1]), a_["reconciliation_error"])
        print("   %-12s Basket %12.0f   Nettofluss %11.0f   Anteilswert %11.0f"
              % (txt, fluss[pct][0], fluss[pct][1], fluss[pct][2]))
    for pct, v in fluss.items():
        pruefe("Zerlegung stimmt bei Fluss %+.1f %% ab" % (pct*100),
               abs(v[3]) < 1.0, "%.4f" % v[3])
    pruefe("Zufluss vergroessert den Basket",
           fluss[0.02][0] > fluss[0.01][0] > fluss[0.005][0] > fluss[0.0][0])
    pruefe("Abfluss verkleinert den Basket", fluss[-0.005][0] < fluss[0.0][0])
    pruefe("ohne Fluss ist der Nettofluss null", abs(fluss[0.0][1]) < 1e-9)
    pruefe("Nettofluss bei Ruecknahmen negativ", fluss[-0.005][1] < 0)
    # OHNE Gebuehren muss der Anteilswert vom Zuflussszenario unabhaengig
    # sein. Das prueft die Anteilsrechnung selbst.
    ohne_geb = {}
    for pct in (0.0, 0.005, 0.01, 0.02, -0.005):
        t_, _, _ = lauf(neu, prices, divs, btc, fx, gew, tx_cost_bps_override=0.0,
                        min_fee_chf=0.0, fx_fee_bps=0.0, min_order_chf=0.0,
                        monthly_flow_pct=pct)
        ohne_geb[pct] = float(t_["nav_per_unit"].iloc[-1])
    _ref = ohne_geb[0.0]
    _max = max(abs(v-_ref)/_ref for v in ohne_geb.values())
    print("   ohne Gebuehren, groesste Abweichung des Anteilswerts: %.4f %%"
          % (_max*100))
    pruefe("ohne Gebuehren ist der Anteilswert vom Zufluss unabhaengig",
           _max < 0.005, "%.4f %%" % (_max*100))

    # MIT Gebuehren sinkt er, und das ist die eigentliche Aussage: jede
    # Zeichnung kostet bei zwanzig Titeln 21 Orderzeilen, und diese Kosten
    # tragen die bestehenden Anleger mit. Je mehr gezeichnet wird, desto mehr
    # Verwaesserung.
    _ref2 = fluss[0.0][2]
    print("   mit Gebuehren, Anteilswert gegenueber dem Lauf ohne Zufluss:")
    for pct in (0.005, 0.01, 0.02):
        print("      Zufluss %+5.1f %% je Monat: %+7.2f %%"
              % (pct*100, (fluss[pct][2]/_ref2 - 1)*100))
    pruefe("Zeichnungen verwaessern den Anteilswert bei Einzeltiteln",
           fluss[0.02][2] < fluss[0.01][2] < fluss[0.005][2] < _ref2)

    print()
    print("=== 9b. Ruecknahmen zehren die Dividendenkasse nicht ins Minus ===")
    # Gefunden von den Zufallslaeufen: eine Ruecknahme nimmt Geld aus der
    # Kasse, auf das die offenen DCA-Tranchen ausgestellt sind. Ohne
    # Nachfuehrung der Tranchen wird die Kasse negativ.
    for _nz in (False, True):
        t_, _, _ = lauf(neu, prices, divs, btc, fx, gew,
                        min_fee_chf=25.0, fx_fee_bps=30.0, min_order_chf=0.0,
                        monthly_flow_pct=-0.005, netting=_nz)
        _kmin = float(t_["dividend_cash"].min())
        _nmin = float(t_["total_value"].min())
        print("   Netting %-4s Kasse mindestens %10.2f   NAV mindestens %12.2f"
              % ("ein" if _nz else "aus", _kmin, _nmin))
        pruefe("Kasse bleibt bei Ruecknahmen nicht negativ (Netting %s)"
               % ("ein" if _nz else "aus"), _kmin >= -1e-6, "%.4f" % _kmin)
        pruefe("NAV bleibt bei Ruecknahmen positiv (Netting %s)"
               % ("ein" if _nz else "aus"), _nmin > 0, "%.4f" % _nmin)
    # Harte Ruecknahme ueber mehrere Jahre: das Produkt muss schrumpfen,
    # nicht kippen.
    t_, _, _ = lauf(neu, prices, divs, btc, fx, gew,
                    min_fee_chf=75.0, fx_fee_bps=30.0, min_order_chf=500.0,
                    monthly_flow_pct=-0.03)
    print("   Ruecknahme 3 %% je Monat: Endbasket %.0f, NAV mindestens %.2f"
          % (float(t_["total_value"].iloc[-1]), float(t_["total_value"].min())))
    pruefe("auch bei 3 %% Ruecknahme je Monat bleibt der NAV positiv",
           float(t_["total_value"].min()) > 0)
    pruefe("Zerlegung stimmt bei harter Ruecknahme ab",
           abs(t_.attrs["attribution"]["reconciliation_error"]) < 1.0,
           "%.4f" % t_.attrs["attribution"]["reconciliation_error"])

    print()
    print("=== 10. Netting der Orderzeilen ===")
    netz = {}
    for nz in (False, True):
        t_, _, _ = lauf(neu, thes, leer_divs, btc, fx, {"ACC.SW": 100.0},
                        min_fee_chf=75.0, fx_fee_bps=30.0, min_order_chf=500.0,
                        harvest_mode="withdrawal", withdrawal_pct_monthly=0.0025,
                        monthly_flow_pct=0.01, netting=nz)
        a_ = t_.attrs["attribution"]; s_ = t_.attrs["cost_stats"]
        _m = len({(d.year, d.month) for d in t_.index})
        netz[nz] = (s_["lines"], t_.attrs["total_tx_costs"], a_["reconciliation_error"], _m)
        print("   Netting %-4s Zeilen %4d (%.1f je Monat)   Kosten %9.0f"
              % ("ein" if nz else "aus", netz[nz][0], netz[nz][0]/_m, netz[nz][1]))
    pruefe("Netting senkt die Zahl der Orderzeilen", netz[True][0] < netz[False][0])
    pruefe("Netting senkt die Kosten", netz[True][1] < netz[False][1])
    pruefe("Zerlegung stimmt mit Netting ab", abs(netz[True][2]) < 1.0,
           "%.4f" % netz[True][2])
    pruefe("Zerlegung stimmt ohne Netting ab", abs(netz[False][2]) < 1.0,
           "%.4f" % netz[False][2])
    # Ein ETF plus Bitcoin, Zeichnung und Entnahme am selben Termin: nach
    # Zusammenfassung bleiben genau zwei Orderzeilen je Monat.
    _je_monat = netz[True][0] / netz[True][3]
    pruefe("ETF mit Zeichnung und Entnahme: rund zwei Zeilen je Monat",
           1.8 <= _je_monat <= 2.4, "%.2f je Monat" % _je_monat)
    # Bei zwanzig Einzeltiteln muss die Zeichnung deutlich mehr Zeilen kosten
    t20, _, _ = lauf(neu, prices, divs, btc, fx, gew, min_fee_chf=75.0,
                     fx_fee_bps=30.0, min_order_chf=500.0,
                     monthly_flow_pct=0.01, netting=True)
    _m20 = len({(d.year, d.month) for d in t20.index})
    _je20 = t20.attrs["cost_stats"]["lines"] / _m20
    print("   zum Vergleich 20 Einzeltitel mit Zeichnung: %.1f Zeilen je Monat" % _je20)
    pruefe("20 Einzeltitel brauchen ein Vielfaches mehr Zeilen je Monat",
           _je20 > 6 * _je_monat, "%.1f gegen %.1f" % (_je20, _je_monat))

    print()
    print("=== 11. Kostenstruktur-Ueberlagerung ===")
    # Die Kontrollrechnung soll die Kosten der Einzeltitelvariante
    # reproduzieren, ohne deren Verzerrung durch die heutige
    # Indexzusammensetzung zu uebernehmen. Gerechnet wird dafuer auf EINEM
    # Titel, aber mit der Gebuehrenstruktur von zwanzig.
    ein_titel = pd.DataFrame({"ETF.SW": (prices*pd.Series(gew)).sum(axis=1)/sum(gew.values())})
    ein_divs = []
    for _j in sorted({d.year for d in prices.index}):
        _k = [d for d in prices.index if d.year == _j and d.month == 4]
        if _k:
            _d = _k[len(_k)//2]
            ein_divs.append({"date": _d, "ticker": "ETF.SW",
                             "dividend_per_share": float(ein_titel.loc[_d, "ETF.SW"])*0.03})
    ein_divs = pd.DataFrame(ein_divs)
    smi_gew = [min(v, 18.0) for v in gew.values()]

    def _k_lauf(px, dv, w, cap, **kw):
        return lauf(neu, px, dv, btc, fx, w, min_fee_chf=75.0, fx_fee_bps=30.0,
                    min_order_chf=500.0, monthly_flow_pct=0.01, **kw)

    k_einfach = _k_lauf(ein_titel, ein_divs, {"ETF.SW": 100.0}, None)[0]
    k_gleich = _k_lauf(ein_titel, ein_divs, {"ETF.SW": 100.0}, None, cost_titles=20)[0]
    k_gewicht = _k_lauf(ein_titel, ein_divs, {"ETF.SW": 100.0}, None,
                        cost_titles=smi_gew)[0]
    k_echt = _k_lauf(prices, divs, gew, 0.18)[0]
    for nm, t_ in (("ein Titel", k_einfach), ("gleichmaessig 20", k_gleich),
                   ("nach Zielgewichten", k_gewicht), ("20 echte Titel", k_echt)):
        print("   %-22s Zeilen %5d   Kosten %9.0f   Anteilswert %11.0f" % (
            nm, t_.attrs["cost_stats"]["lines"], t_.attrs["total_tx_costs"],
            float(t_["nav_per_unit"].iloc[-1])))
    pruefe("Ueberlagerung erhoeht die Zahl der Orderzeilen deutlich",
           k_gewicht.attrs["cost_stats"]["lines"] > 5*k_einfach.attrs["cost_stats"]["lines"])
    # DIE zentrale Eigenschaft: die Kontrolle muss die echte
    # Einzeltitelvariante bei den Kosten treffen, sonst misst sie nichts.
    _dz = k_gewicht.attrs["cost_stats"]["lines"]/k_echt.attrs["cost_stats"]["lines"] - 1
    _dk = k_gewicht.attrs["total_tx_costs"]/k_echt.attrs["total_tx_costs"] - 1
    print("   Kontrolle gegen echte Einzeltitel: Zeilen %+.0f %%, Kosten %+.0f %%"
          % (_dz*100, _dk*100))
    pruefe("Kontrolle trifft die Zeilenzahl der Einzeltitel auf 10 %",
           abs(_dz) < 0.10, "%+.1f %%" % (_dz*100))
    pruefe("Kontrolle trifft die Kosten der Einzeltitel auf 10 %",
           abs(_dk) < 0.10, "%+.1f %%" % (_dk*100))
    # Gleichmaessige Aufteilung ueberzeichnet, weil kleine Zielgewichte real
    # unter die Bagatellgrenze fallen.
    pruefe("gleichmaessige Aufteilung ueberzeichnet gegenueber Zielgewichten",
           k_gleich.attrs["total_tx_costs"] > k_gewicht.attrs["total_tx_costs"])
    pruefe("Ueberlagerung veraendert nur die Kosten, nicht die Zerlegung",
           abs(k_gewicht.attrs["attribution"]["reconciliation_error"]) < 1.0)
    pruefe("mehr Kosten heisst tieferer Anteilswert",
           float(k_gewicht["nav_per_unit"].iloc[-1]) < float(k_einfach["nav_per_unit"].iloc[-1]))
    pruefe("ohne Ueberlagerung bleibt alles wie bisher",
           k_einfach.attrs["cost_stats"]["lines"] ==
           _k_lauf(ein_titel, ein_divs, {"ETF.SW": 100.0}, None,
                   cost_titles=None)[0].attrs["cost_stats"]["lines"])

    print()
    print("-"*60)
    print("Bestanden: %d   Fehlgeschlagen: %d" % (bestanden, fehlgeschlagen))
    sys.exit(1 if fehlgeschlagen else 0)
