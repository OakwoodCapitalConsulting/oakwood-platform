"""
Rauchtest: laesst die ganze Streamlit-Seite einmal durchlaufen.

Streamlit und yfinance werden nachgebaut. Die Widgets liefern ihre
Vorgabewerte zurueck, yfinance liefert synthetische Kurse. Damit lassen sich
Namensfehler, falsche Reihenfolgen und Abstuerze in den neuen Codepfaden
finden, ohne Netzzugang und ohne Streamlit-Server.

Aufruf:  python3 rauchtest.py            Vorgabeeinstellungen
         python3 rauchtest.py etf        ausschuettender ETF-Sleeve
         python3 rauchtest.py thes       thesaurierender ETF mit Entnahme
         python3 rauchtest.py synth      thesaurierender ETF, Historie rekonstruiert
         python3 rauchtest.py thesdiv    Fehlkombination, muss gemeldet werden
         python3 rauchtest.py vergleich  zusaetzlich den Strukturvergleich
         python3 rauchtest.py fluss      mit monatlicher Zeichnung
         python3 rauchtest.py flussetf   Zeichnung plus thesaurierender ETF
         python3 rauchtest.py flussaus   Zeichnung ohne Netting
"""
import sys, types, traceback
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


MODUS = sys.argv[1] if len(sys.argv) > 1 else "standard"

# ------------------------------------------------------------ Streamlit-Ersatz
class Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False

class Spalte:
    def __init__(self, st): self._st = st
    def __getattr__(self, name): return getattr(self._st, name)
    def __enter__(self): return self
    def __exit__(self, *a): return False

class Fortschritt:
    def progress(self, *a, **k): pass
    def empty(self): pass

class Zustand(dict):
    def __getattr__(self, k): return self.get(k)
    def __setattr__(self, k, v): self[k] = v

class StStub(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = Zustand()
        self.sidebar = Ctx()
        self.aufrufe = {"warning": [], "error": [], "success": [], "info": []}
        self.knoepfe = set()

    # Ausgaben
    def _sammle(self, art, *a, **k):
        if a: self.aufrufe.setdefault(art, []).append(str(a[0])[:300])
    def markdown(self, *a, **k): pass
    def caption(self, *a, **k): pass
    def write(self, *a, **k): pass
    def text(self, *a, **k): pass
    def code(self, *a, **k): pass
    def latex(self, *a, **k): pass
    def divider(self, *a, **k): pass
    def title(self, *a, **k): pass
    def header(self, *a, **k): pass
    def subheader(self, *a, **k): pass
    tabellen = []
    def dataframe(self, *a, **k):
        if a: StStub.tabellen.append(a[0])
    def table(self, *a, **k): pass
    def plotly_chart(self, *a, **k): pass
    def pyplot(self, *a, **k): pass
    def image(self, *a, **k): pass
    def metric(self, *a, **k): pass
    def json(self, *a, **k): pass
    def download_button(self, *a, **k): return False
    def warning(self, *a, **k): self._sammle("warning", *a)
    def error(self, *a, **k): self._sammle("error", *a)
    def success(self, *a, **k): self._sammle("success", *a)
    def info(self, *a, **k): self._sammle("info", *a)
    def toast(self, *a, **k): pass
    def balloons(self, *a, **k): pass
    def rerun(self, *a, **k): pass
    def stop(self, *a, **k): raise SystemExit("st.stop()")

    # Layout
    def set_page_config(self, *a, **k): pass
    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [Spalte(self) for _ in range(n)]
    def container(self, *a, **k): return Ctx()
    def expander(self, *a, **k): return Ctx()
    def tabs(self, namen, **k): return [Ctx() for _ in namen]
    def spinner(self, *a, **k): return Ctx()
    def form(self, *a, **k): return Ctx()
    def empty(self, *a, **k): return Spalte(self)
    def progress(self, *a, **k): return Fortschritt()
    def status(self, *a, **k): return Ctx()

    # Eingaben: liefern den Vorgabewert
    def button(self, label, **k): return label in self.knoepfe
    def form_submit_button(self, *a, **k): return False
    checkbox_vorgabe = {}
    def checkbox(self, label, value=False, *a, **k):
        if label in self.checkbox_vorgabe: return self.checkbox_vorgabe[label]
        return value
    def toggle(self, label, value=False, **k): return value
    def radio(self, label, options, index=0, *a, **k):
        opts = list(options)
        if label in self.radio_vorgabe:
            return self.radio_vorgabe[label]
        return opts[index or 0]
    radio_vorgabe = {}
    selectbox_vorgabe = {}
    def selectbox(self, label, options, index=0, *a, **k):
        opts = list(options)
        if label in self.selectbox_vorgabe:
            return self.selectbox_vorgabe[label]
        return opts[index or 0]
    multiselect_vorgabe = {}
    def multiselect(self, label, options, default=None, **k):
        if label in self.multiselect_vorgabe:
            return list(self.multiselect_vorgabe[label])
        return list(default) if default else []
    def _wert(self, a, k, pos=2):
        # Streamlit erlaubt (label, min, max, value, step) positional
        if "value" in k and k["value"] is not None: return k["value"]
        if len(a) > pos and a[pos] is not None: return a[pos]
        if "min_value" in k: return k["min_value"]
        return a[0] if a else 0
    slider_vorgabe = {}
    def slider(self, label, *a, **k):
        if label in self.slider_vorgabe: return self.slider_vorgabe[label]
        return self._wert(a, k)
    def number_input(self, label, *a, **k): return self._wert(a, k)
    def text_input(self, label, value="", **k): return value
    def text_area(self, label, value="", **k): return value
    def date_input(self, label, value=None, *a, **k): return value
    def file_uploader(self, *a, **k): return None
    def color_picker(self, label, value="#000000", *a, **k): return value
    def select_slider(self, label, options=(), value=None, *a, **k):
        opts = list(options)
        if value is not None: return value
        return opts[0] if opts else None
    def data_editor(self, data=None, *a, **k): return data
    def __getattr__(self, name):
        # Jedes unbekannte Streamlit-Element wird zu einer harmlosen Ausgabe.
        def platzhalter(*a, **k): return None
        return platzhalter

    # Cache: muss sowohl als Dekorator als auch als Dekoratorfabrik
    # funktionieren UND ein .clear() tragen.


class CacheStub:
    def __call__(self, *a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f
    def clear(self): pass

st_stub = StStub()
st_stub.cache_data = CacheStub()
st_stub.cache_resource = CacheStub()
sys.modules["streamlit"] = st_stub

# ------------------------------------------------------------ yfinance-Ersatz
TAGE = pd.bdate_range("2015-01-01", "2026-09-24")
_rng = np.random.default_rng(99)
_markt = _rng.normal(0.00028, 0.0088, len(TAGE))

def _reihe(saat, basis=100.0, eigen=0.008):
    r = np.random.default_rng(saat)
    return pd.Series(basis*np.exp(np.cumsum(_markt + r.normal(0, eigen, len(TAGE)))),
                     index=TAGE)

def _saat(name):
    # Pythons String-Hash ist je Prozess zufaellig. Fuer reproduzierbare
    # Testlaeufe braucht es eine stabile Ableitung.
    return sum((i+1)*ord(c) for i, c in enumerate(name)) % 10**6

def _rahmen(ticker):
    s = _reihe(_saat(ticker))
    df = pd.DataFrame({"Open": s, "High": s*1.01, "Low": s*0.99,
                       "Close": s, "Adj Close": s, "Volume": 1000.0}, index=TAGE)
    return df

class TickerStub:
    def __init__(self, symbol): self.symbol = symbol
    # Thesaurierende ETFs schuetten nichts aus. Der Rauchtest muss das
    # abbilden, sonst prueft er den Entnahmepfad nie unter echten Bedingungen.
    THESAURIEREND = {"SW2CHB.SW", "SMIA.SW"}

    @property
    def actions(self):
        if self.symbol in self.THESAURIEREND:
            return pd.DataFrame({"Dividends": pd.Series(dtype=float),
                                 "Stock Splits": pd.Series(dtype=float)})
        kurs = _reihe(_saat(self.symbol))
        zeilen = {}
        for jahr in range(2015, 2027):
            kand = [d for d in TAGE if d.year == jahr and d.month == 4]
            if kand:
                d = kand[10]
                zeilen[d] = float(kurs.loc[d])*0.030
        return pd.DataFrame({"Dividends": pd.Series(zeilen),
                             "Stock Splits": pd.Series(0.0, index=list(zeilen))})
    def get_dividends(self): return self.actions["Dividends"]
    @property
    def splits(self): return pd.Series(dtype=float)

def download(tickers, start=None, end=None, **k):
    liste = tickers if isinstance(tickers, (list, tuple)) else [tickers]
    if len(liste) == 1:
        df = _rahmen(liste[0])
    else:
        teile = {}
        for t in liste:
            r = _rahmen(t)
            for sp in r.columns:
                teile[(t, sp)] = r[sp]
        df = pd.DataFrame(teile)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
    if start: df = df[df.index >= pd.Timestamp(start)]
    if end:   df = df[df.index <= pd.Timestamp(end)]
    return df

yf_stub = types.ModuleType("yfinance")
yf_stub.download = download
yf_stub.Ticker = TickerStub
sys.modules["yfinance"] = yf_stub

# ------------------------------------------------------------ Lauf
_SLEEVES = {
    "etf":       "Ausschüttend · iShares SMI ETF (CSSMI)",
    "vergleich": "Ausschüttend · iShares SMI ETF (CSSMI)",
    "thes":      "Thesaurierend · UBS MSCI Switzerland 20/35 (SW2CHB)",
    "synth":     "Thesaurierend · UBS SMI ETF (SMIA), Historie rekonstruiert",
    "thesdiv":   "Thesaurierend · UBS MSCI Switzerland 20/35 (SW2CHB)",
}
# Kapitalfluss: der Regler steht in der Vorgabe auf null, der Pfad wuerde
# sonst nie durchlaufen.
if MODUS in ("fluss", "flussetf", "flussaus"):
    StStub.slider_vorgabe = {"Netto-Kapitalfluss je Monat (% des NAV)": 1.0}
if MODUS == "flussetf":
    StStub.radio_vorgabe = {
        "Aufbau des Aktienteils":
            "Thesaurierend · UBS SMI ETF (SMIA), Historie rekonstruiert"}
if MODUS == "flussaus":
    # Netting ausgeschaltet: jede Teilbewegung zaehlt einzeln
    StStub.checkbox_vorgabe = {"Orderzeilen je Ausführungstag netten": False}
    StStub.slider_vorgabe = {"Netto-Kapitalfluss je Monat (% des NAV)": 1.0}

if MODUS in _SLEEVES:
    StStub.radio_vorgabe = {"Aufbau des Aktienteils": _SLEEVES[MODUS]}
# thesdiv prueft die Fehlkombination: thesaurierender ETF mit Dividendenernte
if MODUS == "thesdiv":
    StStub.radio_vorgabe["Woher kommt das Geld für Bitcoin?"] = \
        "Dividendenernte über DCA-Fenster (heute)"
if MODUS == "vergleich":
    st_stub.knoepfe.add("Strukturvergleich rechnen")
# raster rechnet die Kalibrierung der Entnahmemechanik auf einem kleinen
# Feld, damit der Test schnell bleibt, aber jeder Pfad durchlaufen wird.
if MODUS in ("raster", "rasterseit", "rastersmi"):
    st_stub.knoepfe.add("Raster starten")
    StStub.multiselect_vorgabe = {"Obere Schwelle (%)": [25.0, 35.0],
                                  "Entnahmesatz je Monat (%)": [0.15, 0.25, 0.50]}
if MODUS == "rasterseit":
    StStub.selectbox_vorgabe = {"Bitcoin-Pfad": "Seitwärts: Bitcoin ohne Trend"}
if MODUS == "rastersmi":
    StStub.selectbox_vorgabe = {
        "Bitcoin-Pfad": "Ohne Überrendite: Bitcoin wächst wie der SMI-ETF"}
# tagestest rechnet den Indifferenz-Test des Ausfuehrungstags unter der
# Entnahme (Patch J), historisch und mit Bitcoin ohne Trend.
if MODUS in ("tagestest", "tagestestseit"):
    st_stub.knoepfe.add("Tagestest starten")
if MODUS == "tagestestseit":
    StStub.selectbox_vorgabe = {
        "Bitcoin-Pfad (Tagestest)": "Seitwärts: Bitcoin ohne Trend"}
st_stub.session_state["smi_has_run"] = True

sys.path.insert(0, REPO)
print("Modus: %s" % MODUS)
try:
    quelle = open(SEITE, encoding="utf-8").read()
    raum = {"__name__": "__main__", "__file__": SEITE}
    exec(compile(quelle, "1_SMI_Strategy.py", "exec"), raum)
    print("Seite vollstaendig durchgelaufen.")
except SystemExit as e:
    print("Seite hat abgebrochen: %s" % e)
except Exception:
    print("ABSTURZ:")
    traceback.print_exc()
    sys.exit(1)

for art in ("error", "warning"):
    for m in st_stub.aufrufe.get(art, []):
        print("  [%s] %s" % (art, m.replace("\n", " ")[:220]))
for art in ("success", "info"):
    for m in st_stub.aufrufe.get(art, []):
        print("  [%s] %s" % (art, m.replace("\n", " ")[:260]))

if MODUS in ("tagestest", "tagestestseit"):
    import pandas as _pd
    _pd.set_option("display.width", 250)
    _tt = [t for t in StStub.tabellen
           if hasattr(t, "columns") and "Konvention" in t.columns]
    print("Tagestest-Tabellen gefunden:", len(_tt))
    for t in _tt:
        print(t.to_string(index=False)); print()
    for m in st_stub.aufrufe.get("caption", []):
        if "rollierende" in m and "Konventionen" in m:
            print("  [caption]", m)

if MODUS in ("raster", "rasterseit", "rastersmi"):
    import pandas as _pd
    _pd.set_option("display.width", 250); _pd.set_option("display.max_columns", 30)
    _gesucht = [t for t in StStub.tabellen
                if hasattr(t, "columns") and ("Obere Schwelle" in t.columns
                                              or "Entnahme je Monat" in t.columns)]
    print("Rastertabellen gefunden:", len(_gesucht))
    for t in _gesucht:
        print(t.to_string(index=False)); print()
