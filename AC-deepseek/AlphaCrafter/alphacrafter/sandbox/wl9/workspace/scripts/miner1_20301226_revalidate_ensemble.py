"""miner_1 re-validation of current effective library (2030-12-26 cycle).

Validates every non-deprecated factor in factors/ against the *fresh* data
window 2026-07-16..2030-12-25 (online start onward, +53 months of out-of-sample
data since the library was originally admitted on warm-up data 2020..2026-07-15).

Metrics: rank IC / ICIR at h=10 on the tradable 15-instrument cross-asset universe.
Gates (shared, 15-name universe): abs(IC) >= 0.0070 AND abs(ICIR) >= 0.0840.
Obs-only macros (DXY USDCNY USDJPY EURUSD VIX) are used only for computation, never traded.

Important safety constraint: the trader agent is the ONLY agent allowed to
advance the live account. This script is pure offline research: it reads
../persistent/*.csv but writes to factors/*.json only.
"""
from __future__ import annotations
import base64, io, json, zlib, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA_DIR = Path("../persistent")
STOCK_DIR = DATA_DIR / "stock_data"
IDX_DIR = DATA_DIR / "index_data"
FACTOR_DIR = Path("factors")

ASSETS = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

VISIBLE_END = "2030-12-25"   # previous completed trading day (2030-12-26 is current date)
VALID_START = "2026-07-16"   # online-start onward (dedicated re-validation window)
HORIZON = 10
GATE_IC = 0.0070
GATE_ICIR = 0.0840


def load_ohlc(end=VISIBLE_END):
    out = {}
    for a in ASSETS:
        f = STOCK_DIR / f"{a}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= end].set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
        out[a] = df
    return out


def load_macro(end=VISIBLE_END):
    out = {}
    for m in MACRO:
        f = IDX_DIR / f"{m}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= end]
        out[m] = df.set_index("date")["close"].astype(float)
    return out


def build_df(end=VISIBLE_END, start=VALID_START):
    ohlc = load_ohlc(end)
    macro = load_macro(end)
    all_dates = set()
    for d in ohlc.values():
        all_dates.update(d.index)
    for d in macro.values():
        all_dates.update(d.index)
    dates = sorted(x for x in all_dates if start <= x.strftime("%Y-%m-%d") <= end and x.weekday() < 5)
    dates = pd.DatetimeIndex(dates)
    df = pd.DataFrame(index=dates)
    for a, d in ohlc.items():
        for col in d.columns:
            df.loc[d.index, f"{a}__{col}"] = d[col].values
    for m, d in macro.items():
        df.loc[d.index, f"{m}__close"] = d.values
    return df


def panel_from_series(series_map):
    idx = sorted({d for s in series_map.values() for d in s.index})
    panel = pd.DataFrame(index=idx, dtype=float)
    for a, s in series_map.items():
        panel[a] = s
    return panel


def fwd_returns(close_panel, horizon=HORIZON):
    return close_panel.shift(-horizon) / close_panel - 1.0


def evaluate(panel, fwd, min_valid=8):
    ics, dates = [], []
    for t in panel.index:
        f = panel.loc[t].values.astype(float)
        r = fwd.loc[t].values.astype(float)
        valid = ~(np.isnan(f) | np.isnan(r))
        if valid.sum() >= min_valid:
            rho, _ = spearmanr(f[valid], r[valid])
            if not np.isnan(rho):
                ics.append(rho)
                dates.append(t)
    ic_arr = np.array(ics)
    if len(ic_arr) < 10:
        return dict(ic=0.0, icir=0.0, n_dates=len(ic_arr), abs_ic=0.0, hit=0.0)
    ic_mean = ic_arr.mean()
    ic_std = ic_arr.std(ddof=1)
    icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
    hit = float((ic_arr > 0).mean()) if ic_mean > 0 else float((ic_arr < 0).mean())
    return dict(ic=float(ic_mean), icir=float(icir), n_dates=int(len(ic_arr)),
                abs_ic=float(abs(ic_arr).mean()), hit=float(hit))


def load_artifact(fact_file):
    """Load signal artifact (date x 15-asset panel) from a persisted factor file."""
    d = json.load(open(fact_file))
    sa = d["validation"]["signal_artifact"]
    raw = base64.b64decode(sa["data"])
    txt = zlib.decompress(raw).decode("utf-8")
    rows = list(csv_iter(txt))
    header = rows[0]
    dates = pd.to_datetime([r[0] for r in rows[1:]])
    M = np.array([[float(x) if x != "" else np.nan for x in r[1:]] for r in rows[1:]])
    return dates, header[1:], M


def csv_iter(txt):
    import csv
    for r in csv.reader(io.StringIO(txt)):
        yield r


def recompute_from_df(df, fid, calc):
    """Compute raw factor panel from the unified df using the calc dict."""
    # calc is dict with 'expression'+'description'; we reconstruct per known factor type
    C = {a: df[f"{a}__close"] for a in ASSETS}
    V = {a: df[f"{a}__volume"] for a in ASSETS}
    H = {a: df[f"{a}__high"] for a in ASSETS}
    L = {a: df[f"{a}__low"] for a in ASSETS}
    ret = {a: df[f"{a}__close"].pct_change() for a in ASSETS}

    def roll(s, n, fn):
        return s.rolling(n).apply(fn, raw=True)

    series = {}
    for a in ASSETS:
        c = C[a]
        if fid.startswith("mom_"):
            skip = 5
            n = 120 if "120" in fid else 10
            series[a] = c / c.shift(n + skip) - 1.0
        elif fid.startswith("kaufman"):
            n = 20
            series[a] = (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()
        elif fid.startswith("ac1_"):
            n = 120
            series[a] = ret[a].rolling(n).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1], raw=True)
        elif fid.startswith("skew_"):
            n = 20
            series[a] = ret[a].rolling(n).skew()
        elif fid.startswith("bb_width_") or fid == "bb_width_20d":
            n = 20
            mid = c.rolling(n).mean()
            sd = c.rolling(n).std()
            series[a] = 4.0 * sd / mid
        elif fid.startswith("vol_z_"):
            n = 20
            v = ret[a].rolling(20).std()
            mu = v.rolling(60).mean()
            sd = v.rolling(60).std()
            series[a] = (v - mu) / sd
        elif fid.startswith("rng_pos_"):
            n = 20