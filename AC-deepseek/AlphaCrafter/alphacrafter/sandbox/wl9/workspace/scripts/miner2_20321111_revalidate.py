"""miner_2 fresh re-validation harness (2032-11-11 cycle).

Recomputes the effective-factor library panels directly from raw CSV data
(restricted to PROTECT <= current date 2032-11-10, the previous completed
trading day) and evaluates rank IC / ICIR at h=10 on the tradable 15-asset
cross-asset universe. This is PURE OFFLINE research: reads ../persistent/*.csv,
writes ONLY to factors/*.json. Never advances the live account.

Gates (shared, 15-name universe): abs(IC) >= 0.0070 AND abs(ICIR) >= 0.0840.
"""
from __future__ import annotations
import json, math
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

VISIBLE_END = "2032-11-10"   # previous completed trading day
HORIZON = 10
GATE_IC = 0.0070
GATE_ICIR = 0.0840

# fresh OOS-ish revalidation window: last ~44 months
WIN_START = "2028-11-01"

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
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= end].set_index("date")["close"].astype(float)
        out[m] = df
    return out

def build_df(end=VISIBLE_END, start="2020-01-01"):
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

def factor_panels(df):
    C = {a: df[f"{a}__close"] for a in ASSETS}
    V = {a: df[f"{a}__volume"] for a in ASSETS}
    H = {a: df[f"{a}__high"] for a in ASSETS}
    L = {a: df[f"{a}__low"] for a in ASSETS}