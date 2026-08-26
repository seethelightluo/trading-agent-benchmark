"""Shared validation harness for miner_1 (2030-02-07 cycle).

Loads the 15-asset tradable universe + macro observation series from
../persistent/ through the visible end, computes factor values, and evaluates
rank IC / ICIR / turnover / coverage / decay / library correlation.
Gates (15-instrument cross-asset universe): abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10.
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

VISIBLE_END = "2030-02-06"   # latest visible data (previous completed trading day)
VALID_START = "2026-07-16"   # recent dedicated validation window (online-start onward)
FULL_START = "2021-01-01"    # longer warm-up validation
HORIZON = 10
GATE_IC = 0.0070
GATE_ICIR = 0.0840


def load_closes(end=VISIBLE_END):
    closes = {}
    for a in ASSETS:
        f = STOCK_DIR / f"{a}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= end]
        closes[a] = df.set_index("date")["close"].astype(float)
    return closes


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


def align_frame(end=VISIBLE_END, start=VALID_START):
    ohlc = load_ohlc(end)
    macro = load_macro(end)
    # union of all dates (weekdays only)
    all_dates = set()
    for df in ohlc.values():
        all_dates.update(df.index)
    for df in macro.values():
        all_dates.update(df.index)
    dates = sorted(d for d in all_dates if start <= d.strftime("%Y-%m-%d") and d.weekday() < 5)
    dates = pd.DatetimeIndex(dates)
    df = pd.DataFrame(index=dates, columns=ASSETS + MACRO, dtype=float)
    for a, d in ohlc.items():
        for col in d.columns:
            df.loc[d.index, f"{a}__{col}"] = d[col].values
    for m, d in macro.items():
        df.loc[d.index, f"{m}__close"] = d.values
    return df


def compute_factor_from_df(df, name, params):
    """Return dict of asset->series (indexed by date). Common helpers use df columns {A}__close etc."""
    C = {a: df[f"{a}__close"] for a in ASSETS}
    P = {a: df[f"{a}__pct_change"] for a in ASSETS}
    out = {}
    for a in ASSETS:
        out[a] = factor_single(C[a], P[a], name, params, df)
    return out


def factor_single(c, p, name, params, df):
    raise NotImplementedError


def build_panel(signal_series, asset_keys=ASSETS):
    """signal_series: dict asset->pd.Series. Return (date_index, panel ndarray)."""
    idx = sorted({d for s in signal_series.values() for d in s.index})
    panel = pd.DataFrame(index=idx, columns=asset_keys, dtype=float)
    for a, s in signal_series.items():
        if a in asset_keys:
            panel[a] = s
    return panel


def fwd_returns(close_panel, horizon=HORIZON):
    fwd = close_panel.shift(-horizon) / close_panel - 1.0
    return fwd


def evaluate(panel, fwd, min_valid=8):
    """Cross-sectional rank IC per date; return metrics dict."""
    ics = []
    dates = []
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
                abs_ic=float(np.abs(ic_arr).mean()), hit=float(hit))


def turnover(panel, q=0.5