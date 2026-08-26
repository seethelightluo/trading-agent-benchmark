"""miner_1 research harness (2031-06-12 cycle).

Reads ../persistent/{stock_data,index_data} with a visible-end date, builds a
unified daily panel, computes candidate factor panels on the tradable 15-asset
cross-asset universe, and evaluates rank IC / ICIR at a given horizon.

Observation-only macros (DXY, USDCNY, USDJPY, EURUSD, VIX) are used only for
computation, never traded.  Pure offline research: writes nothing to
persistent state except factor JSONs when the caller persists.

Gates (shared, 15-name universe): abs(IC) >= 0.0070 AND abs(ICIR) >= 0.0840.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA_DIR = Path("../persistent")
STOCK_DIR = DATA_DIR / "stock_data"
IDX_DIR = DATA_DIR / "index_data"

ASSETS = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

DEFAULT_END = "2031-06-11"
DEFAULT_VALID_START = "2026-07-16"
HORIZON = 10
GATE_IC = 0.0070
GATE_ICIR = 0.0840


def load_ohlc(end=DEFAULT_END):
    out = {}
    for a in ASSETS:
        f = STOCK_DIR / f"{a}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= end].set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
        out[a] = df
    return out


def load_macro(end=DEFAULT_END):
    out = {}
    for m in MACRO:
        f = IDX_DIR / f"{m}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= end]
        out[m] = df.set_index("date")["close"].astype(float)
    return out


def build_df(end=DEFAULT_END, start=DEFAULT_VALID_START):
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


def turnover_10d(panel):
    r = panel.rank(axis=1)
    d = r.diff(10).abs().mean(axis=1)
    return float(d.mean())


def panel_from_series(series_map):
    idx = sorted({d for s in series_map.values() for d in s.index})
    panel = pd.DataFrame(index=idx, dtype=float)
    for a, s in series_map.items():
        panel[a] = s
    return panel


def close_panel(df):
    return panel_from_series({a: df[f"{a}__close"] for a in ASSETS})


def ret_panel(df):
    return panel_from_series({a: df[f"{a}__close"].pct_change() for a in ASSETS})


def full_eval(fn, df, panel, horizons=(5, 10, 20), min_valid=8):
    """Evaluate a factor panel at multiple horizons; return summary dict."""
    cp = close_panel(df)
    out = {"n_dates": int(panel.shape[0]), "n_assets": int(panel.shape[1]),
           "coverage": float((~panel.isna()).mean().mean())}
    for h in horizons:
        fwd = fwd_returns(cp, h)
        m = evaluate(panel, fwd, min_valid=min_valid)
        out[f"h{h}"] = m
    out["turnover_10d"] = turnover_10d(panel)
    return out