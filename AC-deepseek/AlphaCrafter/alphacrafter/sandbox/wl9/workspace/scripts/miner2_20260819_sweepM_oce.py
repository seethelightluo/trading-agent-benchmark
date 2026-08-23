"""miner_2 (2026-08-19): open_close_eff window sweep + overnight-vs-intraday ratio.
Goal: find a variant passing IC/ICIR gate AND max lib corr < 0.5.
"""
from __future__ import annotations
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro

closes = load_closes()
macro = load_macro()


def load_open(days_shift=0):
    out = {}
    for a in ASSETS:
        p = f"../persistent/stock_data/{a}.csv"
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")["open"].astype(float)
    return out


def load_high():
    out = {}
    for a in ASSETS:
        p = f"../persistent/stock_data/{a}.csv"
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")["high"].astype(float)
    return out


opens = load_open()
highs = load_high()


def oce(close, open_, n):
    rng = (close / open_.replace(0, np.nan) - 1.0)
    return rng.rolling(n, min_periods=max(5, n // 2)).mean()


def overnight_ratio(close, open_, prev_close, n=20):
    """(close-open)/(open-prev_close) - overnight vs intraday contribution share."""
    intra = close / open_.replace(0, np.nan) - 1.0
    prev = prev_close.replace(0, np.nan)
    over = open_ / prev - 1.0
    # overnight share of total move: overnight/(|overnight|+|intraday|)
    ratio = over / (over.abs() + intra.abs())
    return ratio.rolling(n, min_periods=max(5, n // 2)).mean()


# prev close series = close shifted by 1 within same asset
prev_closes = {a: closes[a].shift(1) for a in closes}

cand = {}
for n in [10, 15, 20, 30, 40, 60]:
    cand[f"oce_{n}d"] = {a: oce(closes[a], opens[a], n) for a in closes if a in opens}
for n in [10, 20, 30, 40]:
    cand[f"ovr_ratio_{n}d"] = {a: overnight_ratio(closes[a], opens[a], prev_closes[a], n) for a in closes if a in opens}

for name, vals in cand.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()