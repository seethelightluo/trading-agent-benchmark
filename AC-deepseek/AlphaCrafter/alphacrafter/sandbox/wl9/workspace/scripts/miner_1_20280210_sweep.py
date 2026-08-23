"""miner_1 2028-02-10 factor revalidation + new candidate sweep.

Current date: 2028-02-10. Last visible trading day: 2028-02-09.
Uses the shared 15-instrument cross-asset universe and the shared admission
gates: abs(IC) >= 0.0070 and abs(ICIR) >= 0.0840 at h=10.

This script ONLY computes metrics; no factor files are written here.
"""
from __future__ import annotations
import base64, io, json, zlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("../persistent")
STOCK_DIR = DATA_DIR / "stock_data"
IDX_DIR = DATA_DIR / "index_data"
FACTOR_DIR = Path("factors")

ASSETS = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

VISIBLE_END = "2028-02-09"
GRID_START = "2020-01-01"
GRID_END = "2028-02-09"
RECENT_START = "2025-06-01"   # recent regime sub-window for drift check


def load_closes():
    closes = {}
    for a in ASSETS:
        f = STOCK_DIR / f"{a}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= VISIBLE_END]
        closes[a] = df.set_index("date")["close"].astype(float)
    return closes


def load_macro():
    out = {}
    for m in MACRO:
        f = IDX_DIR / f"{m}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= VISIBLE_END]
        out[m] = df.set_index("date")["close"].astype(float)
    return out


def weekday_grid(closes, start=GRID_START, end=GRID_END):
    dates = sorted({d for s in closes.values() for d in s.index})
    dates = [d for d in dates if start <= d.strftime("%Y-%m-%d") <= end and d.weekday() < 5]
    return pd.DatetimeIndex(dates)


def to_frame(closes, values, dates=None):
    if dates is None:
        dates = weekday_grid(closes)
    df = pd.DataFrame(index=dates, columns=ASSETS, dtype=float)
    for a, s in values.items():
        if a in df.columns:
            df[a]