"""Shared validation harness for miner_1 (2030-01-24 cycle).

Loads the 15-asset tradable universe + macro observation series from
../persistent/ through the visible end, computes factor values, and evaluates
rank IC / ICIR / turnover / coverage / decay / library correlation.
Gates (15-instrument cross-asset universe): abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10.
"""
from __future__ import annotations
import base64, io, json, zlib
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

VISIBLE_END = "2030-01-23"   # latest visible data (previous completed trading day)
VALID_START = "2026-07-16"   # recent dedicated validation window (online-start onward)
FULL_START = "2021-01-01"    # longer warm-up validation


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


def to_frame(closes, values, start=VALID_START):
    dates = sorted({d for s in closes.values() for d in s.index})
    dates = [d for d in pd.DatetimeIndex(dates) if start <= d.strftime("%Y-%m-%d") and d.weekday() < 5]
    dates = pd.DatetimeIndex(dates)
    df = pd.DataFrame(index=dates, columns=ASSETS, dtype=float)
    for a, s in values.items():
        if a in