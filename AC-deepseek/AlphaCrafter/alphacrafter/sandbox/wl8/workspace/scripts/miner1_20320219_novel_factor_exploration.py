"""Factor Miner 1 - Novel Factor Exploration
Current date: 2032-02-19, visible through 2032-02-18.
We test several novel factor ideas that haven't been tried before.

All factors are computed per-asset on their dense calendar, then aligned.
"""
import numpy as np
import pandas as pd
import sys, os, json, base64, zlib, io

# Setup paths
sys.path.insert(0, '.')
from scripts.factor_validation_lib import (
    ASSETS, DATA_DIR, INDEX_DIR, IC_GATE, ICIR_GATE, MIN_ASSETS_PER_DATE,
    load_closes, dense_per_asset, factor_panel, fwd_returns,
    ic_series, turnover_rank, coverage, load_library_panels,
    max_library_corr, artifact_b64, print_result
)

# Override CURRENT_DATE to 2032-02-18 (last completed trading day)
CURRENT_DATE = pd.Timestamp("2032-02-18")

# Reload with correct date
def load_closes_custom(end_date):
    closes, vols, opens, highs, lows = {}, {}, {}, {}, {}
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_DIR}/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= end_date].set_index("date").sort_index()
        closes[a] = df["close"].astype(float)
        vols[a] = df["volume"].astype(float)
        opens[a] = df["open"].astype(float)
        highs[a] = df["high"].astype(float)
        lows[a] = df["low"].astype(float)
    close = pd.DataFrame(closes)
    vol = pd.DataFrame(vols)
    open_ = pd.DataFrame(opens)
    high = pd.DataFrame(highs)
    low = pd.DataFrame(lows)
    return close, vol, open_, high, low

def load_index_custom(n