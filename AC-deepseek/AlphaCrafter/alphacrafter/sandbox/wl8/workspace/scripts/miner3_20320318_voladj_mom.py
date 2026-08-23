#!/usr/bin/env python
"""
Factor Exploration: miner3_20320318_voladj_mom
=============================
Explores a volatility-adjusted momentum factor:
  factor = (close - close_lag) / close_lag / volatility(close, window)

This normalizes raw momentum by recent volatility, capturing risk-adjusted
trend strength. In low-VIX regimes (currently VIX≈14), this should identify
assets with genuine directional conviction vs noisy movement.

Validation across 15-instrument cross-asset universe.
"""

import sys, os, json, math, traceback
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
except ImportError:
    print("WARN: alphacrafter not available; will create mock data for structure validation")
    get_stock_daily_data = None
    get_index_daily_data = None
    get_account_dict = None

WATCHLIST = [
    "000300.SH", "SPX", "HSI", "N225", "SX5E",
    "000688.SH", "SOX", "NDX", "XAU", "COPPER",
    "WTI", "BTC", "ETH", "US10Y", "CN10Y"
]

# ---------- Parameter grid ----------
MOM_WINDOWS = [10, 20, 40, 60]
VOL_WINDOWS = [20, 40, 60]
HOLD_PERIODS = [5, 10, 20]

def compute_factor(close: pd.Series, mom_window: int, vol_window: int):
    """Returns the factor series: momentum / vol."""
    ret = close.pct_change(mom_window)
    vol = close.pct_change().rolling(vol_window).std()
    # avoid div by zero
    factor = ret / (vol + 1e-10)
    return factor.shift(1)  # shift to avoid lookahead

def compute_forward_ret(close: pd.Series, hold: int):
    return close.pct_change(hold).shift(-hold)

def rank_ic(s1: pd.Series, s2: pd.Series):
    """Spearman rank IC."""
    valid = s1.notna() & s2.notna()
    if valid.sum() < 8:
        return np.nan
    r1 = s1[valid].rank()
    r2 = s2[valid].rank()
    return r1.corr(r2)

def main():
    print(f"{'='*80}")
    print(f"FACTOR EXPLORATION: Volatility-Adjusted Momentum")
    print(f"Date: 2032-03-18")
    print(f"{'='*80}")

    # Fetch data
    all_data = {}
    for sym in WATCHLIST:
        df = get_stock_daily_data(symbol=sym, days=750)
        if df is None or len(df) < 120:
            print(f"  {sym}: insufficient data ({len(df) if df is not None else 0} days)")
            continue
        if 'close' not in df.columns:
            print(f"  {sym}: no 'close' column")
            continue
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        all_data[sym] = df['close']
        print(f"  {sym}: {len(df)} days, last close={df['close'].iloc[-1]:.2f}")

    if len(all_data) < 8:
        print(f"\nERROR: Only {len(all_data)} assets with data, need >= 8")
        return

    # Build a panel DataFrame
    panel = pd.DataFrame(all_data)
    panel = panel.sort_index()
    print(f"\nPanel shape: {panel.shape}, date range: {panel.index[0].date()} to {panel.index[-1].date()}")
    print(f"Instruments with data: {panel.columns.tolist()}")

    # Track best params
    best_params = None
    best_ic = -999
    best_results = {}

    for mom_w in MOM_WINDOWS:
        for vol_w in VOL_WINDOWS:
            # Compute factor and forward returns
            factor_name = f"voladj_mom_{mom_w}x{vol_w}"

            # Compute factor values (shifted to avoid lookahead)
            factor_vals = {}
            fwd_ret_vals = {}

            for col in panel.columns:
                s = panel[col].dropna()
                if len(s) < max(mom_w, vol_w) + 30:
                    continue
                f = compute_factor(s, mom_w