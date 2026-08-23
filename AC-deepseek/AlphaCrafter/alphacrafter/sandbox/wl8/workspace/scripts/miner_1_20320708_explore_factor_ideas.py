#!/usr/bin/env python
"""
Miner 1 - Factor Exploration: Cross-sectional Sharpe Ratio, Volatility-Weighted Momentum,
and Macro Regime Interaction Factor (VIX-quartile conditional momentum)
Current date: 2032-07-08 (visible through 2032-07-07)
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
from scipy import stats
import json, sys, os, math
from datetime import datetime

# Universe
WATCHLIST = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
INDEX_SIGNALS = ['DXY', 'VIX']

ADMISSION_IC = 0.0070
ADMISSION_ICIR = 0.0840

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
print("=" * 80)
print("FACTOR EXPLORATION - Miner 1 (2032-07-08)")
print("=" * 80)

data = {}
for sym in WATCHLIST:
    df = get_stock_daily_data(sym, days=1000)
    if df is not None and len(df) > 50:
        data[sym] = df.sort_values('date').reset_index(drop=True)

idx_signal = {}
for sym in INDEX_SIGNALS:
    df = get_index_daily_data(sym, days=1000)
    if df is not None and len(df) > 50:
        idx_signal[sym] = df.sort_values('date').reset_index(drop=True)

print(f"\nData loaded: {len(data)} assets, {len(idx_signal)} signals")
common_dates = min(len(df) for df in data.values())
print(f"Common min length: {common_dates} days")

# Build aligned price matrices
closes = {}
dates = None
for sym in WATCHLIST:
    if sym in data:
        closes[sym] = data[sym]['close'].values.astype(float)
        if dates is None:
            dates = data[sym]['date'].values

# Macro signals
vix_close = idx_signal['VIX']['close'].values.astype(float)
dxy_close = idx_signal['DXY']['close'].values.astype(float)

T = len(dates)
print(f"Universe aligned: {T} dates\n")

def compute_forward_returns(c, t, horizons):
    """Compute forward returns for given horizons from index t"""
    fwds = {}
    for h in horizons:
        if t + h < len(c):
            fwds[h] = (c[t+h] - c[t]) / c[t]
        else:
            fwds[h] = np.nan
    return fwds

def compute_ic(factor_scores, fwd_rets, min_cov=8):
    """Cross-sectional Spearman rank IC"""
    pairs = [(s, factor_scores.get(s, np.nan), fwd_rets.get(s, np.nan)) 
             for s in WATCHLIST]
    pairs = [(s, f, r) for s, f, r in pairs 
             if not (np.isnan(f) or np.isnan(r))]
    if len(pairs) >= min_cov:
        factors = [p[1] for p in pairs]
        fwds = [p[2] for p in pairs]
        rho, pval = stats.spearmanr(factors, fwds)
        if not np.isnan(rho):
            return rho, len(pairs)
    return np.nan, 0

def run_ic_test(factor_name, compute_factor, horizons=[5,10,21], min_valid=20):
    """Generic IC test runner"""
    ic_series = {h: [] for h in horizons}
    
    for t in range(200, T - max(horizons) - 2):
        factor_scores, valid = compute_factor(t)
        if valid < 8:
            continue
        
        for h in horizons:
            fwd_rets = {}
            for sym in WATCHLIST:
                if sym in closes and t + h < len(closes[sym]):
                    c = closes[sym]
                    fwd_rets[sym] = (c[t+h] - c[t]) / c[t]
                else:
                    fwd_rets[sym] = np.nan
            
            rho, cov = compute_ic(factor_scores, fwd_rets)
            if not np.isnan(rho):
                ic_series[h].append(rho)
    
    print(f"\n  === {factor_name} ===")
    for h in horizons:
        vals = ic_series[h]
        if len(vals) >= min_valid:
            ic = float(np.mean(vals))
            icir = float(np.mean(vals) / np.std(vals, ddof=1)) if np.std(vals) > 0 else 0.0
            hit = float(np.mean([1 if v > 0 else 0 for v in vals]))
            print(f"  Forward {h:2d}d: IC={ic:.4f} ICIR={icir:.4f} Hit={hit:.2f} n={len(vals)}")
        else:
            print(f"  Forward {h:2d}d: insufficient data ({len(vals)} dates)")


# ==================================================================
# FACTOR 1: Cross-sectional Sharpe Ratio (risk-adjusted momentum)
# ==================================================================
print("-" * 60)
print("FACTOR 1: Cross-sectional Sharpe Ratio (risk-adjusted momentum)")
print("  Factor = mean_return / std_return over window (with skip)")
print("-" * 60)

for win in [20, 40]:
    for skip in [5]:
        factor_id = f"sharpe_mom_{win}d_skip{skip}"
        
        def make_factor_fn(w=win, sk=skip):
            def fn(t):
                scores = {}
                valid = 0
                for sym in WATCHLIST:
                    if sym not in closes or t < w + sk:
                        scores[sym] = np.nan
                        continue
                    c = closes[sym]
                    prices = c[t-w-sk:t-sk]
                    rets = np.diff(prices) / prices[:-1]
                    if len(rets) >= w // 2:
                        mu = np.mean(rets)
                        sigma = np.std(rets, ddof=1)
                        if sigma > 1e-10:
                            scores[sym] = mu / sigma
                            valid += 1
                        else:
                            scores[sym] = 0.0
                    else:
                        scores[sym] = np.nan
                return scores, valid
            return fn
        
        run_ic_test(f"{factor_id}", make_factor_fn())


# ==================================================================
# FACTOR 2: VIX Conditional Sharpe (only when VIX is below median)
# ==================================================================
print("\n" + "-" * 60)
print("FACTOR 2: VIX-Conditional Sharpe Momentum")
print("  Factor = sharpe * I(VIX < rolling_median) - risk-aware in high vol")
print("-" * 60)

# Use fixed VIX regime threshold
vix_median = np.median(vix_close)
print(f"VIX median over full sample: {vix_median:.2f}")
print(f"VIX last value: {vix_close[-1]:.2f}")

for win in [20, 40]:
    for skip in [5]:
        factor_id = f"vix_cond_sharpe_{win}d_skip{skip}"
        
        def make_factor_fn(w=win, sk=skip):
            def fn(t):
                scores = {}
                valid = 0
                vix_t = vix_close[t] if t < len(vix_close) else np.nan
                vix_low = vix_t < vix_median
                
                for sym in WATCHLIST:
                    if sym not in closes or t < w + sk:
                        scores[sym] = np.nan
                        continue
                    c = closes[sym]
                    prices = c[t-w-sk:t-sk]
                    rets = np.diff(prices) / prices[:-1]
                    if len(rets) >= w // 2:
                        mu = np.mean(rets)
                        sigma = np.std(rets, ddof=1)
                        if sigma > 1e-10:
                            sharpe = mu / sigma
                            # In high VIX regimes, we reduce momentum exposure
                            if vix_low:
                                scores[sym] = sharpe  # full long
                            else:
                                scores[sym] = -abs(sharpe)  # negative (defensive)
                            valid += 1
                        else:
                            scores[sym] = 0.0
                    els