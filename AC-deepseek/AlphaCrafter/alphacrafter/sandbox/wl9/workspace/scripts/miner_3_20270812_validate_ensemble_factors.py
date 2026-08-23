"""
Re-validate ensemble factors and several effective ones across recent data.
Date: 2027-08-12
Current last trading day: 2027-08-11
"""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def get_data_dict(days=400):
    """Fetch data for all watchlist items"""
    data = {}
    for sym in WATCHLIST:
        df = get_stock_daily_data(symbol=sym, days=days)
        if df is not None:
            data[sym] = df
    return data

def compute_ic_and_icir(factor_values_dict, forward_returns_dict):
    """
    Compute daily cross-sectional IC and ICIR.
    factor_values_dict: {date: {symbol: value}}
    forward_returns_dict: {date: {symbol: fwd_ret}}
    """
    ic_list = []
    for date in sorted(factor_values_dict.keys()):
        if date not in forward_returns_dict:
            continue
        fvals = factor_values_dict[date]
        fret = forward_returns_dict[date]
        common_syms = [s for s in fvals if s in fret and not (np.isnan(fvals[s]) or np.isnan(fret[s]))]
        if len(common_syms) < 8:
            continue
        x = np.array([fvals[s] for s in common_syms])
        y = np.array([fret[s] for s in common_syms])
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            continue
        ic = np.corrcoef(x, y)[0, 1]
        ic_list.append(ic)
    
    if len(ic_list) < 5:
        return 0.0, 0.0, 0, []
    
    ic_arr = np.array(ic_list)
    mean_ic = float(np.mean(ic_arr))
    std_ic = float(np.std(ic_arr))
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    hit_ratio = float(np.mean(ic_arr > 0))
    
    return mean_ic, icir, len(ic_arr), ic_arr

def compute_forward_return(price_dict, hold_days=1):
    """Compute forward returns for each symbol and date"""
    fwd_ret = {}
    for sym, df in price_dict.items():
        prices = df['close'].values
        dates = df['date'].values
        for i in range(len(prices) - hold_days):
            dt = dates[i]
            fwd = prices[i+hold_days] / prices[i] - 1.0
            if dt not in fwd_ret:
                fwd_ret[dt] = {}
            fwd_ret[dt][sym] = float(fwd)
    return fwd_ret

# ============ FACTOR DEFINITIONS ============

def factor_mom_120d_skip5(data_dict, lookback=120, skip=5):
    """Momentum: close / close_{lookback+skip} - 1"""
    vals = {}
    for sym, df in data_dict.items():
        prices = df['close'].values
        dates = df['date'].values
        for i in range(lookback+skip, len(prices)):
            dt = dates[i]
            ret = prices[i] / prices[i-lookback-skip] - 1.0
            if dt not in vals:
                vals[dt] = {}
            vals[dt][sym] = float(ret)
    return vals

def factor_mom_10d_skip5(data_dict, lookback=10, skip=5):
    """Short momentum: close / close_{lookback+skip} - 1"""
    vals = {}
    for sym, df in data_dict.items():
        prices = df['close'].values
        dates = df['date'].values
        for i in range(lookback+skip, len(prices)):
            dt = dates[i]
            ret = prices[i] / prices[i-lookback-skip] - 1.0
            if dt not in vals:
                vals[dt] = {}
            vals[dt][sym] = float(ret)
    return vals

def factor_vix_beta_cond_60x20(data_dict, index_data_dict, lookback=60, short=20):
    """Conditional beta with VIX"""
    vals = {}
    vix_df = index_data_dict.get('VIX')
    if vix_df is None:
        return vals
    
    vix_dates = vix_df['date'].values
    vix_close = vix_df['close'].values
    
    for sym, df in data_dict.items():
        prices = df['close'].values
        dates = df['date'].values
        
        # Build aligned series
        for i in range(lookback, len(prices)):
            dt = dates[i]
            
            # Find VIX window
            # Find end index of VIX at this date
            vix_idx = None
            for j in range(len(vix_dates)):
                if vix_dates[j] >= dt:
                    vix_idx = j
                    break
            if vix_idx is None or vix_idx < lookback:
                continue
            
            vix_window = vix_close[vix_idx-lookback:vix_idx]
            ret_window = []
            for k in range(lookback):
                p_idx = i - lookback + k
                if p_idx > 0:
                    ret_window.append(prices[p_idx] / prices[p_idx-1] - 1.0)
                else:
                    ret_window.append(0.0)
            ret_window = np.array(ret_window)
            
            # Compute beta
            if np.std(vix_window) > 1e-10 and np.std(ret_window) > 1e-10:
                beta = np.cov(vix_window, ret_window)[0,1] / np.var(vix_window)
            else:
                beta = 0.0
            
            # Short-term vol regime
            vix_short = vix_close[max(0,vix_idx-short):vix_idx]
            vix_vol = np.std(vix_short) / np.mean(vix_short) if np.mean(vix_short) > 0 else 0
            
            # Conditional: scale beta by short-term VIX vol
            cond_val = beta * vix_vol
            
            if dt not in vals:
                vals[dt] = {}
            vals[dt][sym] = flo