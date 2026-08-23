"""
Factor Mining: Explore new factor ideas for high-vol regime (VIX~47)
Date: 2032-07-08

Candidates:
1. sharpe_ratio_20: 20d return / 20d vol (risk-adjusted momentum)
2. vol_trend_10x60: 10d vol / 60d vol ratio (volatility regime factor)
3. expansion_ratio_20: (close - low_20) / (high_20 - low_20) range position
4. mom_dispersion_10: 10d momentum z-scored cross-sectionally
"""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd, numpy as np
from scipy.stats import pearsonr

np.random.seed(42)
watchlist = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

print("=== Fetching data ===")
data = {}
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=800)
    if df is not None and len(df) > 250:
        data[sym] = df
print(f"Loaded {len(data)} instruments with sufficient history")

vix_df = get_index_daily_data(symbol='VIX', days=800)

# ============================================================
# FACTOR DEFINITIONS
# ============================================================

def sharpe_ratio_20(df):
    c = df['close'].values
    n = len(c)
    if n < 21: return None
    r = np.full(n, np.nan)
    r[1:] = c[1:] / c[:-1] - 1
    ret20 = np.full(n, np.nan)
    if n > 20: ret20[20:] = c[20:] / c[:-20] - 1
    vol20 = pd.Series(r).rolling(20).std().values
    sr = np.full(n, np.nan)
    mask = (np.isfinite(ret20)) & (np.isfinite(vol20)) & (vol20 > 1e-10)
    sr[mask] = ret20[mask] / vol20[mask]
    return np.where(np.isfinite(sr), sr, np.nan)

def vol_trend_10x60(df):
    c = df['close'].values
    n = len(c)
    if n < 61: return None
    r = np.full(n, np.nan)
    r[1:] = c[1:] / c[:-1] - 1
    v10 = pd.Series(r).rolling(10).std().values
    v60 = pd.Series(r).rolling(60).std().values
    vr = np.full(n, np.nan)
    mask = (np.isfinite(v10)) & (np.isfinite(v60)) & (v60 > 1e-10)
    vr[mask] = v10[mask] / v60[mask]
    return np.where(np.isfinite(vr), vr, np.nan)

def expansion_ratio_20(df):
    c = df['close'].values
    n = len(c)
    if n < 21: return None
    h20 = pd.Series(c).rolling(20).max().values
    l20 = pd.Series(c).rolling(20).min().values
    er = np.full(n, np.nan)
    mask = (h20 - l20) > 1e-10
    er[mask] = (c[mask] - l20[mask]) / (h20[mask] - l20[mask])
    return np.where(np.isfinite(er), er, np.nan)

def mom_dispersion_10(df):
    c = df['close'].values
    n = len(c)
    if n < 11: return None
    mom10 = np.full(n, np.nan)
    if n > 10: mom10[10:] = c[10:] / c[:-10] - 1
    return np.where(np.isfinite(mom10), mom10, np.nan)

# ============================================================
# IC COMPUTATION WITH CROSS-SECTIONAL Z-SCORING
# ============================================================
def compute_ics(fn, fwd=10, max_dt=None, label='', zscore_cross=True):
    if not data:
        return dict(ic=0, icir=0, n=0, label=label)
    fs = list(data.keys())[0]
    dates = data[fs]['date'].values
    n = len(dates)
    
    ics, ds, cs = [], [], []
    si = 200
    ei = n - fwd
    
    if max_dt and ei - si > max_dt:
        si = ei - max_dt
    
    for i in range(si, ei):
        dt = dates[i]
        fv, rv = {}, {}
        
        for sym in data:
            df = data[sym]
            fa = fn(df.iloc[:i+1])
            if fa is not None and len(fa) > 1:
                v = fa[-1]
                if np.isfinite(v):
                    fv[sym] = v
        
        for sym in data:
            df = data[sym]
            if i + fwd < len(df):
                r = df.iloc[i+fwd]['close'] / df.iloc[i]['close'] - 1
                if np.isfinite(r):
                    rv[sym] = r
        
        common = sorted(set(fv.keys()) & set(rv.keys()))
        if len(common) >= 8:
            fva = np.array([fv[s] for s in common])
            rva = np.array([rv[s] for s in common])
            
            if zscore_cross:
                f_mean = np.mean(fva)
                f_std = np.std(fva) if np.std(fva) > 1e-10 else 1
                fva = (fva - f_mean) / f_std
            
            mask = np.isfinite(fva) & np.isfinite(rva)
            if mask.sum() >= 8:
                ic_val, _ = pearsonr(fva[mask], rva[mask])
                ics.append(ic_val)
                ds.append(dt)
                cs.append(mask.sum())
    
    arr = np.array(ics)
    if len(arr) == 0:
        return dict(ic=0, icir=0, n=0, label=label)
    
    mean_ic = np.mean(arr)
    std_ic = np.std(arr) if np.std(arr) > 0 else 1e-10
    icir_val = mean_ic / std_ic
    hit_ratio = np.mean(np.sign(arr) == np.sign(mean_ic)) if mean_ic != 0 else 0.5
    pos_ratio = np.mean(arr > 0)
    
    return dict(
        ic=float(f'{mean_ic:.6f}'),
        icir=float(f'{icir_val:.6f}'),
        hit=float(f'{hit_ratio:.4f}'),
        pos_ratio=float(f'{pos_ratio:.4f}'),
        n_obs=len(arr),
        avg_cov=float(f'{np.mean(cs):.1f}'),
        frm=str(ds[0])[:10],
        to=str(ds[-1])[:10],
        label=label
    )

# ============================================================
# RUN VALIDATIONS
# ============================================================
print("\n" + "="*70)
print("FACTOR VALIDATION - FULL PERIOD")
ref_sym = list(data.keys())[0]
print(f"Data range: {data[ref_sym].iloc[0]['date']} to {data[ref_sym].iloc[-1]['date']}")
print(f"Using {len(data)} instruments, cross-section z-scoring applied")
print("="*70)

test_factors = [
    (sharpe_ratio_20, "sharpe_ratio_20", "Risk-adjusted momentum (20d ret/20d vol)"),
    (vol_trend_10x60, "vol_trend_10x60", "Vol trend ratio (10d vol/60d vol)"),
    (expansion_ratio_20, "expansion_ratio_20", "Expansion ratio (close-low)/(high-low) 20d"),
    (mom_dispersion_10, "mom_dispersion_10", "10d momentum z-scored cross-sectionally"),
]

for fn, fname, desc in test_factors:
    print(f"\n--- {desc} ---")
    for fwd in [5, 10, 21]:
        r_full = compu