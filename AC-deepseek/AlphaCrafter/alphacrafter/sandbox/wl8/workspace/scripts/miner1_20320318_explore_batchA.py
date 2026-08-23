"""
Miner 1 - Batch A Exploration: Cross-asset dispersion & breadth momentum factors
Date: 2032-03-18

Exploring novel factor ideas for the 15-instrument cross-asset universe:
1. dispersion_20: Cross-sectional std of 20d returns
2. breadth_mom_20: Fraction of assets with positive 20d momentum
3. rank_mom_20: Rank-based momentum (cross-sectional rank of 20d return)
4. dxy_corr_60: Asset correlation with DXY over 60d
5. vix_carry_20: VIX change regime factor
"""
import sys, os, json, numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

sys.setrecursionlimit(10000)

WATCHLIST = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
OBSERVABLES = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
MIN_DAYS = 120

# Load data for all watchlist symbols
prices = {}
for sym in WATCHLIST:
    df = get_stock_daily_data(sym, days=700)
    if df is None or len(df) < MIN_DAYS:
        df = get_index_daily_data(sym, days=700)
    if df is not None and len(df) >= MIN_DAYS:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        prices[sym] = df['close']

# Load observables
obs_data = {}
for sym in OBSERVABLES:
    df = get_index_daily_data(sym, days=700)
    if df is not None and len(df) >= MIN_DAYS:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        obs_data[sym] = df['close']

# Build panel
panel = pd.DataFrame(prices)
panel = panel.dropna(how='all', axis=1)
print(f"Panel shape: {panel.shape}, date range: {panel.index[0].date()} to {panel.index[-1].date()}")
print(f"Assets: {list(panel.columns)}")

# Compute returns
fwd_ret = panel.pct_change(10).shift(-10)
ret_1d = panel.pct_change(1)
ret_5d = panel.pct_change(5)
ret_10d = panel.pct_change(10)
ret_20d = panel.pct_change(20)
ret_60d = panel.pct_change(60)

vix = obs_data.get('VIX')
dxy = obs_data.get('DXY')

common_idx_all = panel.index.intersection(fwd_ret.dropna(how='all').index)
common_idx_all = common_idx_all[common_idx_all >= panel.index[0]]

def compute_ic_cross_sectional(factor_panel, fwd_ret, label, min_assets=8):
    """Compute IC metrics for a cross-sectional factor."""
    valid_dates = []
    ics = []
    n_assets = []
    
    for dt in factor_panel.index.intersection(fwd_ret.index):
        if dt not in factor_panel.index or dt not in fwd_ret.index:
            continue
        fv = factor_panel.loc[dt]
        fr = fwd_ret.loc[dt]
        # Both need valid values
        valid = fv.notna() & fr.notna()
        if valid.sum() >= min_assets:
            valid_dates.append(dt)
            r = np.corrcoef(fv[valid].values.astype(float), fr[valid].values.astype(float))[0, 1]
            if not np.isnan(r):
                ics.append(r)
                n_assets.append(valid.sum())
    
    if len(ics) > 0:
        ic_arr = np.array(ics)
        mean_ic = np.mean(ic_arr)
        std_ic = np.std(ic_arr, ddof=1) if len(ic_arr) > 1 else 1.0
        icir_val = mean_ic / std_ic if std_ic > 0 else 0
        hit = np.sum(np.sign(ic_arr) == np.sign(mean_ic)) / len(ic_arr)
        
        print(f"\n=== {label} ===")
        print(f"  Dates used: {len(valid_dates)}")
        print(f"  Mean assets/date: {np.mean(n_assets):.1f}")
        print(f"  Mean IC: {mean_ic:.6f}")
        print(f"  IC Std: {std_ic:.6f}")
        print(f"  ICIR: {icir_val:.6f}")
        print(f"  IC Hit Ratio: {hit:.4f}")
        print(f"  Gate check: |IC|>={0.0070}? {abs(mean_ic) >= 0.0070}, |ICIR|>={0.0840}? {abs(icir_val) >= 0.0840}")
        return {'mean_ic': float(mean_ic), 'icir': float(icir_val), 'hit': float(hit), 'n_dates': len(valid_dates)}
    else:
        print(f"\n=== {label} === NO VALID DATES")
        return None


# ============================================
# FACTOR 1: dispersion_20
# ============================================
print("\n" + "="*70)
print("FACTOR 1: dispersion_20 - cross-sectional dispersion of 20d returns")
print("="*70)
dispersion = ret_20d.std(axis=1)
factor1 = dispersion.shift(1)
factor1_panel = pd.DataFrame({c: factor1 for c in panel.columns}, index=factor1.index)
res1 = compute_ic_cross_sectional(factor1_panel, fwd_ret, "dispersion_20")

# ============================================
# FACTOR 2: breadth_mom_20
# ============================================
print("\n" + "="*70)
print("FACTOR 2: breadth_mom_20 - fraction of assets with positive 20d momentum")
print("="*70)
breadth = (ret_20d > 0).sum(axis=1) / ret_20d.notna().sum(axis=1)
factor2 = breadth.shift(1)
factor2_panel = pd.DataFrame({c: factor2 for c in panel.columns}, index=factor2.index)
res2 = compute_ic_cross_sectional(factor2_panel, fwd_ret, "breadth_mom_20")

# ============================================
# FACTOR 3: rank_mom_20
# ============================================
print("\n" + "="*70)
print("FACTOR 3: rank_mom_20 - cross-sectional rank of 20d return")
print("="*70)
rank_mom = ret_20d.rank(axis=1, pct=True)
factor3 = rank_mom.shift(1)
# Already asset-specific
res3 = compute_ic_cross_sectional(factor3, fwd_ret, "rank_mom_20")

# ============================================
# FACTOR 4: dxy_corr_60 - asset correlation with DXY
# ============================================
print("\n" + "="*70)
print("FACTOR 4: dxy_corr_60 - asset 60d correlation with DXY")
print("="*70)
if dxy is not None:
    dxy_ret = dxy.pct_change()
    # Align dxy with panel
    aligned_ret = ret_1d.align(dxy_ret, join='inner', axis=0)[0]
    dxy_aligned = dxy_ret.align(ret_1d, join='inner', axis=0)[0]
    
    # Rolling 60d correlation
    corr_vals = {}
    for c in panel.columns:
        if c in ret_1d.columns:
            asset_ret = ret_1d[c]
            combined = pd.concat([asset_ret, dxy_aligned], axis=1, keys=['asset', 'dxy'])
            combined = combined.dropna()
            roll_corr = combined['asset'].rolling(60).corr(combined['dxy'])
            corr_vals[c] = roll_corr.shift(1)
    
    dxy_corr = pd.DataFrame(corr_vals)
    res4 = compute_ic_cross_sectional(dxy_corr, fwd_