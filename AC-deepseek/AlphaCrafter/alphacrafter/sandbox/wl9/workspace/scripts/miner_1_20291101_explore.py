#!/usr/bin/env python
"""FactorMiner exploration: several novel candidate factor ideas on the 15-asset
cross-asset universe. Computes factor values across all assets and evaluates
cross-sectional rank IC / ICIR vs forward 10d return over available history.

Current sim date 2029-11-01 (visible through 2029-10-31).
"""
import pandas as pd, numpy as np, json, os

CUR = '2029-10-31'
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
DATA = '../persistent/stock_data'

# Build close matrix
px = {}
for s in ASSETS:
    df = pd.read_csv(os.path.join(DATA, s + '.csv'))
    df = df[df['date'] <= CUR].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date')
    df = df.set_index('date')
    px[s] = df['close']
px = pd.DataFrame(px).sort_index()

# Also load VIX for macro condition checks
vix = pd.read_csv('../persistent/index_data/VIX.csv')
vix['date'] = pd.to_datetime(vix['date'])
vix = vix[vix['date'] <= CUR].set_index('date')['close'].sort_index()

rets = px.pct_change()
H = 10  # forward return horizon (trading days)
fwd = px.shift(-H) / px - 1.0

def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return np.nan
    ar = pd.Series(a[m]).rank().values
    br = pd.Series(b[m]).rank().values
    ar = ar - ar.mean(); br = br - br.mean()
    den = np.sqrt((ar**2).sum() * (br**2).sum())
    if den == 0:
        return np.nan
    return float(np.dot(ar, br) / den)

def evaluate(factor_df, label):
    # align
    idx = factor_df.index.intersection(fwd.index)
    f = factor_df.loc[idx]
    fr = fwd.loc[idx]
    dates = f.index
    ics = []
    for dt in dates:
        if dt in vix.index and not np.isfinite(vix.loc[dt]):
            continue
        ic = spearman(f.loc[dt].values, fr.loc[dt].values)
        if np.isfinite(ic):
            ics.append((dt, ic))
    if len(ics) < 20:
        print(f'{label}: insufficient dates {len(ics)}')
        return None
    dtarr = np.array([x[0] for x in ics], dtype='datetime64[ns]')
    icarr = np.array([x[