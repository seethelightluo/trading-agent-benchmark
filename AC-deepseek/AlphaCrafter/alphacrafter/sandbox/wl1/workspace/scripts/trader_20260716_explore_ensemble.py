"""Trader exploration: compute ensemble factor signals as of 2026-07-15."""
import json
import math
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

with open('factor_ensemble.json') as f:
    ENS = json.load(f)['selected_factors']
print("Ensemble factors:")
for e in ENS:
    print(f"  {e['factor_id']:35s} w={e['weight']:.4f} dir={e['direction']}")

# --- Load daily OHLCV for each asset ---
frames = {}
for a in WATCH:
    df = get_stock_daily_data(symbol=a, days=250)
    if df is None or len(df) < 40:
        print(f"WARN {a}: insufficient data {0 if df is None else len(df)}")
        continue
    df = df.sort_values('date').reset_index(drop=True)
    frames[a] = df
print("\nLast date per asset:", sorted({str(v['date'].iloc[-1])[:10] for v in frames.values()}))

def factor_value(expr_name, df):
    """Compute the factor value for the LAST row given full df."""
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    o = df['open'].astype(float)
    if expr_name == 'nclv_1d':
        rng = (h - l).clip(lower=1e-12)
        return float(-(c - l) / rng)
    if expr_name == 'nclv_2d':
        rng = (h.rolling(2).max() - l.rolling(2).min()).clip(lower=1e-12)
        return float(-(c - l.rolling(2).min()) / rng)
    if expr_name == 'nclv_3d':
        rng = (h.rolling(3).max() - l.rolling(3).min()).clip(lower=1e-12)
        return float(-(c - l.rolling(3).min()) / rng)
    if expr_name == 'nbody_1d':
        rng = (h - l).clip(lower=1e-12)
        return float(-(c - o) / rng)
    if expr_name == 'rev_1d':
        return float(-(np.log(c.iloc[-1]) - np.log(c.iloc[-2])))
    if expr_name == 'rev_2d':
        return float(-(np.log(c.iloc[-1]) - np.log(c.iloc[-3])))
    if expr_name == 'mom_10d_skip5':
        return float(np.log(c.iloc[-1] / c.iloc[-11]) - np.log(c.iloc[-1] / c.iloc[-6]))
    raise ValueError(expr_name)

# --- Cross-sectional factor values ---
vals = {}
for e in ENS:
    fid = e['factor_id']
    # map factor_id -> expr name
    if 'nclv_1d' in fid:
        expr = 'nclv_1d'
    elif 'nclv_2d' in fid:
        expr = 'nclv_2d'
    elif 'nclv_3d' in fid:
        expr = 'nclv_3d'
    elif 'nbody_1d' in fid:
        expr = 'nbody_1d'
    elif 'rev_1d' in fid:
        expr = 'rev_1d'
    elif 'rev_2d' in fid:
        expr = 'rev_2d'
    elif 'mom_10d_skip5' in fid:
        expr = 'mom_10d_skip5'
    else:
        expr = None
    vals[fid] = {}
    for a, df in frames.items():
        try:
            vals[fid][a] = factor_value(expr, df)
        except Exception as ex:
            vals[fid][a] = None

tab = pd.DataFrame(vals)
print("\nRaw factor values (last obs):")
print(tab.round(4).to_string())

# --- Rank-normalize and combine ---
def ranks(s):
    s = s.dropna()
    r = s.rank(pct=True)
    return r

score = pd.Series(0.0, index=WATCH)
for e in ENS:
    fid = e['factor_id']
    s = tab[fid].dropna()
    r = s.rank(pct=True)  # 0..1 cross-sectionally
    score = score.add(r * e['weight'] * e['direction'], fill_value=0.0)

sc = score.sort_values(ascending=False)
print("\nComposite scores:")
print(sc.round(4).to_string())

# --- Simple rank-based weights ---
N = len(WATCH)
rank_order = list(sc.index)
lin = {a: (N - i) for i, a in enumerate(rank_order)}  # linear rank weights
tot = sum(lin.values())
w = {a: lin[a] / tot for a in WATCH}
print("\nLinear-rank weights:")
for a in rank_order:
    print(f"  {a:10s} rank={rank_order.index(a)+1:2d} w={w[a]:.4f}")

# --- Regime context: recent market moves ---
panel = pd.concat([frames[a]['close'].astype(float).pct_change().rename(a) for a in frames], axis=1).dropna()
mkt = panel.mean(axis=1)
print("\nMarket (equal-weight) 5d/21d/63d ret:",
      round(float((1 + mkt.tail(5)).prod() - 1), 4),
      round(float((1 + mkt.tail(21)).prod() - 1), 4),
      round(float((1 + mkt.tail(63)).prod() - 1), 4))
print("Vol 21d (mkt):", round(float(mkt.tail(21).std() * np.sqrt(252)), 4))
try:
    vix = get_index_daily_data('VIX', days=70)
    if vix is not None:
        vix = vix.sort_values('date')
        print("VIX last:", float(vix['close'].iloc[-1]), "5d ago:", float(vix['close'].iloc[-6]))
except Exception as ex:
    print("vix err", ex)
