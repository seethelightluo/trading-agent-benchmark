"""
miner_1 exploration v3: per-instrument calendar factors -> union-grid panels.
Data through 2029-03-21. Cross-sectional daily Spearman IC, horizons 1,3,5,10,20.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU',
         'COPPER','WTI','BTC','ETH','US10Y','CN10Y']

# ---------- load ----------
closes = {}
for s in WATCH:
    df = get_stock_daily_data(symbol=s, days=4000)
    if df is None or len(df) < 300:
        continue
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    closes[s] = df.set_index('date')['close'].sort_index()
print('loaded:', len(closes))

idx = sorted(set().union(*[set(c.index) for c in closes.values()]))

def per_instrument_factors(c):
    """All factor series computed on the instrument's OWN calendar."""
    r = c.pct_change()
    f = pd.DataFrame(index=c.index)
    f['mom20'] = c.shift(1) / c.shift(21) - 1.0
    f['mom20_skip5'] = c.shift(5) / c.shift(25) - 1.0
    f['mom60'] = c.shift(1) / c.shift(61) - 1.0
    f['mom120'] = c.shift(1) / c.shift(121) - 1.0
    v20 = r.rolling(20).std()
    v60 = r.rolling(60).std()
    v5 = r.rolling(5).std()
    f['volscaled_mom20x60'] = f['mom20'] / v60.replace(0, np.nan)
    f['trend_exhaust_20x120'] = (f['mom20'] - f['mom120']) / v60.replace(0, np.nan)
    f['trend_aligned_ret5'] = r.rolling(5).sum() * np.sign(f['mom60'])
    f['trend_aligned_ret5_neg'] = -r.rolling(5).sum() * np.sign(f['mom60'])
    pos = r.where(r > 0, 0.0)
    neg = r.where(r < 0, 0.0)
    f['downside_ratio_60'] = (-neg.rolling(60).mean()) / pos.rolling(60).mean().replace(0, np.nan)
    f['skew_60'] = r.rolling(60).skew()
    f['bollinger_pos_20'] = (c - c.rolling(20).mean()) / (v20 * c).replace(0, np.nan)
    f['dist_high_60'] = (c / c.rolling(60).max() - 1.0) / v60.replace(0, np.nan)
    f['vol_ratio_5x60'] = v5 / v60.replace(0, np.nan)
    f['calmness20'] = (r.abs() < 0.5 * r.rolling(20).std()).rolling(20).mean()
    f['vol_of_vol20x60'] = v20.rolling(60).std()
    f['mom_sign_consistency'] = np.sign(r.rolling(10).sum()).rolling(60).mean()
    f['maxdd_60'] = c / c.rolling(60).max() - 1.0
    # autocorr lag-1 over 20d
    r1 = r.shift(1)
    num = ((r - r.rolling(20).mean()) * (r1 - r1.rolling(20).mean())).rolling(20).mean()
    den = r.rolling(20).std() * r1.rolling(20).std()
    f['autocorr1_20'] = num / den.replace(0, np.nan)
    # new: 60d rank of 20d momentum (relative trend strength vs own history)
    f['mom20_rank60'] = f['mom20'].rolling(60).rank(pct=True)
    # new: 10d reversal vs 60d trend (buy dips in uptrend, sell rips in downtrend)
    f['dip_quality'] = -np.sign(f['mom60']) * r.rolling(5).sum()
    return f

factors = {s: per_instrument_factors(c) for s, c in closes.items()}
fnames = list(factors[list(closes.keys())[0]].columns)

# forward returns per instrument on own calendar
def fwd_panel(h):
    P = pd.DataFrame(index=idx, columns=WATCH, dtype=float)
    for s, c in closes.items():
        P[s] = (c.shift(-h) / c - 1.0).reindex(idx)
    return P

def factor_panel(name):
    P = pd.DataFrame(index=idx, columns=WATCH, dtype=float)
    for s, f in factors.items():
        P[s] = f[name].reindex(idx)
    return P

def ic_stats(fmat, fr, start=None, end=None):
    dates = fmat.index
    if start is not None:
        dates = dates[dates >= start]
    if end is not None:
        dates = dates[dates <= end]
    ics, ns = [], []
    for t in dates:
        fv = fmat.loc[t].values
        ff = fr.loc[t].values
        mask = np.isfinite(fv) & np.isfinite(ff)
        if mask.sum() >= 8:
            ic = spearmanr(fv[mask], ff[mask]).statistic
            if np.isfinite(ic):
                ics.append(ic); ns.append(mask.sum())
    if len(ics) < 20:
        return None
    ics = np.array(ics)
    return dict(ic=ics.mean(), icir=ics.mean()/ics.std(), hit=(ics>0).mean(),
                n=len(ics), n_med=int(np.median(ns)))

print('\n=== HORIZON 10 full window ===')
res = {}
fr10 = fwd_panel(10)
for name in fnames:
    st = ic_stats(factor_panel(name), fr10)
    if st is None:
        print(f'{name:28s} insufficient'); continue
    res[name] = st
    print(f'{name:28s} IC={st["ic"]:+.4f} ICIR={st["icir"]:+.3f} hit={st["hit"]:.3f} n={st["n"]} medcov={st["n_med"]}')

print('\n=== RECENT 2028-01..2029-03 H10 ===')
for name in fnames:
    st = ic_stats(factor_panel(name), fr10, start='2028-01-01')
    if st is None:
        print(f'{name:28s} insufficient'); continue
    print(f'{name:28s} IC={st["ic"]:+.4f} ICIR={st["icir"]:+.3f} hit={st["hit"]:.3f} n={st["n"]}')

print('\n=== DECAY for top-5 (full window) ===')
top = sorted(res.items(), key=lambda kv: -abs(kv[1]['ic']*kv[1]['icir']))[:5]
for name, st in top:
    line = f'{name:28s}'
    for h in [1,3,5,10,20]:
        s = ic_stats(factor_panel(name), fwd_panel(h))
        line += f' h{h}:{s["ic"]:+.4f}' if s else f' h{h}:NA'
    print(line)
