"""miner_3 exploration 2031-06-26: volume/liquidity and range-based factors.
Validate on full visible history up to 2031-06-25 (last completed day).
Admission gates: abs daily paper IC >= 0.0070 and abs ICIR >= 0.084.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = '2031-06-25'
STOCK_DIR = Path('../persistent/stock_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

closes, highs, lows, vols = {}, {}, {}, {}
for a in ASSETS:
    df = pd.read_csv(STOCK_DIR / f'{a}.csv', parse_dates=['date'])
    df = df[df['date'] <= VISIBLE_END].sort_values('date').set_index('date')
    closes[a] = df['close'].astype(float)
    highs[a] = df['high'].astype(float)
    lows[a] = df['low'].astype(float)
    vols[a] = df['volume'].astype(float)

close = pd.DataFrame(closes).dropna()
high = pd.DataFrame(highs).reindex(close.index)
low = pd.DataFrame(lows).reindex(close.index)
vol = pd.DataFrame(vols).reindex(close.index)
rets = close.pct_change().dropna()
ret_idx = rets.index
fwd = rets.shift(-10).rolling(10).mean()
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, "
      f"{close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}")

def compute_ic(fv):
    fv = fv.reindex(ret_idx)
    ics = []
    for d in ret_idx:
        f = fv.loc[d]; r = fwd.loc[d]
        m = f.notna() & r.notna()
        if m.sum() >= 8:
            fv_ = f[m].rank().values; rv_ = r[m].rank().values
            if fv_.std() > 0 and rv_.std() > 0:
                ics.append(np.corrcoef(fv_, rv_)[0,1])
    ics = np.array(ics)
    if len(ics) < 20:
        return {'IC': 0.0, 'ICIR': 0.0, 'n': len(ics), 'hit': 0.0, 'cov': 0.0}
    hit = float((ics > 0).mean())
    cov = float(fv.notna().mean().mean())
    return {'IC': float(ics.mean()),
            'ICIR': float(ics.mean()/ics.std()*np.sqrt(len(ics))) if ics.std()>0 else 0.0,
            'n': len(ics), 'hit': hit, 'cov': cov}

def turnover(fv):
    fv = fv.reindex(ret_idx)
    s = np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0)
    return float((s.diff()!=0).mean().mean())

def report(name, fv):
    ic = compute_ic(fv)
    print(f"{name}: IC={ic['IC']:.4f} ICIR={ic['ICIR']:.4f} n={ic['n']} "
          f"hit={ic['hit']:.3f} cov={ic['cov']:.3f} tov={turnover(fv):.3f}")

# A: Volume trend ratio 5/20
vtrend = (vol.rolling(5).mean() / vol.rolling(20).mean())
report("A vol_trend_5_20", vtrend)

# B: Amihud illiquidity 20d (|ret|/volume), log-scale
vchg = rets.abs() / vol.replace(0, np.nan)
amihud = np.log1p(vchg.rolling(20).mean().replace(0, np.nan))
report("B amihud_illiq_20", amihud)

# C: Volume z-score vs 20d mean
vz = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
report("C volume_z_20", vz)

# D: Realized range (high-low)/close 20d
rng = ((high-low)/close).rolling(20).mean()
report("D realized_range_20", rng)

# E: Intraday close location (close-low)/(high-low) 10d
loc = ((close-low)/(high-low).replace(0,np.nan)).rolling(10).mean()
report("E intraday_up_pos_10", loc)

# F: Drawdown depth 60d (distance from rolling max)
maxdd = pd.DataFrame({a: close[a]/close[a].rolling(60).max()-1 for a in ASSETS})
report("F drawdown_depth_60", maxdd)