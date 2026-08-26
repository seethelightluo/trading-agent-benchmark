"""miner_2 2033-01-06: validate candidate 'bull_grip_20d'.
Bull grip = mean over 20d of (close-low)/(high-low), a measure of persistent
closing near highs (sustained intraday buying pressure). Direction +1.
Visible through 2033-01-05. No lookahead. 15-asset cross-asset universe.
Admission gates: abs daily paper IC >= 0.0070 and abs ICIR >= 0.084.
"""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = '2033-01-05'
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(assets, end):
    closes, highs, lows = {}, {}, {}
    for a in assets:
        f = STOCK_DIR / f'{a}.csv'
        if not f.exists():
            f = INDEX_DIR / f'{a}.csv'
        df = pd.read_csv(f, parse_dates=['date'])
        df = df[df['date'] <= end].sort_values('date').set_index('date')
        closes[a] = df['close'].astype(float)
        highs[a] = df['high'].astype(float)
        lows[a] = df['low'].astype(float)
    return closes, highs, lows

closes, highs, lows = load(ASSETS, VISIBLE_END)
close = pd.DataFrame(closes).dropna()
high = pd.DataFrame(highs).reindex(close.index)
low = pd.DataFrame(lows).reindex(close.index)
rets = close.pct_change().dropna()
fwd5  = rets.shift(-5).rolling(5).mean()
fwd10 = rets.shift(-10).rolling(10).mean()
fwd20 = rets.shift(-20).rolling(20).mean()
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, "
      f"{close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}", flush=True)

# daily candle position
rng = (high - low).replace(0, np.nan)
body_pos = (close - low) / rng
cand = body_pos.rolling(20).mean().reindex(fwd10.index)

def compute_ic(fv, fwd, start=None, min_dates=30):
    fv = fv.reindex(fwd.index); idx = fwd.index
    if start: idx = idx[idx >= pd.Timestamp(start)]
    ics = []
    for d in idx:
        f = fv.loc[d]; r = fwd.loc[d]; m = f.notna() & r.notna()
        if m.sum() >= 8:
            a = f[m].rank().values; b = r[m].rank().values
            if a.std() > 0 and b.std() > 0:
                ics.append(np.corrcoef(a, b)[0, 1])
    ics = np.array(ics)
    if len(ics) < min_dates:
        return dict(ic=0.0, icir=0.0, n=len(ics), hit=0.0)
    mu = ics.mean(); sd = ics.std()
    icir = mu/sd*np.sqrt(len(ics)) if sd > 0 else 0
    return dict(ic=float(mu), icir=float(icir), n=len(ics), hit=float((ics>0).mean()))

print("\n== bull_grip_20d decay (full sample) ==")
for h, fd in [('5', fwd5), ('10', fwd10), ('20', fwd20)]:
    r = compute_ic(cand, fd)
    print(f"  {h}d: IC={r['ic']:.4f} ICIR={r['icir']:.4f} n={r['n']} hit={r['hit']:.3f}")

print("\n== sub-regime IC (10d horizon) ==")
for s in ['2020-01-01','2022-01-01','2024-01-01','2026-01-01','2028-01-01',
          '2030-01-01','2031-06-01','2032-01-01','2032-06-01','2032-10-01']:
    r = compute_ic(cand, fwd10, start=s)
    print(f"  from {s}: IC={r['ic']:.4f} ICIR={r['icir']:.4f} n={r['n']} hit={r['hit']:.3f}")

cov = float(cand.notna().mean().mean())
def turnover(fv):
    s = np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0)
    return float((s.diff() != 0).mean().mean())
print(f"\ncoverage={cov:.3f} turnover_10d_rank={turnover(cand):.3f}")

# opposite direction check (negative of factor) for robustness reporting
r = compute_ic(-cand, fwd10)
print(f"negated 10d: IC={r['ic']:.4f} ICIR={r['icir']:.4f}")
