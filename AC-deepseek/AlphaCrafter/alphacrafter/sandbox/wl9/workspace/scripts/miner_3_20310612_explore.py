"""miner_3 cycle 2031-06-12: explore candidate factor ideas on data up to 2031-06-12 (no lookahead)."""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = pd.Timestamp('2031-06-12')
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

closes = {}
vols = {}
for a in ASSETS:
    f = STOCK_DIR / f'{a}.csv'
    if not f.exists():
        f = INDEX_DIR / f'{a}.csv'
    df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
    df = df[df['date'] <= VISIBLE_END].set_index('date')
    closes[a] = df['close'].astype(float)
    vols[a] = df['volume'].astype(float) if 'volume' in df else pd.Series(np.nan, index=df.index)

rets = pd.DataFrame({a: closes[a].pct_change() for a in ASSETS}).dropna()
rets = rets[rets.index >= '2020-03-01']
print(f"Panel: {rets.shape[0]} dates x {rets.shape[1]} assets from {rets.index[0]:%Y-%m-%d} to {rets.index[-1]:%Y-%m-%d}")

vix = pd.read_csv(INDEX_DIR/'VIX.csv', parse_dates=['date'])
vix = vix[vix['date']<=VISIBLE_END].set_index('date')['close'].astype(float)

def compute_ic(factor_vals, forward_rets, min_dates=60):
    common = sorted(set(factor_vals.index) & set(forward_rets.index))
    ics = []
    ndates_ok = 0
    for d in common:
        f = factor_vals.loc[d]; r = forward_rets.loc[d]
        valid = f.notna() & r.notna()
        if valid.sum() >= 8:
            ndates_ok += 1
            fv = f[valid].rank().values; rv = r[valid].rank().values
            if np.std(fv)>0 and np.std(rv)>0:
                ics.append(np.corrcoef(fv,rv)[0,1])
    ics = np.array(ics)
    if len(ics) < min_dates:
        return {'IC':0.0,'ICIR':0.0,'n':len(ics),'dates_ok':ndates_ok}
    mu = ics.mean(); sd = ics.std()
    icir = mu/sd*np.sqrt(len(ics)) if sd>0 else 0.0
    return {'IC':float(mu),'ICIR':float(icir),'n':len(ics),'dates_ok':ndates_ok}

def coverage(fv):
    cov = float(fv.notna().sum().sum())/(fv.shape[0]*fv.shape[1])
    dates_ok = float((fv.notna().sum(axis=1)>=8).mean())
    return cov, dates_ok

def turnover(fv):
    s = np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0)
    diff = (s.diff()!=0).mean().mean()
    return float(diff)

def apply_rolling_asset(func, window):
    out={}
    for a in ASSETS:
        out[a] = rets[a].rolling(window).apply(func, raw=True).reindex(rets.index)
    return pd.DataFrame(out)

fwd = rets.shift(-10).rolling(10).mean()
print("forward horizon: 10 trading days")

# A: 20d close position within 20d high-low range (RSI-like)
pos20 = pd.DataFrame({a: ((closes[a]-closes[a].rolling(20).min())/(closes[a].rolling(20).max()-closes[a].rolling(20).min())).reindex(rets.index) for a in ASSETS})
icA = compute_ic(pos20, fwd)
print(f"A close_pos_20d: IC={icA['IC']:.4f} ICIR={icA['ICIR']:.4f} n={icA['n']} cov={coverage(pos20)[0]:.3f} tov={turnover(pos20):.3f}")

# B: momentum acceleration mom10 - mom60 demeaned
mom10 = pd.DataFrame({a: closes[a].pct_change(10).reindex(rets.index) for a in ASSETS})
mom60 = pd.DataFrame({a: closes[a].pct_change(60).reindex(rets.index) for a in ASSETS})
accel = mom10 - mom60
accel_dm = accel.sub(accel.mean(axis=1), axis=0)
icB = compute_ic(accel_dm, fwd)
print(f"B accel_10_60_dm: IC={icB['IC']:.4f} ICIR={icB['ICIR']:.4f} n={icB['n']} cov={coverage(accel_dm)[0]:.2f} tov={turnover(accel_dm):.3f}")

# C: risk-adjusted trend 20d
sharpe20 = pd.DataFrame({a: (rets[a].rolling(20).mean()/rets[a].rolling(20).std()).reindex(rets.index) for a in ASSETS})
icC = compute_ic(sharpe20, fwd)
print(f"C sharpe_20d: IC={icC['IC']:.4f} ICIR={icC['ICIR']:.4f} n={icC['n']} cov={coverage(sharpe20)[0]:.2f} tov={turnover(sharpe20):.3f}")

# D: left partial moment 10d
lpm10 = pd.DataFrame({a: rets[a].rolling(10).apply(lambda x: -x[x<0].sum()/max(len(x[x<0]),1) if (x<0).any() else 0.0, raw=True).reindex(rets.index) for a in ASSETS})
icD = compute_ic(lpm10, fwd)
print(f"D lpm10: IC={icD['IC']:.4f} ICIR={icD['ICIR']:.4f} n={icD['n']} cov={coverage(lpm10)[0]:.2f} tov={turnover(lpm10):.3f}")

# E: max winning day streak over 10d
def maxwin(x):
    s=0;m=0
    for v in x:
        s = s+1 if v>0 else 0
        m=max(m,s)
    return m
win10 = pd.DataFrame({a: rets[a].rolling(10).apply(maxwin, raw=True).reindex(rets.index) for a in ASSETS})
icE = compute_ic(win10, fwd)
print(f"E maxwin_streak_10: IC={icE['IC']:.4f} ICIR={icE['ICIR']:.4f} n={icE['n']} cov={coverage(win10)[0]:.2f} tov={turnover(win10):.3f}")