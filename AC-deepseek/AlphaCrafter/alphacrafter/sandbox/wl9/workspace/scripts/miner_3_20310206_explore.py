"""miner_3 cycle 2031-02-06: explore candidate factor ideas on data up to current date (up to 2031-02-06)."""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = pd.Timestamp('2031-02-06')
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
macros = {}
for m in ['DXY','USDCNY','USDJPY','EURUSD']:
    mm = pd.read_csv(INDEX_DIR/f'{m}.csv', parse_dates=['date'])
    macros[m] = mm[mm['date']<=VISIBLE_END].set_index('date')['close'].astype(float)

def compute_ic(factor_vals, forward_rets):
    common = sorted(set(factor_vals.index) & set(forward_rets.index))
    ics = []
    for d in common:
        f = factor_vals.loc[d]; r = forward_rets.loc[d]
        valid = f.notna() & r.notna()
        if valid.sum() >= 8:
            fv = f[valid].rank().values; rv = r[valid].rank().values
            if np.std(fv)>0 and np.std(rv)>0:
                ics.append(np.corrcoef(fv,rv)[0,1])
    ics = np.array(ics)
    if len(ics)<20:
        return {'IC':0.0,'ICIR':0.0,'n':len(ics)}
    mu = ics.mean(); sd = ics.std()
    return {'IC':float(mu),'ICIR':float(mu/sd*np.sqrt(len(ics))) if sd>0 else 0.0,'n':len(ics)}

fwd = rets.shift(-10).rolling(10).mean()

def coverage(factor_vals):
    tot = factor_vals.notna().sum().sum()
    n_dates_ge8 = sum((factor_vals.notna().sum(axis=1)>=8).sum() for _ in [0])
    dates_ok = (factor_vals.notna().sum(axis=1)>=8).mean()
    return tot/(factor_vals.shape[0]*factor_vals.shape[1]), dates_ok

# Candidate A: cross-sectional demeaned momentum 20d (relative momentum)
mom20 = pd.DataFrame({a: closes[a].pct_change(20).reindex(rets.index) for a in ASSETS})
rel_mom20 = mom20.sub(mom20.mean(axis=1), axis=0)
icA = compute_ic(rel_mom20, fwd)
covA = compute_ic(rel_mom20, fwd)
print(f"\nA rel_mom20 (demeaned 20d mom): IC={icA['IC']:.6f} ICIR={icA['ICIR']:.6f} n={icA['n']}")

# Candidate B: 20d cumulative signed path (strategy: momentum with overnight gaps)
path = {}
for a in ASSETS:
    r = rets[a]
    path[a] = (r.rolling(20).apply(lambda x: (x>0).sum()/len(x), raw=True)).reindex(rets.index)
skip = pd.DataFrame(path)
icB = compute_ic(skip, fwd)
print(f"B 20d winning-day ratio: IC={icB['IC']:.6f} ICIR={icB['ICIR']:.6f} n={icB['n']}")

# Candidate C: volume z-score * 20d momentum (volume confirmation)
lvol = pd.DataFrame({a: np.log(vols[a].clip(lower=1)).reindex(rets.index) for a in ASSETS})
vol_z = (lvol - lvol.rolling(20).mean()) / lvol.rolling(20).std()
vmom = mom20 * vol_z
icC = compute_ic(vmom, fwd)
print(f"C vol-confirmed mom20: IC={icC['IC']:.6f} ICIR={icC['ICIR']:.6f} n={icC['n']}")

# Candidate D: price range (high-low)/close 5d (short-term range, momentum-reversal)
hl = pd.DataFrame({a: ((df['high']-df['low'])/df['close']).reindex(rets.index) for a in ASSETS})
hl5 = hl.rolling(5).mean()
icD = compute_ic(hl5, fwd)
print(f"D 5d intraday range: IC={icD['IC']:.6f} ICIR={icD['ICIR']:.6f} n={icD['n']}")

# Candidate E: 60d vol / 20d vol ratio (vol regime shift)
v20 = rets.rolling(20).std()
v60 = rets.rolling(60).std()
vreg = (v60 / v20)
icE = compute_ic(vreg, fwd)
print(f"E 60d/20d vol ratio: IC={icE['IC']:.6f} ICIR={icE['ICIR']:.6f} n={icE['n']}")

# Candidate F: beta to DXY change (macro regime tilt)
dxy_ret = macros['DXY'].pct_change().reindex(rets.index)
dbeta = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    j = pd.concat([rets[a].rename('a'), dxy_ret.rename('d')], axis=1).dropna()
    if len(j)>=60:
        dbeta[a] = j['d'].rolling(60).cov(j['a'])/j['d'].rolling(60).var()
icF = compute_ic(dbeta, fwd)
print(f"F beta to DXY 60d: IC={icF['IC']:.6f} ICIR={icF['ICIR']:.6f} n={icF['n']}")