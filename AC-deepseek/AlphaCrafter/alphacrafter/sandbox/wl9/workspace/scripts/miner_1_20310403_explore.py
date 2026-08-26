"""miner_1 cycle 2031-04-03: explore candidate factor ideas (data up to 2031-04-03)."""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = pd.Timestamp('2031-04-03')
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

closes = {}; vols = {}; highs = {}; lows = {}
for a in ASSETS:
    f = STOCK_DIR / f'{a}.csv'
    if not f.exists():
        f = INDEX_DIR / f'{a}.csv'
    df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
    df = df[df['date'] <= VISIBLE_END].set_index('date')
    closes[a] = df['close'].astype(float)
    vols[a] = df['volume'].astype(float) if 'volume' in df else pd.Series(np.nan, index=df.index)
    highs[a] = df['high'].astype(float)
    lows[a] = df['low'].astype(float)

rets = pd.DataFrame({a: closes[a].pct_change() for a in ASSETS}).dropna()
rets = rets[rets.index >= '2020-03-01']
print(f"Panel: {rets.shape[0]} dates x {rets.shape[1]} assets from {rets.index[0]:%Y-%m-%d} to {rets.index[-1]:%Y-%m-%d}")

fwd10 = rets.shift(-10).rolling(10).mean()

def compute_ic(factor_vals, fwd):
    if isinstance(factor_vals, pd.Series):
        factor_vals = factor_vals.to_frame()
    common = sorted(set(factor_vals.index) & set(fwd.index))
    ics = []
    for d in common:
        f = factor_vals.loc[d]; r = fwd.loc[d]
        valid = f.notna() & r.notna()
        if valid.sum() >= 8:
            fv = f[valid].rank().values; rv = r[valid].rank().values
            if np.std(fv) > 0 and np.std(rv) > 0:
                ics.append(np.corrcoef(fv, rv)[0, 1])
    ics = np.array(ics)
    if len(ics) < 30:
        return {'IC': 0.0, 'ICIR': 0.0, 'n': int(len(ics))}
    mu = ics.mean(); sd = ics.std()
    return {'IC': float(mu), 'ICIR': float(mu/sd*np.sqrt(len(ics))) if sd > 0 else 0.0, 'n': int(len(ics))}

def covr(factor_vals):
    if isinstance(factor_vals, pd.Series):
        factor_vals = factor_vals.to_frame()
    return float((factor_vals.notna().sum(axis=1) >= 8).mean())

# A: 60d drawdown depth (deviation from trailing 60d high), avg 10d
dd = pd.DataFrame({a: (closes[a]/closes[a].rolling(60).max()-1).reindex(rets.index) for a in ASSETS})
dd_avg = dd.rolling(10).mean()
icA = compute_ic(dd_avg, fwd10)
print(f"A 60d_drawdown_avg10 : IC={icA['IC']:.6f} ICIR={icA['ICIR']:.6f} n={icA['n']} cov={covr(dd_avg):.3f}")

# B: cross-sectional demeaned 90d momentum
mom90 = pd.DataFrame({a: closes[a].pct_change(90).reindex(rets.index) for a in ASSETS})
rel90 = mom90.sub(mom90.mean(axis=1), axis=0)
icB = compute_ic(rel90, fwd10)
print(f"B rel_mom90_demean   : IC={icB['IC']:.6f} ICIR={icB['ICIR']:.6f} n={icB['n']} cov={covr(rel90):.3f}")

# C: 20d mean intraday range (high-low)/close
rng20 = pd.DataFrame({a: ((highs[a]-lows[a])/closes[a]).rolling(20).mean().reindex(rets.index) for a in ASSETS})
icC = compute_ic(rng20, fwd10)
print(f"C 20d_intraday_range : IC={icC['IC']:.6f} ICIR={icC['ICIR']:.6f} n={icC['n']} cov={covr(rng20):.3f}")

# D: beta to USDCNY change (macro China sensitivity)
usdcny = pd.read_csv(INDEX_DIR/'USDCNY.csv', parse_dates=['date'])
usdcny = usdcny[usdcny['date']<=VISIBLE_END].set_index('date')['close'].astype(float)
uc = usdcny.pct_change().reindex(rets.index)
cny_beta = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    j = pd.concat([rets[a].rename('a'), uc.rename('u')], axis=1).dropna()
    if len(j) >= 60:
        cny_beta[a] = j['u'].rolling(60).cov(j['a'])/j['u'].rolling(60).var()
icD = compute_ic(cny_beta, fwd10)
print(f"D cny_beta_60        : IC={icD['IC']:.6f} ICIR={icD['ICIR']:.6f} n={icD['n']} cov={covr(cny_beta):.3f}")

# E: deviation from 250d MA (cycle position)
ma250 = pd.DataFrame({a: closes[a].rolling(250).mean().reindex(rets.index) for a in ASSETS})
lvl = pd.DataFrame({a: (closes[a].reindex(rets.index)-ma250[a])/ma250[a] for a in ASSETS})
icE = compute_ic(lvl, fwd10)
print(f"E 250d_ma_deviation  : IC={icE['IC']:.6f} ICIR={icE['ICIR']:.6f} n={icE['n']} cov={covr(lvl):.3f}")

# F: downside/upside asymmetry 20d (negative -> prefer upside days)
up = rets.clip(lower=0).rolling(20).sum()
dn = rets.clip(upper=0).rolling(20).sum()
asym = dn / (up.abs() + 1e-9)
icF = compute_ic(-asym, fwd10)
print(f"F updown_ratio_20d(neg): IC={icF['IC']:.6f} ICIR={icF['ICIR']:.6f} n={icF['n']} cov={covr(asym):.3f}")

# G: volume trend relative to price momentum (liquidity confirming)
lvol = pd.DataFrame({a: np.log(vols[a].clip(lower=1)).reindex(rets.index) for a in ASSETS})
vol_trend = lvol.rolling(20).mean() - lvol.rolling(60).mean()
icG = compute_ic(vol_trend, fwd10)
print(f"G vol_trend_20_60    : IC={icG['IC']:.6f} ICIR={icG['ICIR']:.6f} n={icG['n']} cov={covr(vol_trend):.3f}")