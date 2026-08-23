"""miner_1 re-validation of full effective factor library through 2029-05-02."""
import json, os, sys, zlib, base64, hashlib, io
import pandas as pd
import numpy as np
from pathlib import Path

VISIBLE_END = '2029-05-30'
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

closes = {}
for a in ASSETS:
    f = STOCK_DIR / f'{a}.csv'
    df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
    df = df[df['date'] <= VISIBLE_END]
    closes[a] = df.set_index('date')['close'].astype(float)

P = pd.DataFrame(closes)
rets = P.pct_change()
print(f"Close panel {P.shape} from {P.index.min().date()} to {P.index.max().date()}")

vix = pd.read_csv(INDEX_DIR / 'VIX.csv', parse_dates=['date'])
vix = vix[vix['date'] <= VISIBLE_END].set_index('date')['close'].astype(float)
macros = {}
for m in ['DXY','USDCNY','USDJPY','EURUSD']:
    f = INDEX_DIR / f'{m}.csv'
    df = pd.read_csv(f, parse_dates=['date'])
    df = df[df['date'] <= VISIBLE_END].set_index('date')['close'].astype(float)
    macros[m] = df

vix_rets = vix.pct_change()
dxy_rets = macros['DXY'].pct_change()
usdjpy_rets = macros['USDJPY'].pct_change()
cny_rets = macros['USDCNY'].pct_change()

def fwd_ret(h):
    return P.shift(-(h-1)) / P - 1.0

def ic_stats(sig, ret, min_assets=8):
    dates, ics = [], []
    for dt, row in sig.iterrows():
        s = row.dropna(); rr = ret.loc[dt].dropna()
        idx = s.index.intersection(rr.index)
        if len(idx) < min_assets: continue
        ic = np.corrcoef(s[idx], rr[idx])[0,1]
        if not np.isnan(ic): dates.append(dt); ics.append(ic)
    ics = np.array(ics); dates = pd.DatetimeIndex(dates)
    m = ics.mean(); sd = ics.std(ddof=1) if len(ics)>1 else np.nan
    icir = m/sd if sd and sd==sd and sd>0 else np.nan
    hit = (np.sign(ics)==np.sign(m)).mean() if len(ics) else np.nan
    return dict(n=len(ics), first=dates.min().date() if len(dates) else None,
                last=dates.max().date() if len(dates) else None, ic=m, icir=icir, hit=hit)

def coverage_turnover(sig, min_assets=8):
    valid = sig.notna()
    cov_date = float((valid.sum(axis=1) >= min_assets).mean())
    cov_ad = float(valid.mean().mean())
    ranks = sig.rank(axis=1, pct=True)
    turn = float(ranks.diff().abs().mean().mean())
    return cov_date, cov_ad, turn

fwd10 = fwd_ret(10)

def report(fid, sig, res, direction):
    ic10 = res['ic']
    print(f"{fid}: full IC={'%+.4f'%ic10['ic']} ICIR={'%+.4f'%ic10['icir']} n={ic10['n']} hit={ic10['hit']:.3f} cov_d={res['cov_d']:.3f} turn={res['turn']:.3f} dir={direction:+d} PASS={abs(ic10['ic'])>=0.0070 and abs(ic10['icir'])>=0.0840}")

factors = {}

# 1. beta_VIX_60
sig = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    j = pd.concat([rets[a].rename('a'), vix_rets.rename('v')], axis=1).dropna()
    sig[a] = j['a'].rolling(60).cov(j['v']) / j['v'].rolling(60).var()
sig = sig.shift(1)
res = dict(ic=ic_stats(sig, fwd10))
res['cov_d'], res['cov_ad'], res['turn'] = coverage_turnover(sig)
report('beta_VIX_60', sig, res, -1)
factors['beta_VIX_60'] = dict(sig=sig, **res, direction=-1)

# 2. kaufman_eff_20d
sig = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    c = P[a]
    sig[a] = c.diff(20).abs() / c.diff().abs().rolling(20).sum()
sig = sig.shift(1)
res = dict(ic=ic_stats(sig, fwd10))
res['cov_d'], res['cov_ad'], res['turn'] = coverage_turnover(sig)
report('kaufman_eff_20d', sig, res, 1)
factors['kaufman_eff_20d'] = dict(sig=sig, **res, direction=1)

# 3. mom_120d_skip5
sig = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    sig[a] = P[a].pct_change(125)
sig = sig.shift(1)
res = dict(ic=ic_stats(sig, fwd10))
res['cov_d'], res['cov_ad'], res['turn'] = coverage_turnover(sig)
report('mom_120d_skip5', sig, res, 1)
factors['mom_120d_skip5'] = dict(sig=sig, **res, direction=1)

# 4. bb_width_20d
sig = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    c = P[a]
    sig[a] = (c - c.rolling(20).mean()) / c.rolling(20).std()
sig = sig.shift(1)
res = dict(ic=ic_stats(sig, fwd10))
res['cov_d'], res['cov_ad'], res['turn'] = coverage_turnover(sig)
report('bb_width_20d', sig, res, 1)
factors['bb_width_20d'] = dict(sig=sig, **res, direction=1)

# 5. cny_beta_60
sig = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    j = pd.concat([rets[a].rename('a'), cny_rets.rename('m')], axis=1).dropna()
    sig[a] = j['a'].rolling(60).cov(j['m']) / j['m'].rolling(60).var()
sig = sig.shift(1)
res = dict(ic=ic_stats(sig, fwd10))
res['cov_d'], res['cov_ad'], res['turn'] = coverage_turnover(sig)
report('cny_beta_60', sig, res, 1)
factors['cny_beta_60'] = dict(sig=sig, **res, direction=1)

print('PART1_DONE')