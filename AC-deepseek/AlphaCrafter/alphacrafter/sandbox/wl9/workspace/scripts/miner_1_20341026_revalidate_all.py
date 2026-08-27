"""Re-validate all effective factors through 2034-10-25 (current date 2034-10-26)."""
import json, os
import pandas as pd
import numpy as np
from pathlib import Path

VISIBLE_END = '2034-10-25'
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

closes = {}
for a in ASSETS:
    f = STOCK_DIR / f'{a}.csv'
    df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
    df = df[df['date'] <= VISIBLE_END]
    closes[a] = df.set_index('date')['close'].astype(float)

rets = pd.DataFrame({a: closes[a].pct_change() for a in ASSETS}).dropna()
rets = rets[rets.index >= '2020-03-01']
print(f"Panel: {rets.shape[0]} dates x {rets.shape[1]} assets from {rets.index[0]:%Y-%m-%d} to {rets.index[-1]:%Y-%m-%d}")

# OHLC needed for low/high-based factors
hl = {}
for a in ASSETS:
    df = pd.read_csv(STOCK_DIR / f'{a}.csv', parse_dates=['date']).sort_values('date')
    df = df[df['date'] <= VISIBLE_END].set_index('date')
    hl[a] = df[['open','close','high','low','volume']].astype(float)

vix = pd.read_csv(INDEX_DIR / 'VIX.csv', parse_dates=['date'])
vix = vix[vix['date'] <= VISIBLE_END].set_index('date')['close'].astype(float)

macros = {}
for m in ['DXY','USDCNY','USDJPY','EURUSD']:
    df = pd.read_csv(INDEX_DIR / f'{m}.csv', parse_dates=['date'])
    df = df[df['date'] <= VISIBLE_END].set_index('date')['close'].astype(float)
    macros[m] = df

def factor_frame(build_fn):
    """build_fn(a) returns a pd.Series indexed by date."""
    out = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
    for a in ASSETS:
        s = build_fn(a)
        s = s.reindex(rets.index)
        out[a] = s
    return out

def compute_ic(factor_vals, forward_rets):
    common_dates = sorted(set(factor_vals.index) & set(forward_rets.index))
    ics = []
    for d in common_dates:
        f = factor_vals.loc[d]
        r = forward_rets.loc[d]
        valid = f.notna() & r.notna()
        if valid.sum() >= 8:
            f_vals = f[valid].rank().values
            r_vals = r[valid].rank().values
            if np.std(f_vals) > 0 and np.std(r_vals) > 0:
                ic = np.corrcoef(f_vals, r_vals)[0,1]
                ics.append(ic)
    ics = np.array(ics)
    if len(ics) < 20:
        return {'IC': 0.0, 'ICIR': 0.0, 'n_dates': len(ics)}
    return {
        'IC': float(np.mean(ics)),
        'ICIR': float(np.mean(ics) / np.std(ics) * np.sqrt(len(ics))) if np.std(ics) > 0 else 0.0,
        'n_dates': len(ics)
    }

fwd_10d = rets.shift(-10).rolling(10).mean()
vix_rets = vix.pct_change()
dxy_rets = macros['DXY'].pct_change()
usdcny_rets = macros['USDCNY'].pct_change()

factors = {}

# momentum family
factors['mom_10d_skip5'] = factor_frame(lambda a: closes[a].pct_change(15))
factors['mom_120d_skip5'] = factor_frame(lambda a: closes[a].pct_change(125))

# VIX beta
def vix_beta_fn(a, win):
    joint = pd.concat([rets[a].rename('a'), vix_rets.rename('v')], axis=1).dropna()
    return joint['a'].rolling(win).cov(joint['v']) / joint['v'].rolling(win).var()
factors['beta_VIX_60'] = factor_frame(lambda a: vix_beta_fn(a, 60))
factors['vix_beta_cond_60x20'] = factor_frame(lambda a: vix_beta_fn(a,60) - vix_beta_fn(a,20))

# vix_roc_20d
vix_roc = vix.pct_change(20)
def vroc_fn(a):
    safe = ['XAU','US10Y','CN10Y']
    return (vix_roc if a in safe else -vix_roc).reindex(rets.index)
factors['vix_roc_20d'] = factor_frame(vroc_fn)

# mom_10_vixreg
def mom10vix_fn(a):
    c = closes[a]
    return (c.pct_change(5) * np.sign(vix.pct_change(10).shift(5))).reindex(rets.index)
factors['mom_10_vixreg'] = factor_frame(mom10vix_fn)

# ac1_120d
def ac1_fn(a):
    return rets[a].rolling(120).apply(lambda x: x.autocorr(1) if len(x)>=30 else np.nan, raw=False)
factors['