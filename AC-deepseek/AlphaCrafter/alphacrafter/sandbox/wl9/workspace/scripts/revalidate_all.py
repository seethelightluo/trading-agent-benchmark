"""Re-validate all existing factors on data up to 2027-07-28"""
import json, os, sys
import pandas as pd
import numpy as np
from pathlib import Path

VISIBLE_END = '2027-07-28'
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

vix = pd.read_csv(INDEX_DIR / 'VIX.csv', parse_dates=['date'])
vix = vix[vix['date'] <= VISIBLE_END].set_index('date')['close'].astype(float)

macros = {}
for m in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    f = INDEX_DIR / f'{m}.csv'
    df = pd.read_csv(f, parse_dates=['date'])
    df = df[df['date'] <= VISIBLE_END].set_index('date')['close'].astype(float)
    macros[m] = df

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

results = {}

# 1. mom_10d_skip5
print("\n=== mom_10d_skip5 ===")
mom10 = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    mom10[a] = closes[a].pct_change(15).reindex(rets.index)
ic = compute_ic(mom10, fwd_10d)
print(f"IC={ic['IC']:.6f}, ICIR={ic['ICIR']:.6f}, n={ic['n_dates']}")
results['mom_10d_skip5'] = ic

# 2. mom_120d_skip5
print("\n=== mom_120d_skip5 ===")
mom120 = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    mom120[a] = closes[a].pct_change(125).reindex(rets.index)
ic = compute_ic(mom120, fwd_10d)
print(f"IC={ic['IC']:.6f}, ICIR={ic['ICIR']:.6f}, n={ic['n_dates']}")
results['mom_120d_skip5'] = ic

# 3. vix_beta_cond_60x20
print("\n=== vix_beta_cond_60x20 ===")
vbeta = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    joint = pd.concat([rets[a].rename('a'), vix_rets.rename('v')], axis=1).dropna()
    vbeta[a] = (joint['a'].rolling(60).cov(joint['v']) / joint['v'].rolling(60).var() -
                joint['a'].rolling(20).cov(joint['v']) / joint['v'].rolling(20).var())
ic = compute_ic(vbeta, fwd_10d)
print(f"IC={ic['IC']:.6f}, ICIR={ic['ICIR']:.6f}, n={ic['n_dates']}")
results['vix_beta_cond_60x20'] = ic

# 4. beta_VIX_60
print("\n=== beta_VIX_60 ===")
b60 = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    joint = pd.concat([rets[a].rename('a'), vix_rets.rename('v')], axis=1).dropna()
    b60[a] = joint['a'].rolling(60).cov(joint['v']) / joint['v'].rolling(60).var()
ic = compute_ic(b60, fwd_10d)
print(f"IC={ic['IC']:.6f}, ICIR={ic['ICIR']:.6f}, n={ic['n_dates']}")
results['beta_VIX_60'] = ic

# 5. vix_roc_20d
print("\n=== vix_roc_20d ===")
vix_roc = vix.pct_change(20)
vroc = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
safe = ['XAU','US10Y','CN10Y']
for a in ASSETS:
    vroc[a] = vix_roc.reindex(rets.index) if a in safe else -vix_roc.reindex(rets.index)
ic = compute_ic(vroc, fwd_10d)
print(f"IC={ic['IC']:.6f}, ICIR={ic['ICIR']:.6f}, n={ic['n_dates']}")
results['vix_roc_20d'] = ic

# 6. ac1_120d
print("\n=== ac1_120d ===")
ac1 = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    ac1[a] = rets[a].rolling(120).apply(lambda x: x.autocorr(1) if len(x)>=30 else np.nan, raw=False)
ic = compute_ic(ac1, fwd_10d)
print(f"IC={ic['IC']:.6f}, ICIR={ic['ICIR']:.6f}, n={ic['n_dates']}")
results['ac1_120d'] = ic

# 7. bb_width_20d
print("\n=== bb_width_20d ===")
bb = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    c = closes[a]
    bb[a] = ((c - c.rolling(20).mean()) / c.rolling(20).std()).reindex(rets.index)
ic = compute_ic(bb, fwd_10d)
print(f"IC={ic['IC']:.6f}, ICIR={ic['ICIR']:.6f}, n={ic['n_dates']}")
results['bb_width_20d'] = ic

# 8. kaufman_eff_20d
print("\n=== kaufman_eff_20d ===")
kf = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    c = closes[a]
    kf[a] = (c.diff(20).abs() / c.diff().abs().rolling(20).sum()).reindex(rets.index)
ic = compute_ic(kf, fwd_10d)
print(f"IC={ic['IC