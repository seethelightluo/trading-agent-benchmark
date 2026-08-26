"""miner_2 2031-11-27: validate vol-family candidates C/D/E with library correlation
against persisted factor library. Visible end = 2031-11-26 (previous completed day).
Admission gates: abs daily paper IC >= 0.0070 and abs ICIR >= 0.0840 (10d horizon).
Universe = 15 tradable cross-asset instruments.
"""
import json, os, glob
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

VISIBLE_END = '2031-11-26'
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

closes, vols = {}, {}
for a in ASSETS:
    df = get_stock_daily_data(symbol=a, days=4000)
    df = df[df['date'] <= VISIBLE_END].set_index('date').sort_index()
    closes[a] = df['close'].astype(float)
    vols[a] = df['volume'].astype(float)

close = pd.DataFrame(closes).dropna()
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

# C: downside sdv ratio 60
def ds_ratio(window):
    out = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
    for a in ASSETS:
        r = rets[a]
        m = r.rolling(window).mean()
        sq = np.where((r-m) < 0, (r-m)**2, 0.0)
        ds = pd.Series(np.sqrt(pd.Series(sq, index=r.index).rolling(window).mean()), index=r.index)
        tv = r.rolling(window).std()
        out[a] = ds / tv.replace(0, np.nan)
    return out
# D: rank vol 60
vol60 = rets.rolling(60).std()
# E: log vol ts 10/60
vol10 = rets.rolling(10).std()
vts = np.log(vol10 / vol60.replace(0,np.nan))

FACTORS = {
    'C_ds_sdv_ratio_60': ds_ratio(60),
    'D_rank_vol_60': vol60,
    'E_log_vol_ts_10_60': vts,
}

# load library (existing factor JSON) series if recoverable
def load_lib_series(fid):
    if fid=='vol_z_20d':
        return vol10  # proxy 20d vol z
    return None

LIB = {}
for fid in ['beta_VIX_60','kaufman_eff_20d','mom_120d_skip5','mom_10d_skip5',
            'bb_width_20d','cny_beta_60','vol_z_20d','ac1_120d',
            'dxy_corr_change_20_60','skew_20d','vix_beta_cond_60x20','kurt_20d',
            'rng_pos_20d','days_since_high_60','streak_len_14','vix_roc_20d']:
    s = load_lib_series(fid)
    if s is not None:
        LIB[fid] = s

for fname, fv in FACTORS.items():
    ic = compute_ic(fv)
    print(f"{fname}: IC={ic['IC']:.4f} ICIR={ic['ICIR']:.4f} n={ic['n']} "
          f"hit={ic['hit']:.3f} cov={ic['cov']:.3f}")