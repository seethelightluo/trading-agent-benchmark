"""
miner1_20351108_exp01_on_minus_id_full.py
Full validation of overnight-minus-intraday return spread factor (on_minus_id_20).
Includes: IC by horizon, OOS regime-split, decay analysis, coverage, turnover,
library correlation, signal persistence.
"""
import json, base64, zlib, io, csv, hashlib, numpy as np, pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

P = Path('../persistent'); SD = P/'stock_data'; ID = P/'index_data'
ASSETS = ['000300.SH','000688.SH','SPX','HSI','N225','SX5E','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
MACRO = ['DXY','USDCNY','USDJPY','EURUSD','VIX']
LIBS = ['ac1_120d','bb_width_20d','beta_VIX_60','cny_beta_60','days_since_high_60',
        'dxy_corr_change_20_60','kaufman_eff_20d','kurt_20d','mom_10_vixreg',
        'mom_10d_skip5','mom_120d_skip5','rng_pos_20d','skew_20d','streak_len_14',
        'vix_beta_cond_60x20','vix_roc_20d','vol_z_20d']
OOS_START = pd.Timestamp('2026-07-16')
END = pd.Timestamp('2035-10-24')
np.seterr(all='ignore')

def build(end):
    o = {}
    for a in ASSETS:
        f = SD/f'{a}.csv'
        if f.exists():
            d = pd.read_csv(f, parse_dates=['date']).sort_values('date')
            d.columns = [str(c).lower() for c in d.columns]
            av = [c for c in ['open','high','low','close','volume'] if c in d.columns]
            d = d[d['date']<=end].set_index('date')[av].astype(float)
            o[a] = d
    m = {}
    for x in MACRO:
        d = pd.read_csv(ID/f'{x}.csv', parse_dates=['date']).sort_values('date')
        d.columns = [str(c).lower() for c in d.columns]
        m[x] = d[d['date']<=end].set_index('date')['close'].astype(float)
    al = set()
    for d in o.values(): al.update(d.index)
    for d in m.values(): al.update(d.index)
    df = pd.DataFrame(index=pd.DatetimeIndex(sorted(al)))
    for a,d in o.items():
        for c in d.columns: df[f'{a}__{c}'] = d[c]
    for x,d in m.items(): df[f'{x}__close'] = d
    return df

def cols(df, n): return pd.DataFrame({a: df[f'{a}__{n}'] for a in ASSETS}).dropna(axis=1,how='all')

def sig(fname):
    d = json.load(open(f'factors/{fname}.json'))
    sa = d.get('validation', {}).get('signal_artifact', {})
    if not sa: return None
    rows = list(csv.reader(io.StringIO(zlib.decompress(base64.b64decode(sa['data'])).decode())))
    dt = [r[0] for r in rows[1:]]
    M = np.array([[float(x) if x!='' else np.nan for x in r[1:]] for r in rows[1:]])
    return pd.DataFrame(M, index=pd.DatetimeIndex(dt), columns=rows[0][1:])

print('='*70)
print('FACTOR: on_minus_id_20 (Overnight - Intraday Return Spread, 20d avg)')
print('='*70)

df = build(END)
close = cols(df, 'close')
openp = cols(df, 'open')

idx = close.index.intersection(openp.index)
close = close.loc[idx]; openp = openp.loc[idx]

# Factor calculation
on_ret = openp / close.shift(1) - 1.0
intra_ret = close / openp - 1.0
fac = (on_ret - intra_ret).rolling(20, min_periods=12).mean()
fac = fac.replace([np.inf, -np.inf], np.nan)

print(f'Factor shape: {fac.shape}')
print(f'Date range: {fac.index[0].date()} to {fac.index[-1].date()}')
print(f'Assets: {fac.shape[1]}')
print()

# ---- IC by Horizon Full Sample ----
print('--- IC by Horizon (FULL SAMPLE) ---')
all_ic = {}
for H in [1, 2, 3, 5, 10, 20]:
    fwd = close.shift(-H) / close - 1.0
    ics = []; n_obs = 0
    for t in fac.index:
        if t not in fwd.index: continue
        fr = fac.loc[t]; fw = fwd.loc[t]
        m = fr.notna() & fw.notna()
        if m.sum() >= 8 and fr[m].std() > 0 and fw[m].std() > 0:
            rho, _ = spearmanr(fr[m], fw[m])
            if not np.isnan(rho):
                ics.append(rho); n_obs += 1
    ia = np.array(ics)
    if len(ia) >= 12:
        ic = float(ia.mean()); s = float(ia.std(ddof=1))
        icir = float(ic/s if s>1e-10 else 0)
        hit = float((ia>0).mean())
        pass_gate = ic >= 0.007 and abs(icir) >= 0.084
        print(f'H={H:2d}  IC={ic:+.6f}  ICIR={icir:+.6f}  hit={hit:.3f}  n={n_obs:4d}  PASS={pass_gate}')
        all_ic[H] = {'ic': ic, 'icir': icir, 'hit': hit, 'n': n_obs}

# ---- OOS / IS Split (H=10) ----
print()
print('--- REGIME SPLIT (H=10) ---')
for split_dt, label in [(None, 'IS (2020-01 to 2026-07)'), (OOS_START, 'OOS (2026-07 to 2035-10)')]:
    if split_dt is None:
        mask = fac.index < OOS_START
    else:
        mask = fac.index >= split_dt
    fac_sub = fac[mask]
    close_sub = close.loc[fac_sub.index]
    fwd = close_sub.shift(-10) / close_sub - 1.0
    ics = []
    for t in fac_sub.index:
        if t not in fwd.index: continue
        fr = fac_sub.loc[t]; fw = fwd.loc[t]
        m = fr.notna() & fw.notna()
        if m.sum() >= 8 and fr[m].std() > 0 and fw[m].std() > 0:
            rho, _ = spearmanr(fr[m], fw[m])
            if not np.isnan(rho): ics.append(rho)
    ia = np.array(ics)
    if len(ia) >= 12:
        ic = float(ia.mean()); s = float(ia.std(ddof=1))
        icir = float(ic/s if s>1e-10 else 0)
        print(f'  {label:30s}: IC={ic:+.6f}  ICIR={icir:+.6f}  hit={(ia>0).mean():.3f}  n={len(ia)}')

# ---- Decay Analysis ----
print()
print('--- DECAY ANALYSIS (avg IC by horizon) ---')
for H in [1, 2, 3, 5, 10, 15, 20]:
    if H in all_ic:
        print(f'  H={H:2d}  IC={all_ic[H]["ic"]:+.6f}  ICIR={all_ic[H]["icir"]:+.6f}')

# ---- Coverage ----
cov = fac.notna().sum(axis=1)
cov_all = fac.dropna(how='all')
print(f'\n--- COVERAGE ---')
print(f'  Mean assets/date: {cov.mean():.1f}')
print(f'