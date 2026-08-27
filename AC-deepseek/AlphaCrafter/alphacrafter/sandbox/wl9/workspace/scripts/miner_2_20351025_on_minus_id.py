"""Explore: overnight-minus-intraday return spread factor (on_minus_id_20).
Hypothesis: Assets with strong positive overnight returns vs intraday weakness
(often institutional accumulation / retail selling pattern) predict positive
forward returns. The spread captures a flow signal.
"""
import json, base64, zlib, io, csv, numpy as np, pandas as pd
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
print('FACTOR: on_minus_id_20 (Overnight - Intraday return spread, 20d avg)')
print('='*70)

df = build(END)
close = cols(df, 'close')
openp = cols(df, 'open')

idx = close.index.intersection(openp.index)
close = close.loc[idx]; openp = openp.loc[idx]

on_ret = openp / close.shift(1) - 1.0        # overnight return
intra_ret = close / openp - 1.0              # intraday return
fac = (on_ret - intra_ret).rolling(20, min_periods=12).mean()
fac = fac.replace([np.inf, -np.inf], np.nan)

print(f'Factor shape: {fac.shape}')
print(f'Date range: {fac.index[0].date()} to {fac.index[-1].date()}')
print(f'Assets: {fac.shape[1]}')
print()

print('--- IC by Horizon ---')
for H in [1, 3, 5, 10, 20]:
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
        print(f'H={H:2d}  IC={ic:+.6f}  ICIR={icir:+.6f}  '
              f'hit={(ia>0).mean():.3f}  dates={n_obs:4d}  '
              f'PASS={ic>=0.007 and abs(icir)>=0.084}')
    else:
        print(f'H={H:2d}  too few dates ({len(ia)})')

cov = fac.notna().sum(axis=1)
print(f'\nCoverage: mean_assets_per_date={cov.mean():.1f} '
      f'min={cov.min()} max={cov.max()} frac_ge8={(cov>=8).mean():.3f}')

r = fac.rank(axis=1)
turn = float(r.diff(10).abs().mean(axis=1).sum())
print(f'Turnover (10d rank, sum/asset): {turn/15:.4f}')

print(f'\n--- Max Abs Library Correlation ---')
mp = fac.dropna(how='all')
max_corr = 0.0; max_lib = None
for lib in LIBS:
    lp = sig(lib)
    if lp is None: continue
    c = mp.index.intersection(pd.DatetimeIndex(lp.index))
    if len(c) < 60: continue
    a = mp.loc[c].rank(axis=1).values
    b = lp.loc[c].rank(axis=1).values
    rhos = [spearmanr(a[i], b[i])[0] for i in range(a.shape[0])
            if not np.isnan(a[i]).all() and not np.isnan(b[i]).all()
            and pd.Series(a[i]).nunique()>1 and pd.Series(b[i]).nunique()>1]
    rhos = np.array([x for x in rhos if not np.isnan(x)])
    if len(rhos):
        mx = np.abs(rhos).max()
        if mx > max_corr: max_corr = mx; max_lib = lib
print(f'max_abs_library_correlation (rank IC): {max_corr:.4f} vs {max_lib}')