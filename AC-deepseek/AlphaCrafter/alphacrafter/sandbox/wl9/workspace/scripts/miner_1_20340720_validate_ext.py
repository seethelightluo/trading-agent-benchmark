"""
miner_1 2034-07-20: Extended factor validation (part 2).
Covers momentum, vol, AC, DXY macro factors + novel candidates.
Loads all from CSV files.
"""
import json, math, numpy as np, pandas as pd
from pathlib import Path

VISIBLE_END = '2034-07-19'
SD = Path('../persistent/stock_data')
ID = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
MACRO = ['VIX','DXY','USDCNY','EURUSD','USDJPY']

def load_assets(end=VISIBLE_END):
    C = {}; V = {}; H = {}; L = {}
    for a in ASSETS:
        f = SD / f'{a}.csv'
        df = pd.read_csv(f, parse_dates=['date'])
        df = df[df['date'] <= end].sort_values('date').set_index('date')
        C[a] = df['close'].astype(float)
        V[a] = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(index=df.index, dtype=float)
        H[a] = df['high'].astype(float); L[a] = df['low'].astype(float)
    return C, V, H, L

def load_macro(end=VISIBLE_END):
    M = {}
    for a in MACRO:
        f = ID / f'{a}.csv'
        df = pd.read_csv(f, parse_dates=['date'])
        df = df[df['date'] <= end].sort_values('date').set_index('date')
        M[a] = df['close'].astype(float)
    return M

closes, volumes, highs, lows = load_assets()
macro = load_macro()
close = pd.DataFrame(closes).dropna(how='all')
volume = pd.DataFrame(volumes).reindex(close.index)
high = pd.DataFrame(highs).reindex(close.index)
low = pd.DataFrame(lows).reindex(close.index)
vix = macro['VIX'].reindex(close.index)
dxy = macro['DXY'].reindex(close.index)
usdcny = macro['USDCNY'].reindex(close.index)
vix_ret = vix.pct_change(); dxy_ret = dxy.pct_change()

print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets", flush=True)
rets = close.pct_change().dropna()
def fwd_rets(h): return rets.rolling(h).mean().shift(-h)
fwd5 = fwd_rets(5); fwd10 = fwd_rets(10); fwd20 = fwd_rets(20)

def compute_ic(fv, fw, min_dates=30, start=None, flip=False):
    f = fv.reindex(fw.index)
    if flip: f = -f
    idx = fw.index
    if start: idx = idx[idx >= pd.Timestamp(start)]
    ics = []; ok = 0
    for d in idx:
        x = f.loc[d]; y = fw.loc[d]; m = x.notna() & y.notna()
        if m.sum() >= 8:
            ok += 1
            xr = x[m].rank().values; yr = y[m].rank().values
            if np.std(xr) > 0 and np.std(yr) > 0:
                ics.append(np.corrcoef(xr, yr)[0, 1])
    ics = np.array(ics)
    if len(ics) < min_dates:
        return dict(IC=0., ICIR=0., n=len(ics), hit=0., cov=0., ok=ok)
    mu = ics.mean(); sd = ics.std()
    return dict(IC=float(mu), ICIR=float(mu/sd*np.sqrt(len(ics)) if sd>0 else 0),
                n=len(ics), hit=float((ics>0).mean()), cov=float(f.notna().mean().mean()), ok=ok)

def turnover(fv):
    s = np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0)
    return float((s.diff()!=0).mean().mean())

def report(name, fv, start=None, flip=False):
    a = compute_ic(fv, fwd10, start=start, flip=flip)
    b = compute_ic(fv, fwd5, start=start, flip=flip)
    c = compute_ic(fv, fwd20, start=start, flip=flip)
    ok = abs(a['IC'])>=0.0070 and abs(a['ICIR'])>=0.084
    print(f"[{'OK' if ok else '--'}] {name}: IC={a['IC']:.4f} ICIR={a['ICIR']:.4f} "
          f"n={a['n']} ok_dates={a['ok']} hit={a['hit']:.3f} cov={a['cov']:.3f} "
          f"tov={turnover(fv):.3f} | [5]{b['IC']:.3f} [20]{c['IC']:.3f}", flush=True)
    return a, ok

FULL='2022-01-01'; RECENT='2026-01-01'; RECENT2='2032-01-01'

# mom_120d_skip5
print("--- mom_120d_skip5 ---", flush=True)
report('mom_120d_skip5', close/close.shift(125)-1, start=FULL)
report('mom_120d_skip5', close/close.shift(125)-1, start=RECENT)
report('mom_120d_skip5', close/close.shift(125)-1, start=RECENT2)

# mom_10d_skip5
print("--- mom_10d_skip5 ---", flush=True)
report('mom_10d_skip5', close/close.shift(15)-1, start=FULL)
report('mom_10d_skip5', close/close.shift(15)-1, start=RECENT)

# bb_width_20d
print("--- bb_width_20d ---", flush=True)
ma = close.rolling(20).mean(); std = close.rolling(20).std()
bbw = (2*std)/ma.replace(0,np.nan)
report('bb_width_20d', bbw, start=FULL)
report('bb_width_20d', bbw, start=RECENT)

# vol_z_20d
print("--- vol_z_20d ---", flush=True)
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()
vol_z = (vol20 - vol60.rolling(20).mean())/(vol60.rolling(20).std().replace(0,np.nan))
report('vol_z_20d', vol_z, start=FULL)
report('vol_z_20d', vol_z, start=RECENT)

# ac1_120d
print("--- ac1_120d ---", flush=True)
ac1 = rets.rolling(120).apply(lambda x: x.autocorr() if len(x)>5 else 0, raw=False)
report('ac1_120d (neg=use)', -ac1, start=FULL)
report('ac1_120d (neg=use)', -ac1, start=RECENT)

# cny_beta_60
print("--- cny_beta_60 ---", flush=True)
cny_ret = usdcny.pct_change()
beta_cny = rets.rolling(60).cov(cny_ret)/cny_ret.rolling(60).var()
report('cny_beta_60', beta_cny, start=FULL, flip=False)
report('cny_beta_60', beta_cny, start=RECENT, flip=False)

# dxy_corr_change_20_60
print("--- dxy_corr_change_20_60 ---", flush=True)
corr20 = rets.rolling