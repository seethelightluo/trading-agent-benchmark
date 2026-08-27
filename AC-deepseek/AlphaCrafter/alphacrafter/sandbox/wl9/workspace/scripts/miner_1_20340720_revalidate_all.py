"""
miner_1 2034-07-20: Comprehensive re-validation of ALL factor candidates.
Load data from CSV files (stock_data + index_data for macro).
Compute each factor properly with VIX, DXY, USDCNY macro.
Gate: abs daily IC >= 0.0070, abs ICIR >= 0.084 (10-day horizon).
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
        if 'volume' in df.columns:
            V[a] = df['volume'].astype(float)
        else:
            V[a] = pd.Series(index=df.index, dtype=float)
        H[a] = df['high'].astype(float)
        L[a] = df['low'].astype(float)
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

close = pd.DataFrame(closes)
volume = pd.DataFrame(volumes).reindex(close.index)
high = pd.DataFrame(highs).reindex(close.index)
low = pd.DataFrame(lows).reindex(close.index)
close = close.dropna(how='all')
volume = volume.reindex(close.index)
high = high.reindex(close.index)
low = low.reindex(close.index)

vix = macro['VIX'].reindex(close.index)
dxy = macro['DXY'].reindex(close.index)
usdcny = macro['USDCNY'].reindex(close.index)

print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, "
      f"{close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}", flush=True)

rets = close.pct_change().dropna()

# Forward returns
def fwd_rets(h):
    return rets.rolling(h).mean().shift(-h)

fwd5 = fwd_rets(5)
fwd10 = fwd_rets(10)
fwd20 = fwd_rets(20)

def compute_ic(fv, fw, min_dates=30, start=None, flip=False):
    f = fv.reindex(fw.index)
    if flip:
        f = -f
    idx = fw.index
    if start:
        idx = idx[idx >= pd.Timestamp(start)]
    ics = []
    ok = 0
    for d in idx:
        x = f.loc[d]
        y = fw.loc[d]
        m = x.notna() & y.notna()
        if m.sum() >= 8:
            ok += 1
            xr = x[m].rank().values
            yr = y[m].rank().values
            if np.std(xr) > 0 and np.std(yr) > 0:
                ics.append(np.corrcoef(xr, yr)[0, 1])
    ics = np.array(ics)
    if len(ics) < min_dates:
        return dict(IC=0., ICIR=0., n=len(ics), hit=0., cov=0., ok=ok)
    mu = ics.mean()
    sd = ics.std()
    return dict(IC=float(mu), ICIR=float(mu / sd * np.sqrt(len(ics)) if sd > 0 else 0),
                n=len(ics), hit=float((ics > 0).mean()), cov=float(f.notna().mean().mean()), ok=ok)

def turnover(fv):
    s = np.sign(fv.rank(axis=1).sub(fv.shape[1] / 2)).fillna(0)
    return float((s.diff() != 0).mean().mean())

def report(name, fv, start=None, flip=False):
    a = compute_ic(fv, fwd10, start=start, flip=flip)
    b = compute_ic(fv, fwd5, start=start, flip=flip)
    c = compute_ic(fv, fwd20, start=start, flip=flip)
    ok = abs(a['IC']) >= 0.0070 and abs(a['ICIR']) >= 0.084
    print(f"[{'OK' if ok else '--'}] {name}: IC={a['IC']:.4f} ICIR={a['ICIR']:.4f} "
          f"n={a['n']} ok_dates={a['ok']} hit={a['hit']:.3f} cov={a['cov']:.3f} "
          f"tov={turnover(fv):.3f} | [5]{b['IC']:.3f} [20]{c['IC']:.3f}", flush=True)
    return a, ok

# Test window: 2020-2026 warmup then 2026-2034 live
RECENT = '2026-01-01'
RECENT2 = '2032-01-01'
FULL = '2022-01-01'

print("\n========== FACTOR RE-VALIDATION ==========", flush=True)

# ========== EXISTING ENSEMBLE FACTORS ==========

# 1. beta_VIX_60: 60d rolling beta of asset returns vs VIX returns
print("\n--- beta_VIX_60 ---", flush=True)
vix_ret = vix.pct_change()
asset_ret = rets
beta_vix = asset_ret.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var()
report('beta_VIX_60 (raw)', beta_vix, start=FULL)
report('beta_VIX_60 (neg)', beta_vix, start=FULL, flip=True)
report('beta_VIX_60 (raw)', beta_vix, start=RECENT)
report('beta_VIX_60 (neg)', beta_vix, start=RECENT, flip=True)
report('beta_VIX_60 (neg)', beta_vix, start=RECENT2, flip=True)

# 2. kaufman_eff_20d
print("\n--- kaufman_eff_20d ---", flush=True)
change = close.diff(20).abs()
path = close.diff().abs().rolling(20).sum()
kaufman = change / path.replace(0, np.nan)
report('kaufman_eff_20d', kaufman, start=FULL)
report('kaufman_eff_20d', kaufman, start=RECENT)