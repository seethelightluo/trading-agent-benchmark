"""
miner_1 cycle 2034-08-03. Visible through 2034-08-02.
Goals:
1. Re-validate existing effective library factors on recent data
2. Explore fresh candidate factor families
3. Persist new passers / deprecate stale factors

Gates: abs IC>=0.0070, abs ICIR>=0.084 at 10d horizon; >=8 valid names.
"""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import numpy as np
import pandas as pd
from scipy import stats
import json, os, glob

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
             'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
MACRO_IDS = ['VIX','DXY','USDCNY','USDJPY','EURUSD']
N_DAYS = 2000
VISIBLE = pd.Timestamp('2034-08-02')

print("="*80, flush=True)
print(f"MINER CYCLE 2034-08-03 | Visible through {VISIBLE.date()}", flush=True)
print("="*80, flush=True)

# Fetch instrument data
inst_data = {}
for sym in WATCHLIST:
    df = get_stock_daily_data(symbol=sym, days=N_DAYS)
    if df is not None and len(df) > 120:
        s = df.set_index('date')['close'].astype(float)
        s = s[s.index <= VISIBLE]
        inst_data[sym] = s
close = pd.DataFrame(inst_data)
print(f"Close panel: {close.shape[0]} dates x {close.shape[1]} assets", flush=True)
print(f"  Range: {close.index[0].date()}..{close.index[-1].date()}", flush=True)
rets = close.pct_change().dropna()

# High/low
high_d, low_d = {}, {}
for sym in WATCHLIST:
    df_h = get_stock_daily_data(symbol=sym, days=N_DAYS)
    if df_h is not None and 'high' in df_h.columns and 'low' in df_h.columns:
        h = df_h.set_index('date')['high'].astype(float)
        l = df_h.set_index('date')['low'].astype(float)
        high_d[sym] = h[h.index <= VISIBLE].reindex(close.index)
        low_d[sym] = l[l.index <= VISIBLE].reindex(close.index)
    else:
        high_d[sym] = close[sym].copy()
        low_d[sym] = close[sym].copy()
high_df = pd.DataFrame(high_d)
low_df = pd.DataFrame(low_d)

# Volume
vol_d = {}
for sym in WATCHLIST:
    df_v = get_stock_daily_data(symbol=sym, days=N_DAYS)
    if df_v is not None and 'volume' in df_v.columns:
        v = df_v.set_index('date')['volume'].astype(float)
        vol_d[sym] = v[v.index <= VISIBLE].reindex(close.index)
vol_df = pd.DataFrame(vol_d)

# Macro data
macro = {}
for m in MACRO_IDS:
    df = get_index_daily_data(symbol=m, days=N_DAYS)
    if df is not None and len(df) > 120:
        s = df.set_index('date')['close'].astype(float)
        s = s[s.index <= VISIBLE]
        macro[m] = s
    else:
        print(f"  WARNING: {m} has no data", flush=True)

vix = macro.get('VIX')
dxy = macro.get('DXY')
cny = macro.get('USDCNY')
jpy = macro.get('USDJPY')
eur = macro.get('EURUSD')
dVIX = vix.pct_change() if vix is not None else None
dDXY = dxy.pct_change() if dxy is not None else None
dCNY = cny.pct_change() if cny is not None else None
dJPY = jpy.pct_change() if jpy is not None else None
dEUR = eur.pct_change() if eur is not None else None

# Forward returns
fwd_5d = rets.shift(-5).rolling(5).sum()
fwd_10d = rets.shift(-10).rolling(10).sum()
fwd_20d = rets.shift(-20).rolling(20).sum()

def compute_ic(fv, fwd, min_assets=8, min_dates=30):
    common = sorted(set(fv.index) & set(fwd.index))
    if not common:
        return {'IC': 0.0, 'ICIR': 0.0, 'n': 0, 'hit': 0.0}
    ics = []
    for d in common:
        x = fv.loc[d]; y = fwd.loc[d]
        m = x.notna() & y.notna()
        if m.sum() < min_assets: continue
        xv = x[m].rank().values; yv = y[m].rank().values
        if np.std(xv) > 0 and np.std(yv) > 0:
            ics.append(np.corrcoef(xv, yv)[0, 1])
    ics = np.array(ics)
    if len(ics) < min_dates:
        return {'IC': 0.0, 'ICIR': 0.0, 'n': len(ics), 'hit': 0.0}
    mu, sd = float(ics.mean()), float(ics.std())
    ir = mu / sd * np.sqrt(len(ics)) if sd > 0 else 0.0
    return {'IC': mu, 'ICIR': ir, 'n': len(ics), 'hit': float((ics > 0).mean())}

def report(name, fv, verbose=True):
    a5 = compute_ic(fv, fwd_5d)
    a10 = compute_ic(fv, fwd_10d)
    a20 = compute_ic(fv, fwd_20d)
    ok = abs(a10['IC']) >= 0.0070 and abs(a10['ICIR']) >= 0.084
    if verbose:
        flag = 'OK' if ok else '--'
        print(f"  [{flag}] {name:28s} IC={a10['IC']:+.4f} ICIR={a10['ICIR']:+.4f} n={a10['n']:4d} hit={a10['hit']:.3f} | [5]{a5['IC']:+.3f}[20]{a20['IC']:+.3f}", flush=True)
    return a10, ok

def beta_win(rd, mr, w):
    if mr is None: return None
    rd, mr = rd.align(mr, join='inner', axis=0)
    cov = rd.rolling(w).cov(mr)
    var = mr.rolling(w).var().replace(0, np.nan)
    return cov.div(var, axis=0)

def corr_win(rd, mr, w):
    if mr is None: return None
    rd, mr = rd.align(mr, join='inner', axis=0)
    out = pd.DataFrame(np.nan, index=rd.index, columns=rd.columns)
    for c in rd.columns:
        out[c] = rd[c].rolling(w).corr(mr)
    return out

def kaufman(c, w=20):
    num = (c - c.shift(w)).abs()
    den = c.diff().abs().rolling(w).sum().replace(0, np.nan)
    return num / den

def roll_ac(ser, w=120):
    def _ac(x):
        xc = x[~np.isnan(x)]
        if len(xc) < 5 or np.std(xc) < 1e-12: return np.nan
        return np.corrcoef(xc[:-1], xc[1:])[0,1]
    return ser.rolling(w).apply(_ac, raw=True)

# ============================
# PART 1: REVALIDATE EXISTING FACTORS
# ============================
print("\n=== PART 1: REVALIDATE EXISTING EFFECTIVE FACTORS ===", flush=True)

lib_factors = {}
lib_factors['beta_VIX_60'] = -beta_win(rets, dVIX, 60)
lib_factors['kaufman_eff_20d'] = kaufman(close, 20)
lib_factors['mom_120d_skip5'] = close.pct_change(120)
lib_factors['