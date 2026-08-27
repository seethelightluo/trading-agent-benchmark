"""
miner1_20351206_revalidate_all.py
Comprehensive revalidation of all EFFECTIVE factors in the factor library.
Current date: 2035-12-06, visible through 2035-12-05.
"""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import numpy as np
import pandas as pd
from scipy import stats
import json, os, glob, base64, zlib, io, csv, hashlib

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
             'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
MACRO_IDS = ['VIX','DXY','USDCNY','USDJPY','EURUSD']
N_DAYS = 2600
VISIBLE = pd.Timestamp('2035-12-05')

print(f"MINER CYCLE 2035-12-06 | Visible through {VISIBLE.date()}", flush=True)

# --- Data ---
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

high_d, low_d, vol_d, open_d = {}, {}, {}, {}
for sym in WATCHLIST:
    df_h = get_stock_daily_data(symbol=sym, days=N_DAYS)
    if df_h is not None:
        for col in ['high','low','volume','open']:
            if col in df_h.columns:
                s = df_h.set_index('date')[col].astype(float)
                s = s[s.index <= VISIBLE].reindex(close.index)
                if col == 'high': high_d[sym] = s
                elif col == 'low': low_d[sym] = s
                elif col == 'volume': vol_d[sym] = s
                elif col == 'open': open_d[sym] = s
high_df = pd.DataFrame(high_d)
low_df = pd.DataFrame(low_d)
vol_df = pd.DataFrame(vol_d)
open_df = pd.DataFrame(open_d)

macro = {}
for m in MACRO_IDS:
    df = get_index_daily_data(symbol=m, days=N_DAYS)
    if df is not None and len(df) > 120:
        s = df.set_index('date')['close'].astype(float)
        s = s[s.index <= VISIBLE]
        macro[m] = s
vix = macro.get('VIX'); dxy = macro.get('DXY')

fwd_5d_all = rets.shift(-5).rolling(5).sum()
fwd_10d_all = rets.shift(-10).rolling(10).sum()
fwd_20d_all = rets.shift(-20).rolling(20).sum()

def compute_ic(fv, fwd, min_assets=8, min_dates=24):
    common = sorted(set(fv.index) & set(fwd.index))
    if not common: return {'IC':0.0,'ICIR':0.0,'n':0,'hit':0.0}
    ics = []
    for d in common:
        x = fv.loc[d]; y = fwd.loc[d]
        m = x.notna() & y.notna()
        if m.sum() < min_assets: continue
        xv = x[m].rank().values; yv = y[m].rank().values
        if np.std(xv) > 0 and np.std(yv) > 0:
            ics.append(np.corrcoef(xv, yv)[0,1])
    ics = np.array(ics)
    if len(ics) < min_dates: return {'IC':0.0,'ICIR':0.0,'n':len(ics),'hit':0.0}
    mu, sd = float(ics.mean()), float(ics.std(ddof=1))
    ir = mu/sd if sd > 0 else 0.0
    return {'IC': mu, 'ICIR': ir, 'n': len(ics), 'hit': float((ics>0).mean())}

def report(name, fv):
    a5 = compute_ic(fv, fwd_5d_all)
    a10 = compute_ic(fv, fwd_10d_all)
    a20 = compute_ic(fv, fwd_20d_all)
    ok = abs(a10['IC']) >= 0.0070 and abs(a10['ICIR']) >= 0.084
    flag = 'OK' if ok else '--'
    print(f"  [{flag}] {name:28s} IC={a10['IC']:+.4f} ICIR={a10['ICIR']:+.4f} n={a10['n']:4d} hit={a10['hit']:.3f} | [5]{a5['IC']:+.3f}[20]{a20['IC']:+.3f}", flush=True)
    return a10, ok

def beta_win(rd, mr, w):
    if mr is None: return None
    rd, mr = rd.align(mr, join='inner', axis=0)
    cov = rd.rolling(w, min_periods=max(w//2,20)).cov(mr)
    var = mr.rolling(w, min_periods=max(w//2,20)).var().replace(0, np.nan)
    return cov.div(var, axis=0)

def corr_win(rd, mr, w):
    if mr is None: return None
    rd, mr = rd.align(mr, join='inner', axis=0)
    out = pd.DataFrame(np.nan, index=rd.index, columns=rd.columns)
    for c in rd.columns:
        out[c] = rd[c].rolling(w, min_periods=max(w//2,20)).corr(mr)
    return out

def kaufman(c, w=20):
    num = (c - c.shift(w)).abs()
    den = c.diff().abs().rolling(w, min_periods=w//2).sum().replace(0, np.nan)
    return num/den

def roll_ac(ser, w=120):
    def _ac(x):
        xc = x[~np.isnan(x)]
        if len(xc) < 5 or np.std(xc) < 1e-12: return np.nan
        return np.corrcoef(xc[:-1], xc[1:])[0,1]
    return ser.rolling(w, min_periods=w//2).apply(_ac, raw=True)

def roll_skew(ser, w=20):
    return ser.rolling(w, min_periods=w//2).apply(lambda x: stats.skew(x) if len(x) >= w//2 else np.nan, raw=True)

#================= REVALIDATION =================
print("\n=== REVALIDATING ALL 10 ENSEMBLE FACTORS (recent 500d) ===\n", flush=True)
RECENT = close.index >= (VISIBLE - pd.Timedelta(days=760))
close_r = close[RECENT]
rets_r = rets.reindex(close_r.index).dropna()
high_r = high_df.reindex(close_r.index) if high_df is not None else None
low_r = low_df.reindex(close_r.index) if low_df is not None else None
vol_r = vol_df.reindex(close_r.index) if vol_df is not None else None
open_r = open_df.reindex(close_r.index) if open_df is not None else None
vix_r = vix.reindex(close_r.index) if vix is not None else None
dxy_r = dxy.reindex(close_r.index) if dxy is not None else None

fwd5_f = close_r.shift(-5) / close_r - 1.0
fwd10_f = close_r.shift(-10) / close_r - 1.0
fwd20_f = close_r.shift(-20) / close_r - 1.0

def compute_ic_r(fv, fwd, min_assets=8, min_dates=20):
    common = sorted(set(fv.index) & set(fwd.index))
    ics = []