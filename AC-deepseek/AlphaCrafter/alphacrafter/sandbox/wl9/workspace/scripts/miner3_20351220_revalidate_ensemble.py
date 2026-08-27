"""
miner3_20351220_revalidate_ensemble.py
Focused revalidation of all 10 ensemble factors on recent data.
Current date: 2035-12-20, visible through 2035-12-19.
"""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import numpy as np
import pandas as pd
from scipy import stats
import json, os

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
             'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
MACRO_IDS = ['VIX','DXY','USDCNY','USDJPY','EURUSD']
N_DAYS = 1300
VISIBLE = pd.Timestamp('2035-12-19')

print(f"MINER3 REVALIDATION 2035-12-20 | Visible: {VISIBLE.date()}", flush=True)

inst_data = {}
for sym in WATCHLIST:
    df = get_stock_daily_data(symbol=sym, days=N_DAYS)
    if df is not None and len(df) > 120:
        s = df.set_index('date')['close'].astype(float)
        s = s[s.index <= VISIBLE]
        inst_data[sym] = s
close = pd.DataFrame(inst_data)
print(f"Close: {close.shape[0]} dates x {close.shape[1]} assets, {close.index[0].date()}..{close.index[-1].date()}", flush=True)

high_d, low_d, vol_d = {}, {}, {}
for sym in WATCHLIST:
    df_h = get_stock_daily_data(symbol=sym, days=N_DAYS)
    if df_h is not None:
        for col, d in [('high',high_d),('low',low_d),('volume',vol_d)]:
            if col in df_h.columns:
                s = df_h.set_index('date')[col].astype(float)
                s = s[s.index <= VISIBLE].reindex(close.index)
                d[sym] = s
high_df = pd.DataFrame(high_d) if high_d else None
low_df = pd.DataFrame(low_d) if low_d else None
vol_df = pd.DataFrame(vol_d) if vol_d else None

macro = {}
for m in MACRO_IDS:
    df = get_index_daily_data(symbol=m, days=N_DAYS)
    if df is not None and len(df) > 120:
        s = df.set_index('date')['close'].astype(float)
        s = s[s.index <= VISIBLE]
        macro[m] = s
vix = macro.get('VIX')
dxy = macro.get('DXY')
usdcny = macro.get('USDCNY')

RECENT = close.index >= (VISIBLE - pd.Timedelta(days=1100))
close_r = close[RECENT]
high_r = high_df.reindex(close_r.index) if high_df is not None else None
low_r = low_df.reindex(close_r.index) if low_df is not None else None
vol_r = vol_df.reindex(close_r.index) if vol_df is not None else None
vix_r = vix.reindex(close_r.index) if vix is not None else None
dxy_r = dxy.reindex(close_r.index) if dxy is not None else None
usdcny_r = usdcny.reindex(close_r.index) if usdcny is not None else None

print(f"Recent panel: {close_r.shape[0]} dates, {close_r.shape[1]} assets", flush=True)

fwd_5 = close_r.shift(-5) / close_r - 1.0
fwd_10 = close_r.shift(-10) / close_r - 1.0
fwd_20 = close_r.shift(-20) / close_r - 1.0

def ic_analysis(fv, fwd, min_assets=8, min_dates=20):
    common = sorted(set(fv.index) & set(fwd.index))
    ics = []
    for d in common:
        x = fv.loc[d]; y = fwd.loc[d]
        m = x.notna() & y.notna()
        if m.sum() < min_assets: continue
        xv = x[m].values; yv = y[m].values
        rx = stats.rankdata(xv); ry = stats.rankdata(yv)
        if np.std(rx) > 0 and np.std(ry) > 0:
            ics.append(np.corrcoef(rx, ry)[0,1])
    ics = np.array(ics)
    if len(ics) < min_dates:
        return {'IC':0.0,'ICIR':0.0,'n_dates':len(ics),'hit':0.0}
    mu, sd = float(ics.mean()), float(ics.std(ddof=1))
    ir = mu/sd if sd > 0 else 0.0
    return {'IC': mu, 'ICIR': ir, 'n_dates': len(ics), 'hit': float((ics>0).mean())}

def report(fv, label):
    r5 = ic_analysis(fv, fwd_5)
    r10 = ic_analysis(fv, fwd_10)
    r20 = ic_analysis(fv, fwd_20)
    ok10 = abs(r10['IC']) >= 0.0070 and abs(r10['ICIR']) >= 0.084
    flag = 'PASS' if ok10 else 'FAIL'
    print(f"  [{flag}] {label:28s} IC10={r10['IC']:+.4f} IR10={r10['ICIR']:+.4f} nD={r10['n_dates']:4d} hit={r10['hit']:.3f} | IC5={r5['IC']:+.3f} IC20={r20['IC']:+.3f}", flush=True)
    return r10, ok10

print("\n=== Computing 10 ensemble factors on recent data ===", flush=True)
results = {}

# 1. beta_VIX_60
if vix_r is not None:
    rets_c = close_r.pct_change()
    rets_v = vix_r.pct_change()
    cov = rets_c.rolling(60, min_periods=30).cov(rets_v)
    var = rets_v.rolling(60, min_periods=30).var().replace(0, np.nan)
    f_beta_vix = cov.div(var, axis=0)
    r, ok = report(f_beta_vix, 'beta_VIX_60')
    results['beta_VIX_60'] = {'pass': ok, 'metrics': r}
else:
    print("  [SKIP] beta_VIX_60 (no VIX data)")
    results['beta_VIX_60'] = {'pass': False, 'metrics': {'IC':0,'ICIR':0,'n_dates':0,'hit':0}}

# 2. kaufman_eff_20d
f_kauf = (close_r - close_r.shift(20)).abs() / close_r.diff().abs().rolling(20, min_periods=10).sum().replace(0, np.nan)
r, ok = report(f_kauf, 'kaufman_eff_20d')
results['kaufman_eff_20d'] = {'pass': ok, 'metrics': r}

# 3. mom_120d_skip5
f_mom120 = close_r.shift(5) / close_r.shift(125) - 1.0
r, ok = report(f_mom120, 'mom_120d_skip5')
results['mom_120d_skip5'] = {'pass': ok, 'metrics': r}

# 4. mom_10d_skip5
f_mom10 = close_r.shift(5) / close_r.shift(15) - 1.0
r, ok = report(f_mom10, 'mom_10d_skip5')
results['mom_10d_skip5'] = {'pass': ok, 'metrics': r}

# 5. bb_width_20d
ma = close_r.rolling(20, min_periods=10).mean()
std = close_r.rolling(20, min_periods=10).std(ddof=0)
f_bb = (close_r - ma) / (2 * std).replace(0, np.nan)
r, ok = report(f_bb, 'bb_width_20d')
results['bb_width_20d'] = {'pass': ok, 'metrics': r}

# 6. cny_beta_60