\"""
miner2_20341109_revalidate_all.py
Re-validate ALL EFFECTIVE factors in the factor library on current data (2034-11-09).
Detect drift on recent window, flag stale factors, compute signal artifacts.
Gate: abs IC>=0.0070, abs ICIR>=0.084 at 10d horizon, >=8 valid names.
Cross-asset universe of 15 tradable instruments (small universe, robust across dates).
"""
import pandas as pd, numpy as np, json, os
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
VISIBLE = pd.Timestamp('2034-11-08')

uni = {}
for s in WATCH:
    df = get_stock_daily_data(symbol=s, days=4000)
    if df is None or len(df) < 300:
        df = get_index_daily_data(symbol=s, days=4000)
    if df is not None and len(df) >= 300:
        df = df.copy(); df['date'] = pd.to_datetime(df['date']); df = df.set_index('date').sort_index()
        df = df[df.index <= VISIBLE]
        uni[s] = df
    else:
        print("MISSING", s)

close = pd.DataFrame({s: uni[s]['close'] for s in uni}).sort_index()
ret = close.pct_change()
fwd_10 = ret.shift(-10)
print(f"close: {close.index[0].date()}..{close.index[-1].date()} rows={len(close)} assets={close.shape[1]}", flush=True)

def rank_ic_series(factor_df, fwd_ret, min_valid=8):
    ics = {}
    for dt in factor_df.index:
        if dt not in fwd_ret.index: continue
        f = factor_df.loc[dt]; r = fwd_ret.loc[dt]
        mask = f.notna() & r.notna()
        if mask.sum() < min_valid: continue
        ic = f[mask].corr(r[mask], method='spearman')
        if not np.isnan(ic): ics[dt] = ic
    return pd.Series(ics)

def summarize(name, ic_series, window=400):
    full = ic_series
    if len(full) < 20:
        print(f"{name:26s}: TOO FEW DATES ({len(full)})")
        return None
    mean = full.mean(); std = full.std(ddof=1)
    icir = mean/std if std>0 else 0.0
    hit = (full>0).mean()
    extra = ""
    recent = None
    if window is not None and len(full) >= window:
        r = full.iloc[-window:]
        rmean = r.mean(); rstd = r.std(ddof=1); ricir = rmean/rstd if rstd>0 else 0.0
        extra = f"  recent{window}: ic={rmean:+.4f} icir={ricir:+.4f}"
        recent = {'ic': round(rmean,4), 'icir': round(ricir,4)}
    ok_full = abs(mean)>=0.0070 and abs(icir)>=0.084
    ok_recent = recent is not None and abs(recent['ic'])>=0.0070 and abs(recent['icir'])>=0.084
    flag = 'OK' if ok_full else '--'
    print(f"[{flag}]{name:24s}: ic={mean:+.4f} icir={icir:+.4f} hit={hit:.3f} n={len(full):5d}{extra}", flush=True)
    return {'ic':round(mean,4),'icir':round(icir,4),'hit':round(hit,4),'n_dates':len(full),
            'recent':recent,'ok_full':ok_full,'ok_recent':ok_recent}

# Macro signals
dxy_csv = pd.read_csv('../persistent/index_data/DXY.csv')
d_dxy = dxy_csv.set_index(pd.to_datetime(dxy_csv['date'])).sort_index()['pct_change']
vix_csv = pd.read_csv('../persistent/index_data/VIX.csv')
vix_lvl = vix_csv.set_index(pd.to_datetime(vix_csv['date'])).sort_index()['close']
d_vix = vix_lvl.pct_change()
usdcny_csv = pd.read_csv('../persistent/index_data/USDCNY.csv')
d_usdcny = usdcny_csv.set_index(pd.to_datetime(usdcny_csv['date'])).sort_index()['pct_change']
usdjpy_csv = pd.read_csv('../persistent/index_data/USDJPY.csv')
d_usdjpy = usdjpy_csv.set_index(pd.to_datetime(usdjpy_csv['date'])).sort_index()['pct_change']

F = {}
# --- Price / momentum ---
F['mom_10d_skip5'] = close / close.shift(10) - 1
F['mom_120d_skip5'] = close / close.shift(120) - 1
ma20 = close.rolling(20).mean(); sd20 = close.rolling(20).std()
F['bb_width_20d'] = (2 * sd20) / ma20
F['rng_pos_20d'] = (close - close.shift(20)) / close.shift(20)

# --- Volatility ---
vol20 = ret.rolling(20).std()
vol120_mean = vol20.rolling(120).mean(); vol120_std = vol20.rolling(120).std(ddof=0)
F['vol_z_20d'] = (vol20 - vol120_mean) / vol120_std
F['skew_20d'] = ret.rolling(20).skew()
F['kurt_20d'] = ret.rolling(20).kurt()

def autocorr_srs(s, w=120):
    return s.rolling(w, min_periods=60).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
F['ac1_120d'] = pd.DataFrame({s: autocorr_srs(ret[s], 120) for s in close.columns})
F['kaufman_eff_20d'] = pd.DataFrame({s: (close[s]-close[s].shift(20)).abs()/close[s].diff().abs().rolling(20).sum() for s in close.columns})

def streak_srs(s, w=14):
    out = pd.Series(np.nan, index=s.index)
    for i in range(w, len(s)):
        win = s.iloc[i-w+1:i+1]; pc = 0
        for j in range(len(win)-1, -1, -1):
            if win.iloc[j] > 0: pc += 1
            else: break
        out.iloc[i] = pc
    return out
F['streak_len_14'] = pd.DataFrame({s: streak_srs(ret[s], 14) for s in close.columns})

def dshs(s, w=60):
    h = s.rolling(w, min_periods=1).max(); days = pd.Series(0.0, index=s.index); c = 0
    for i in range(len(s)):
        if s.iloc[i] >= h.iloc