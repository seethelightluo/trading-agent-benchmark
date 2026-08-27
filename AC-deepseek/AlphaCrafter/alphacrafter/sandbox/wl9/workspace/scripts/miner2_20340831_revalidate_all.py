"""
miner2_20340831_revalidate_all.py
Re-validate ALL EFFECTIVE factors in the factor library on current data (2034-08-31).
Report full-sample and recent-window metrics, detect drift, and build signal artifacts
for ensemble reconciliation.
"""
import pandas as pd, numpy as np, json, os
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

uni = {}
for s in WATCH:
    df = get_stock_daily_data(symbol=s, days=4000)
    if df is None or len(df) < 300:
        df = get_index_daily_data(symbol=s, days=4000)
    if df is not None and len(df) >= 300:
        df = df.copy(); df['date'] = pd.to_datetime(df['date']); df = df.set_index('date').sort_index()
        uni[s] = df
    else:
        print("MISSING", s)

close = pd.DataFrame({s: uni[s]['close'] for s in uni}).sort_index()
ret = close.pct_change()
fwd_10 = ret.shift(-10)

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

def summarize(name, ic_series, window=None):
    full = ic_series
    if len(full) < 20:
        print(f"{name:26s}: TOO FEW DATES ({len(full)})")
        return None
    mean = full.mean(); std = full.std(ddof=1)
    icir = mean/std if std>0 else 0.0
    hit = (full>0).mean()
    first = full.index.min().strftime('%Y-%m-%d'); last = full.index.max().strftime('%Y-%m-%d')
    extra = ""
    if window is not None and len(full) >= window:
        r = full.iloc[-window:]
        rmean = r.mean(); rstd = r.std(ddof=1); ricir = rmean/rstd if rstd>0 else 0.0; rhit=(r>0).mean()
        extra = f"  recent{window}: ic={rmean:+.4f} icir={ricir:+.4f} hit={rhit:.3f}"
    print(f"{name:24s}: ic={mean:+.4f} icir={icir:+.4f} hit={hit:.3f} n={len(full):5d} [{first}~{last}]{extra}")
    out = {'ic':float(round(mean,4)),'icir':float(round(icir,4)),'hit':float(round(hit,4)),'n_dates':len(full),
           'first':first,'last':last}
    if window is not None and len(full) >= window:
        r = full.iloc[-window:]
        out['recent_ic']=float(round(r.mean(),4)); out['recent_icir']=float(round(r.mean()/r.std(ddof=1) if r.std(ddof=1)>0 else 0,4))
    return out

# Macro signals
dxy_csv = pd.read_csv('../persistent/index_data/DXY.csv')
d_dxy = dxy_csv.set_index(pd.to_datetime(dxy_csv['date'])).sort_index()['pct_change']
vix_csv = pd.read_csv('../persistent/index_data/VIX.csv')
d_vix = vix_csv.set_index(pd.to_datetime(vix_csv['date'])).sort_index()['close']
usdcny_csv = pd.read_csv('../persistent/index_data/USDCNY.csv')
d_usdcny = usdcny_csv.set_index(pd.to_datetime(usdcny_csv['date'])).sort_index()['pct_change']
usdjpy_csv = pd.read_csv('../persistent/index_data/USDJPY.csv')
d_usdjpy = usdjpy_csv.set_index(pd.to_datetime(usdjpy_csv['date'])).sort_index()['pct_change']

print(f"close: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} rows n_assets={close.shape[1]}")

F = {}
F['mom_10d_skip5'] = close / close.shift(10) - 1
F['mom_120d_skip5'] = close / close.shift(120) - 1
ma20 = close.rolling(20).mean(); sd20 = close.rolling(20).std()
F['bb_width_20d'] = (2 * sd20) / ma20
vol20 = ret.rolling(20).std()
vol120_mean = vol20.rolling(120).mean(); vol120_std = vol20.rolling(120).std(ddof=0)
F['vol_z_20d'] = (vol20 - vol120_mean) / vol120_std
F['skew_20d'] = ret.rolling(20).skew()
F['kurt_20d'] = ret.rolling(20).kurt()

def autocorr_srs(s, w=120):
    return s.rolling(w, min_periods=60).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
F['ac1_120d'] = pd.DataFrame({s: autocorr_srs(ret[s], 120) for s in close.columns})
F['kaufman_eff_20d'] = pd.DataFrame({s: (close[s]-close[s].shift(20)).abs()/close[s].diff().abs().rolling(20).sum() for s in close.columns})
F['rng_pos_20d'] = (close - close.shift(20)) / close.shift(20)

def streak_srs(s, w=14):
    out = pd.Series(np.nan, index=s.index)
    for i in range(w, len(s)):
        win = s.iloc[i-w+1:i+1]; pc = 0
        for j in range(len(win)-1, -1, -1):
            if win.iloc[j] > 0: pc += 1
            else: break
        out.iloc[i] = pc
    return out
streak_dict = {s: streak_srs(ret[s], 14) for s in close.columns}
F['streak_len_14'] = pd.DataFrame(streak_dict)

def dshs(s, w=60):
    h = s.rolling(w, min_periods=1).max(); days = pd.Series(0, index=s.index); c = 0
    for i in range(len(s)):
        if s.iloc[i] >= h.iloc[i]: c = 0
        else: c += 1
        days.iloc[i] = c
    return days
F['days_since_high_60'] = pd.DataFrame({k: dshs(close[k], 60) for k in close.columns}).replace(0, np.nan)

def rolling_beta_macro(ret_df, macro, wins):
    si = pd.concat([ret_df, macro.rename('M')], axis=1, join='inner')
    cov = si[ret_df.columns].rolling(wins).cov(si['M'])
    var = si['M'].rolling(wins).var()
    return cov / var
F['beta_VIX_60'] = rolling_beta_macro(ret, d_vix, 60)
F['cny_beta_60'] = rolling_beta_macro(ret, d_usdcny, 60)
F['vix_beta_cond_60x20'] = rolling_beta_macro(ret,
def dxy_corr_change(ret_df, fw, sw):
    si = pd.concat([ret_df, d_dxy.rename('M')], axis=1, join='inner')
    rf = si[ret_df.columns].rolling(fw).cov(si['M']); rs = si[ret_df.columns].rolling(sw).cov(si['M'])
    sf = si[ret_df.columns].rolling(fw).std(); sm = si['M'].rolling(fw).std()
    ss = si[ret_df.columns].rolling(sw).std(); sm2 = si['M'].rolling(sw).std()
    return rf/(sf*sm) - rs/(ss*sm2)
F['dxy_corr_change_20_60'] = dxy_corr_change(ret, 20, 60)
F['vix_roc_20d'] = pd.DataFrame({s: (d_vix/d_vix.shift(20)-1) for s in close.columns})

names = ['mom_10d_skip5','mom_120d_skip5','bb_width_20d','vol_z_20d','skew_20d','kurt_20d','ac1_120d',
         'kaufman_eff_20d','rng_pos_20d','beta_VIX_60','cny_beta_60','vix_beta_cond_60x20',
         'dxy_corr_change_20_60','vix_roc_20d','days_since_high_60','streak_len_14']

print()
print('=== FULL SAMPLE (start~2034-08-30) ===')
results = {}
for name in names:
    res = summarize(name, rank_ic_series(F[name], fwd_10), window=520)
    results[name] = res

# Save signal artifacts for ensemble reconciliation
artifact = {k: {'ic':v['ic'],'icir':v['icir'],'recent_ic':v.get('recent_ic'),'last_date':v['last']} for k,v in results.items() if v}
json.dump(artifact, open('miner2_revalid_20340831.json','w'), indent=1)
print('saved miner2_revalid_20340831.json')
