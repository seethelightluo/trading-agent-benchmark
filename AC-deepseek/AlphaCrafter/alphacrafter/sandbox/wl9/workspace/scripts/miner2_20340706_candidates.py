"""Miner2 periodic revalidation of ensemble + candidate factor exploration."""
from alphacrafter.sim.utils import get_stock_daily_data
import pandas as pd
import numpy as np
from scipy import stats

watchlist = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load_macro(sym):
    df = pd.read_csv(f'../persistent/index_data/{sym}.csv', parse_dates=['date'])
    return df.set_index('date')['close']

data = {}
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=1800)
    if df is not None and len(df) >= 800:
        data[sym] = df.set_index('date')['close']
close_df = pd.DataFrame(data).dropna()
ret_df = close_df.pct_change().dropna()
fwd_10d = ret_df.rolling(10).sum().shift(-10)
print(f"Close: {close_df.shape} {close_df.index[0].date()}..{close_df.index[-1].date()}")

d_vix = load_macro('VIX').pct_change()
d_dxy = load_macro('DXY').pct_change()
d_usdcny = load_macro('USDCNY').pct_change()
d_usdjpy = load_macro('USDJPY').pct_change()

def ic_series(factor_df, fwd_ret):
    out = []
    for date in factor_df.index:
        if date not in fwd_ret.index: continue
        f = factor_df.loc[date]; r = fwd_ret.loc[date]
        valid = (~f.isna()) & (~r.isna())
        if valid.sum() < 8: continue
        fv = f[valid].values.astype(float); rv = r[valid].values.astype(float)
        if np.std(fv)<1e-12 or np.std(rv)<1e-12: continue
        ic,_ = stats.pearsonr(fv, rv)
        out.append((date, ic))
    return out

def summarize(name, factor_df, fwd_ret, window=None):
    ff = factor_df if window is None else factor_df.iloc[-window:]
    res = ic_series(ff, fwd_ret)
    if len(res) < 20:
        print(f"{name:30s}: too few ({len(res)})"); return None
    ic_arr = np.array([x[1] for x in res])
    mean=ic_arr.mean(); std=ic_arr.std()
    icir = mean/std if std>0 else 0; hit=np.mean(ic_arr>0)
    print(f"{name:30s}: ic={mean:+.4f} icir={icir:+.3f} hit={hit:.3f} ndates={len(res)}")
    return {'ic':mean,'icir':icir,'hit':hit}

def rolling_beta(ret_df, macro_ret, wins):
    si = pd.concat([ret_df, macro_ret.rename('M')], axis=1, join='inner')
    return si[watchlist].rolling(wins).cov(si['M']) / si['M'].rolling(wins).var()

# --- Ensemble factors revalidation ---
F = {}
F['beta_VIX_60'] = rolling_beta(ret_df, d_vix, 60)
F['cny_beta_60'] = rolling_beta(ret_df, d_usdcny, 60)
s = pd.concat([ret_df, d_dxy.rename('M')], axis=1, join='inner')
F['dxy_corr_change_20_60'] = s[watchlist].rolling(20).corr(s['M']) - s[watchlist].rolling(60).corr(s['M'])
ma = close_df.rolling(20).mean(); sd = close_df.rolling(20).std()
F['bb_width_20d'] = (2*sd)/ma
vol = ret_df.rolling(20).std()
F['vol_z_20d'] = (vol - vol.rolling(120).mean())/vol.rolling(120).std()
F['ac1_120d'] = ret_df.rolling(120).apply(lambda x: pd.Series(x).autocorr(1) if len(x)>=5 else np.nan, raw=False)
F['skew_20d'] = ret_df.rolling(20).skew()
def kaufman(df,w=20):
    return (df-df.shift(w)).abs() / df.diff().abs().rolling(w).sum()
F['kaufman_eff_20d'] = kaufman(close_df,20)
F['mom_120d_skip5'] = close_df/close_df.shift(120)-1
F['mom_10d_skip5'] = close_df/close_df.shift(10)-1

print("=== Ensemble revalidation (last 600 obs, 10d horizon) ===")
for fid in ['beta_VIX_60','kaufman_eff_20d','mom_120d_skip5','mom_10d_skip5',
            'bb_width_20d','cny_beta_60','vol_z_20d','ac1_120d',
            'dxy_corr_change_20_60','skew_20d']:
    try: summarize(fid, F[fid], fwd_10d, window=600)
    except Exception as e: print(f"{fid}: ERR {e}")

print("\n=== Candidate exploration ===")

# rng_pos_20d
rng = (close_df.rolling(20).max()-close_df.rolling(20).min())/close_df
summarize('rng_pos_20d', rng, fwd_10d, window=600)

# ac acceleration
ac20 = ret_df.rolling(20).apply(lambda x: pd.Series(x).autocorr(1) if len(x)>=5 else np.nan, raw=False)
ac60 = ret_df.rolling(60).apply(lambda x: pd.Series(x).autocorr(1) if len(x)>=5 else np.nan, raw=False)
summarize('ac_accel_20_60', ac20-ac60, fwd_10d, window=600)

# beta_DXY_60
summarize('beta_DXY_60', rolling_beta(ret_df, d_dxy, 60), fwd_10d, window=600)

# beta_USDJPY: carry/risk proxy
summarize('beta_USDJPY_60', rolling_beta(ret_df, d_usdjpy, 60), fwd_10d, window=600)

# vol_mom interaction
summarize('vol_mom_interact', F['vol_z_20d']*F['mom_10d_skip5'], fwd_10d, window=600)

# return deviation from 20d median (regime-adjusted rank)
# cross-sectional z of momentum
cs_z = F['mom_10d_skip5'].subtract(F['mom_10d_skip5'].mean(axis=1), axis=0).div(F['mom_10d_skip5'].std(axis=1), axis=0)
summarize('mom10_cs_z', cs_z, fwd_10d, window=600)

# days_since_high_60 inverse: recency of 60d high
def days_since_high(df, w=60):
    roll = df.rolling(w).max()
    # count days since last max via expanding argmax
    out = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    for c in df.columns:
        series = df[c]
        recent_max = series.rolling(w).max()
        # days since reaching max: cumcount loops
        days = []
        running = []
        last_max_day = np.nan
        for d in series.index:
            r = series.loc[d]
            s = series.loc[:d]
            maxval = s.iloc[-w:].max()
            idx = s.iloc[-w:]
            # find last index where equals maxval
            pos = np.where(idx.values == maxval)[0]
            if len(pos):
                days.append(d - idx.index[pos[-1]])
            else:
                days.append(np.nan)
        out[c] = days
    return out
# skip slow computation; approximate with streak
streak = ret_df.rolling(20).apply(lambda x: (x>0).sum()/len(x) if len(x)>=5 else np.nan, raw=False)
summarize('upside_ratio_20d', streak, fwd_10d, window=600)