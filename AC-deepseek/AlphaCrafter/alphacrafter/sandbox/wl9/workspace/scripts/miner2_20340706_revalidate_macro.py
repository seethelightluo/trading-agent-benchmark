"""Miner2 periodic re-validation (2028-06..2034-07) of ensemble factors incl. macro-fixed,
plus exploration of candidate factors."""
from alphacrafter.sim.utils import get_stock_daily_data
import pandas as pd
import numpy as np
from scipy import stats

watchlist = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load_macro(sym):
    p = f'../persistent/index_data/{sym}.csv'
    df = pd.read_csv(p, parse_dates=['date'])
    return df.set_index('date')['close']

data = {}
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=1800)
    if df is not None and len(df) >= 800:
        data[sym] = df.set_index('date')['close']
close_df = pd.DataFrame(data).dropna()
ret_df = close_df.pct_change().dropna()
fwd_10d = ret_df.rolling(10).sum().shift(-10)
print(f"Close frame: {close_df.shape}, {close_df.index[0].date()}..{close_df.index[-1].date()}")

# macro series aligned
vix = load_macro('VIX'); dxy = load_macro('DXY'); usdjpy = load_macro('USDJPY'); usdcny = load_macro('USDCNY')
d_vix = vix.pct_change(); d_dxy = dxy.pct_change(); d_usdjpy = usdjpy.pct_change(); d_usdcny = usdcny.pct_change()

def ic_series(factor_df, fwd_ret):
    out = []
    for date in factor_df.index:
        if date not in fwd_ret.index:
            continue
        f = factor_df.loc[date]; r = fwd_ret.loc[date]
        valid = (~f.isna()) & (~r.isna())
        if valid.sum() < 8: continue
        fv = f[valid].values.astype(float); rv = r[valid].values.astype(float)
        if np.std(fv) < 1e-12 or np.std(rv) < 1e-12: continue
        ic, _ = stats.pearsonr(fv, rv)
        out.append((date, ic))
    return out

def summarize(name, factor_df, fwd_ret, horizon, window=None):
    ff = factor_df
    if window is not None:
        ff = ff.iloc[-window:]
    res = ic_series(ff, fwd_ret)
    if len(res) < 20:
        print(f"{name:28s} h={horizon}: too few ({len(res)})")
        return None
    ic_arr = np.array([x[1] for x in res])
    mean = ic_arr.mean(); std = ic_arr.std()
    icir = mean/std if std>0 else 0
    hit = np.mean(ic_arr > 0)
    print(f"{name:28s} h={horizon:2d}: ic={mean:+.4f} icir={icir:+.3f} hit={hit:.3f} ndates={len(res)}")
    return {'ic':mean,'icir':icir,'hit':hit,'n':len(res)}

# Rebuild ensemble factors
F = {}
# beta_VIX_60 (neg). macro asset constant cov
def rolling_beta(close_df, ret_df, macro_ret, wins):
    si = pd.concat([ret_df, macro_ret.rename('M')], axis=1, join='inner')
    cov = si[watchlist].rolling(wins).cov(si['M'])
    var = si['M'].rolling(wins).var()
    return cov/var
F['beta_VIX_60'] = rolling_beta(close_df, ret_df, d_vix, 60)
F['cny_beta_60'] = rolling_beta(close_df, ret_df, d_usdcny, 60)
s = pd.concat([ret_df, d_dxy.rename('M')], axis=1, join='inner')
F['dxy_corr_change_20_60'] = s[watchlist].rolling(20).corr(s['M']) - s[watchlist].rolling(60).corr(s['M'])
# bollinger, cny beta, vol_z, ac1, skew
ma = close_df.rolling(20).mean(); sd = close_df.rolling(20).std()
F['bb_width_20d'] = (2*sd)/ma
vol = ret_df.rolling(20).std()
F['vol_z_20d'] = (vol - vol.rolling(120).mean())/vol.rolling(120).std()
F['ac1_120d'] = ret_df.rolling(120).apply(lambda x: pd.Series(x).autocorr(1) if len(x)>=5 else np.nan, raw=False)
F['skew_20d'] = ret_df.rolling(20).skew()
# Kaufman efficiency
def kaufman(df, w=20):
    num = (df - df.shift(w)).abs(); den = df.diff().abs().rolling(w).sum()
    return num/den
F['kaufman_eff_20d'] = kaufman_df(close_df, 20)
F['mom_120d_skip5'] = close_df/close_df.shift(120)-1
F['mom_10d_skip5'] = close_df/close_df.shift(10)-1

print("\n=== Re-validation on 10d horizon, full recent (last 600 obs) ===")
for fid in F:
    try:
        summarize(fid, F[fid], fwd_10d, 10, window=600)
    except Exception as e:
        print(f"{fid:28s}: ERR {e}")