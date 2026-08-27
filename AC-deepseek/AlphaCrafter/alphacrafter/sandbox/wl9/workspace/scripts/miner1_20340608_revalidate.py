"""Re-validate existing ensemble factors on recent data (2028-09 .. 2034-06)."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd
import numpy as np
from scipy import stats

watchlist = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX', 'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

data = {}
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=1800)
    if df is not None and len(df) >= 800:
        data[sym] = df.set_index('date')['close']

close_df = pd.DataFrame(data).dropna()
ret_df = close_df.pct_change().dropna()
print(f"Close frame: {close_df.shape}, {close_df.index[0].date()}..{close_df.index[-1].date()}")

# forward returns on same date grid
fwd_10d = ret_df.rolling(10).sum().shift(-10)
fwd_5d = ret_df.rolling(5).sum().shift(-5)
fwd_1d = ret_df.shift(-1)

# macro series
vix = get_index_daily_data(symbol='VIX', days=1800).set_index('date')['close']
dxy = get_index_daily_data(symbol='DXY', days=1800).set_index('date')['close']
usdjpy = get_index_daily_data(symbol='USDJPY', days=1800).set_index('date')['close']
usdcny = get_index_daily_data(symbol='USDCNY', days=1800).set_index('date')['close']

d_vix = vix.pct_change()
d_dxy = dxy.pct_change()
d_usdjpy = usdjpy.pct_change()
d_usdcny = usdcny.pct_change()


def ic_series(factor_df, fwd_ret):
    """Cross-sectional IC per date. Returns list of (date, ic) and n_valid."""
    out = []
    for date in factor_df.index:
        if date not in fwd_ret.index:
            continue
        f = factor_df.loc[date]
        r = fwd_ret.loc[date]
        valid = (~f.isna()) & (~r.isna())
        if valid.sum() < 8:
            continue
        fv = f[valid].values.astype(float)
        rv = r[valid].values.astype(float)
        if np.std(fv) < 1e-12 or np.std(rv) < 1e-12:
            continue
        ic, _ = stats.pearsonr(fv, rv)
        out.append((date, ic))
    return out


def summarize(name, factor_df, fwd_ret, horizon):
    res = ic_series(factor_df, fwd_ret)
    if len(res) < 10:
        print(f"{name:28s} horizon={horizon}: too few IC dates ({len(res)})")
        return None
    ic_arr = np.array([x[1] for x in res])
    mean_ic = ic_arr.mean()
    std_ic = ic_arr.std()
    icir = mean_ic / std_ic if std_ic > 0 else 0
    hit = np.mean(ic_arr > 0)
    n_dates = len(res)
    print(f"{name:28s} h={horizon:2d}: ic={mean_ic:+.4f} icir={icir:+.3f} hit={hit:.3f} ndates={n_dates}")
    return {'mean_ic': mean_ic, 'icir': icir, 'hit': hit, 'n_dates': n_dates}


def load_factor(fid):
    """Rebuild factor signal frame from definition for revalidation."""
    if fid == 'beta_VIX_60':
        # cov(r_i, dvix)/var(dvix)
        s = pd.concat([ret_df, d_vix], axis=1, join='inner')
        val = s[watchlist].rolling(60).cov(s['VIX']) / s['VIX'].rolling(60).var()
        return val
    if fid == 'kaufman_eff_20d':
        # Kaufman efficiency ratio: |close - close[-20]| / sum(|dclose|) over 20
        def kramer(df):
            num = (df - df.shift(20)).abs()
            den = df.diff().abs().rolling(20).sum()
            return (num / den)
        return kramer(close_df)
    if fid == 'mom_120d_skip5':
        return close_df / close_df.shift(120) - 1
    if fid == 'mom_10d_skip5':
        return close_df / close_df.shift(10) - 1
    if fid == 'bb_width_20d':
        ma = close_df.rolling(20).mean()
        sd = close_df.rolling(20).std()
        return (2 * sd) / ma
    if fid == 'cny_beta_60':
        s = pd.concat([ret_df, d_usdcny], axis=1, join='inner')
        return s[watchlist].rolling(60).cov(s['USDCNY']) / s['USDCNY'].rolling(60).var()
    if fid == 'vol_z_20d':
        vol = ret_df.rolling(20).std()
        return (vol - vol.rolling(120).mean()) / vol.rolling(120).std()
    if fid == 'ac1_120d':
        return ret_df.rolling(120).apply(lambda x: pd.Series(x).autocorr(1) if len(x) >= 5 else np.nan, raw=False)
    if fid == 'dxy_corr_change_20_60':
        s = pd.concat([ret_df, d_dxy], axis=1, join='inner')
        c20 = s[watchlist].rolling(20).corr(s['DXY'])
        c60 = s[watchlist].rolling(60).corr(s['DXY'])
        return c20 - c60
    if fid == 'skew_20d':
        return ret_df.rolling(20).skew()
    if fid == 'vix_roc_20d':
        s = d_vix.rolling(20).sum()
        return s  # asset-constant; cross-sectional IC meaningless but kept for reference
    return None


# Re-validate the 10 ensemble factors on 10d horizon (admission horizon)
factors_to_check = [
    'beta_VIX_60', 'kaufman_eff_20d', 'mom_120d_skip5', 'mom_10d_skip5',
    'bb_width_20d', 'cny_beta_60', 'vol_z_20d', 'ac1_120d',
    'dxy_corr_change_20_60', 'skew_20d'
]

print("\n=== Re-validation on 10d forward horizon (2028-09..2034-06) ===")
for fid in factors_to_check:
    try:
        factor = load_factor(fid)
        if factor is None:
            print(f"{fid:28s}: SKIP (not rebuilt)")
            continue
        # Only evaluate over recent validation window: last ~600 days
        factor_recent = factor.iloc[-600:]
        fwd_recent = fwd_10d.loc[factor_recent.index[0]:]
        # align
        common = factor_recent.index.intersection(fwd_recent.index)
        summarize(fid, factor_recent.loc[common], fwd_10d.loc[common], 10)
    except Exception as e:
        print(f"{fid:28s}: ERROR {e}")
