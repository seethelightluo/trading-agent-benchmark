"""miner_1 2031-05-19: Re-validation of the 3 currently effective factors
(vol_adj_mom_accel_20x60, dn_mkt_beta_60d, rate_beta_cn10y_60d) on the full
sample and the recent 2y window (2029-05-19..2031-05-16)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts')
from miner_1_20310519_lib import (WATCH, load_prices, fwd_ret, ic_series,
                                  summarize_ic, decay_profile, coverage_stats,
                                  turnover_10d_rank)

px, vo = load_prices()
ret = px.pct_change()
print(f'Price matrix: {px.shape} dates {px.index[0].date()}..{px.index[-1].date()}\n')

# ---------------- factor builders ----------------
def f_vol_adj_mom_accel(px, ret, fast=20, slow=60, vol=20):
    mom_fast = px / px.shift(fast) - 1.0
    mom_slow = px / px.shift(slow) - 1.0
    vol20 = ret.rolling(vol).std()
    return (mom_fast - mom_slow) / vol20

def f_dn_mkt_beta(px, ret, win=60, min_obs=40):
    mkt = ret.mean(axis=1)          # equal-weight 15-asset daily return
    x = mkt.clip(upper=0.0)         # down-market regressor (zeros on up days)
    out = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
    for t in range(win, len(ret)):
        xw = x.iloc[t-win:t]
        yw = ret.iloc[t-win:t]
        nz = (xw != 0).sum()
        if nz < min_obs:
            continue
        vx = xw.var(ddof=1)
        if vx <= 0:
            continue
        beta = yw.apply(lambda col: col.cov(xw) / vx)
        out.iloc[t] = beta.values
    return out

def f_rate_beta_cn10y(px, ret, win=60, min_obs=40):
    if 'CN10Y' not in px.columns:
        return pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
    x = px['CN10Y'].pct_change()
    out = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
    for t in range(win, len(ret)):
        xw = x.iloc[t-win:t]
        yw = ret.iloc[t-win:t]
        nn = xw.notna().sum()
        if nn < min_obs:
            continue
        vx = xw.var(ddof=1)
        if vx <= 0:
            continue
        beta = yw.apply(lambda col: col.cov(xw) / vx)
        out.iloc[t] = beta.values
    return out

factors = {
    'vol_adj_mom_accel_20x60': f_vol_adj_mom_accel(px, ret),
    'dn_mkt_beta_60d': f_dn_mkt_beta(px, ret),
    'rate_beta_cn10y_60d': f_rate_beta_cn10y(px, ret),
}

WINDOWS = {
    'full': (px.index[0], px.index[-1]),
    'recent2y': (pd.Timestamp('2029-05-19'), px.index[-1]),
}

for fid, fv in factors.items():
    print('=' * 90)
    print(f'FACTOR {fid}')
    for wname, (a, b) in WINDOWS.items():
        sub = fv.loc[a:b]
        r10 = fwd_ret(px.loc[a:b], 10)
        ic = ic_series(sub, r10)
        res = summarize_ic(ic, label=f'  [{wname}]', horizon=10)
        if len(ic):
            dec = decay_profile(sub, px.loc[a:b], horizons=(1, 2, 3, 5, 10, 20), verbose=True)
            cs = coverage_stats(sub)
            to = turnover_10d_rank(sub)
            print(f'    coverage_asset_days={cs["coverage_asset_days"]:.3f} '
                  f'coverage_dates_ge8={cs["coverage_dates_ge8"]:.3f} turnover_10d_rank={to:.3f}')
            # also 5d horizon for robustness
            r5 = fwd_ret(px.loc[a:b], 5)
            ic5 = ic_series(sub, r5)
            summarize_ic(ic5, label=f'  [{wname}] h=5', horizon=5)
    print()
