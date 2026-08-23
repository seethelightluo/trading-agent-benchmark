"""miner3_20270128: Explore 'up_down_ratio' factor.
Factor: ratio of realized upside (mean positive daily return) to realized downside
(mean abs negative daily return) over rolling window. Tests asymmetry of the
return distribution - a risk/quality tilt factor. Direction: +1 (favor assets
whose daily upside consistently outweighs downside)."""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, 'scripts')
from miner3_20270128_common import load_data, build_panel, rank_ic, summarize

uni = load_data()
close, ret = build_panel(uni)
pos = ret.clip(lower=0)
neg = (-ret).clip(lower=0)

# rolling upside/downside mean ratio over window days (min periods)
for W in [20, 40, 60]:
    win_pos = pos.rolling(W, min_periods=15).mean()
    win_neg = neg.rolling(W, min_periods=15).mean()
    # denominator floor to avoid div0
    denom = win_neg.replace(0, np.nan)
    factor = (win_pos / denom).replace([np.inf,-np.inf], np.nan)
    # rank factor across assets per date
    fwd = 10
    fwd_ret = ret.shift(-fwd)
    dates=[]; ics=[]
    for dt in factor.index:
        frow = factor.loc[dt]; rrow = fwd_ret.loc[dt]
        m = frow.notna() & rrow.notna()
        if m.sum() < 8: continue
        ic = frow[m].corr(rrow[m], method='spearman')
        if not np.isnan(ic): dates.append(dt); ics.append(ic)
    s = pd.Series(ics, index=dates)
    cov = factor.notna().sum().sum()/(factor.shape[0]*factor.shape[1])
    summarize(f"updown_ratio_{W}", s, extra={"coverage": round(float(cov),4)})