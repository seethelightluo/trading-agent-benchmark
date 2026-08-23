"""miner3_20270128: Explore 'down_days_ratio' family.
Fraction of down (negative return) days over rolling window. Measures skew of
trend direction distribution; a bearish-trend-quality signal. Direction: -1
(favor assets with fewer down days / more consistent upside)."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, 'scripts')
from miner3_20270128_common import load_data, build_panel, summarize

uni = load_data()
close, ret = build_panel(uni)
neg_flag = (ret < 0).astype(float)

for W in [10, 20, 40]:
    factor = neg_flag.rolling(W, min_periods=int(W*0.6)).mean()
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
    summarize(f"down_days_ratio_{W}", s, extra={"coverage": round(float(cov),4)})