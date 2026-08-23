"""miner3_20270128: Explore longer-horizon momentum (63/126/252d) and close/SMA ratio."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, 'scripts')
from miner3_20270128_common import load_data, build_panel, summarize

uni = load_data()
close, ret = build_panel(uni)

# longer momentum skip5
for W, skip in [(63,5),(126,5),(252,10)]:
    tot = close.pct_change(W)
    skip_ret = close.pct_change(skip) if skip>0 else 0
    # cumulative over interior: approximate first-order momentum = tot (ret over W)
    factor = tot
    fwd=10; fwd_ret = ret.shift(-fwd)
    dates=[];ics=[]
    for dt in factor.index:
        frow=factor.loc[dt]; rrow=fwd_ret.loc[dt]
        m=frow.notna()&rrow.notna()
        if m.sum()<8: continue
        ic=frow[m].corr(rrow[m],method='spearman')
        if not np.isnan(ic): dates.append(dt); ics.append(ic)
    s=pd.Series(ics,index=dates)
    cov=factor.notna().sum().sum()/(factor.shape[0]*factor.shape[1])
    summarize(f"mom_{W}d", s, extra={"coverage": round(float(cov),4)})

# close/SMA ratio
for N in [20,60,120]:
    sma = close.rolling(N, min_periods=N).mean()
    factor = close/sma
    fwd=10; fwd_ret=ret.shift(-fwd)
    dates=[];ics=[]
    for dt in factor.index:
        frow=factor.loc[dt]; rrow=fwd_ret.loc[dt]
        m=frow.notna()&rrow.notna()
        if m.sum()<8: continue
        ic=frow[m].corr(rrow[m],method='spearman')
        if not np.isnan(ic): dates.append(dt); ics.append(ic)
    s=pd.Series(ics,index=dates)
    cov=factor.notna().sum().sum()/(factor.shape[0]*factor.shape[1])
    summarize(f"close_sma_{N}", s, extra={"coverage": round(float(cov),4)})