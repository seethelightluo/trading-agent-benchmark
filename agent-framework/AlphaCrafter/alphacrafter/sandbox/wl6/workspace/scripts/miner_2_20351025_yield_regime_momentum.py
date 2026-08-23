import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={a:get_stock_daily_data(a, days=6000) for a in assets}
px={a: (d.set_index('date')['close'].astype(float) if d is not None else pd.Series(dtype=float)) for a,d in raw.items()}
P=pd.DataFrame(px).sort_index().ffill()
us=P['US10Y'].pct_change(20); cn=P['CN10Y'].pct_change(20)
stress=((us > us.rolling(252,min_periods=126).quantile(.65)) & (cn < cn.rolling(252,min_periods=126).quantile(.50))).astype(float)
mom=P.pct_change(40); vol=P.pct_change().rolling(60,min_periods=40).std()*np.sqrt(252)
base=mom/vol.replace(0,np.nan)
F=base.mul(1-2*stress,axis=0).shift(1)
fwd=P.shift(-1).div(P.shift(-1).shift(h) if False else P)-1
for h in [5,10,20,40]:
    fwd=P.shift(-h).div(P)-1; vals=[]; dates=[]; counts=[]
    for dt in F.index:
        z=pd.concat([F.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); counts.append(len(z))
    s=pd.Series(vals,index=dates).dropna()
    print(h,'dates',len(s),'avgN',round(np.mean(counts),3),'coverage',round(len(s)/max(1,len(F)-h),4),'IC',round(s.mean(),8),'ICIR',round(s.mean()/s.std(ddof=1),5),'hit',round((s>0).mean(),4))
rank=F.rank(axis=1,pct=True); print('turnover',round(rank.diff().abs().mean(axis=1).dropna().mean(),6),'stress_frac',round(stress.mean(),4),'rows',len(P),'assets',P.shape[1])
