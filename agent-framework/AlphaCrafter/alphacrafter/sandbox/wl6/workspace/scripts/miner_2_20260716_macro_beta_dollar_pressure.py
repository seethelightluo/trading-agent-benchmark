import pandas as pd, numpy as np, glob
from pathlib import Path
from scipy.stats import spearmanr

root=Path('../persistent')
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
    f=root/'stock_data'/f'{a}.csv'
    if not f.exists(): f=root/'index_data'/f'{a}.csv'
    d=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].astype(float)
    px[a]=d
p=pd.DataFrame(px).sort_index()
dxy=pd.read_csv(root/'index_data'/'DXY.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(p.index).ffill()
r=p.pct_change()
dr=dxy.pct_change()
# macro-beta pressure: negative beta times recent dollar impulse; beta estimated only through t
beta=dr.rolling(60,min_periods=40).cov(r).div(dr.rolling(60,min_periods=40).var(),axis=0)
# cross-sectional factor: assets expected to benefit from dollar weakness score higher
fac=-(beta.mul(dr.rolling(5,min_periods=5).sum(),axis=0))
# avoid same day: signal at t uses close t, forward starts t+1
for h in [1,5,10]:
    fwd=p.shift(-h).div(p)-1
    vals=[]; dates=[]; ns=[]
    for dt in p.index:
        x=fac.loc[dt]; y=fwd.loc[dt]
        z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
    q=pd.Series(vals,index=dates)
    print(h,'dates',len(q),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5),'hit',round((q>0).mean(),4))
# turnover ranks
rr=fac.rank(axis=1,pct=True)
t=(rr.diff().abs().mean(axis=1)>0.15).mean()
print('turnover_proxy',round(t,4),'valid coverage',round(fac.notna().sum(axis=1).mean()/15,4))
print('regimes',[(str(s),round(pd.Series(vals,index=dates).loc[s:e].mean(),4),len(pd.Series(vals,index=dates).loc[s:e])) for s,e in [('2020','2021-12-31'),('2022','2023-12-31'),('2024','2025-12-31'),('2026','2026-07-15')]])
