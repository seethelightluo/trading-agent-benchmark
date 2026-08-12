import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    d=get_stock_daily_data(s,days=3200)
    if d is None or len(d)<120: d=get_index_daily_data(s,days=3200)
    if d is not None and len(d)>=120:
        D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Downside-adjusted medium-term momentum: trailing net return scaled by downside deviation.
down=r.clip(upper=0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
f=r.rolling(20,min_periods=15).sum()/(down*np.sqrt(20)+1e-12)
# Cross-sectional demean and lag one completed bar.
f=f.sub(f.median(axis=1),axis=0).shift(1)
print('rows',len(p),'assets',len(D),'coverage',round(f.notna().mean().mean(),4))
for h in [1,5,10,20]:
    y=np.log(p).shift(-h)-np.log(p); a=[]; ns=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
            if np.isfinite(q): a.append(q); ns.append(len(z))
    a=np.array(a)
    print('h',h,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)/2).mean(),6))
for label,lo in [('2027+','2027-01-01'),('2028YTD','2028-01-01')]:
    y=np.log(p).shift(-10)-np.log(p); a=[]
    for dt in f.loc[lo:].index:
        z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
            if np.isfinite(q): a.append(q)
    a=np.array(a); print(label,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
f.to_csv('scripts/miner_2_20281102_downside_momentum_signal.csv')
