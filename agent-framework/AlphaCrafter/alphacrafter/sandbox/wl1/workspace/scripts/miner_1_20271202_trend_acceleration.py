import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
    d=get_stock_daily_data(s,days=4000)
    if d is None: d=get_index_daily_data(s,days=4000)
    if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change()
# Candidate: acceleration of medium trend, volatility-normalized. Uses only information at t and lags one session.
acc=(P/P.shift(20)-1)-((P/P.shift(60)-1)/3.0)
vol=r.rolling(20).std()
f=(acc/(vol+0.003)).shift(1)
print('assets',len(px),'dates',len(P))
for h in [5,10,20]:
    vals=[]; ns=[]
    for i in range(len(P)-h):
        z=pd.concat([f.iloc[i],(P.iloc[i+h]/P.iloc[i]-1)],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
    a=np.asarray(vals); mu=np.nanmean(a); sd=np.nanstd(a,ddof=1)
    print('h',h,'dates',len(a),'avgN',np.mean(ns),'coverage',np.mean(ns)/len(U),'IC',mu,'ICIR',mu/sd,'hit',np.mean(a>0))
rank=f.rank(axis=1,pct=True)
print('turnover',((rank-rank.shift(1)).abs().mean(axis=1)).mean())
for label,start in [('2025','2025-01-01'),('2026','2026-01-01'),('2027','2027-01-01')]:
    vals=[]; h=20
    for i in range(len(P)-h):
        if P.index[i]<pd.Timestamp(start): continue
        z=pd.concat([f.iloc[i],(P.iloc[i+h]/P.iloc[i]-1)],axis=1).dropna()
        if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    a=np.asarray(vals); print(label,'dates',len(a),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1))
f.to_csv('scripts/miner_1_20271202_trend_acceleration_signal.csv')
