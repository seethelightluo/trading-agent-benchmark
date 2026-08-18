import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,days=5000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index(); D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# Compression strength: lower recent realized volatility relative to medium-term volatility.
f=(-r.rolling(20).std()/(r.rolling(60).std()+1e-12)).shift(1)
for h in [5,10,20]:
    fw=p.shift(-h)/p-1; ics=[]; ns=[]; turns=[]; prev=None
    for dt in sorted(set(f.index)&set(fw.index)):
        z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
        if len(z)<8: continue
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        if pd.notna(ic): ics.append(ic); ns.append(len(z))
        rank=f.loc[dt].rank(pct=True)
        turns.append((rank-prev).abs().mean() if prev is not None else np.nan); prev=rank
    q=pd.Series(ics); recent=q.tail(252)
    print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(np.nanmean(turns),4),'recent252',round(recent.mean(),6),'recentIR',round(recent.mean()/recent.std(ddof=1),6))
print('assets',len(D),'dates',len(p),'coverage',round(f.notna().sum(axis=1).mean()/len(U),4))
