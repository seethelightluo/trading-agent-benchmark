import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv(Path('../persistent/stock_data')/(a+'.csv'),parse_dates=['date']).set_index('date')['close'] for a in A}
p=pd.DataFrame(D).ffill().loc[:'2027-06-30']; r=p.pct_change()
# lagged 30d return divided by trailing downside deviation, no look-ahead
neg=r.where(r<0,0.0)
dd=np.sqrt((neg**2).rolling(30,min_periods=20).mean())
f=(p.pct_change(30)/(dd+1e-12)).shift(1)
y=p.shift(-10).div(p)-1
vals=[]; ns=[]; dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
s=pd.Series(vals,index=dates).dropna()
print('period',p.index.min().date(),p.index.max().date(),'assets',len(A),'dates',len(s),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4))
print('IC',round(s.mean(),5),'ICIR',round(s.mean()/s.std(ddof=1),5),'hit',round((s>0).mean(),4))
for nm,(a,b) in {'2020-22':('2020','2022-12-31'),'2023-24':('2023','2024-12-31'),'2025-26':('2025','2026-12-31'),'2027 YTD':('2027','2027-06-30'),'recent90':('2027-03-01','2027-06-30')}.items():
 q=s.loc[a:b]; print(nm,'dates',len(q),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5) if len(q)>1 else np.nan,'hit',round((q>0).mean(),4) if len(q) else np.nan)
rank=f.rank(axis=1,pct=True); print('turnover',round(rank.diff().abs().mean().mean(),5))
