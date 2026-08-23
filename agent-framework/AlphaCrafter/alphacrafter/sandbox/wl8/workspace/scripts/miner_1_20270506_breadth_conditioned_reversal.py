import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.concat(P,axis=1).sort_index(); r=p.pct_change()
# Lagged market breadth: strengthen short-term reversal after unusually one-sided cross-asset moves.
breadth=r.gt(0).rolling(5,min_periods=3).mean().mean(axis=1)
imb=(breadth-0.5).abs(); base=imb.rolling(60,min_periods=30).median()
condition=(1+0.8*np.clip(imb/base.replace(0,np.nan)-1,-0.5,1.5)).shift(1)
f=-r.rolling(3,min_periods=3).sum().mul(condition,axis=0)
ics=[]; ns=[]; dates=[]; turns=[]; prev=None
for i in range(65,len(p)-1):
 z=pd.concat([f.iloc[i],r.iloc[i+1]],axis=1).dropna()
 if len(z)>=8:
  x=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic; ics.append(x);ns.append(len(z));dates.append(p.index[i])
  if prev is not None: turns.append(np.mean(np.sign(f.iloc[i].reindex(U))!=np.sign(prev.reindex(U))))
  prev=f.iloc[i]
a=np.array(ics); print('dates',len(a),'rows',sum(ns),'avg_n',np.mean(ns),'coverage',sum(ns)/(len(ns)*15),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'turn',np.mean(turns))
for lo,hi in [(2020,2022),(2023,2025),(2026,2026),(2027,2027)]:
 b=a[[x.year>=lo and x.year<=hi for x in dates]]; print('regime',lo,hi,'n',len(b),'IC',b.mean(),'ICIR',b.mean()/b.std(ddof=1) if len(b)>1 else np.nan)
for h in [5,10]:
 q=[]
 for i in range(65,len(p)-h):
  z=pd.concat([f.iloc[i],p.pct_change(h).iloc[i+h]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q); print('horizon',h,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
