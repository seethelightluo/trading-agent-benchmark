import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U}).sort_index(); r=p.pct_change(); bm=r['SPX']
for w in [20,60,120]:
 f=pd.DataFrame(index=r.index,columns=U,dtype=float)
 # signal at t uses completed returns strictly before t; explicit rolling moments
 for i in range(w+1,len(r)):
  x=r.iloc[i-w:i]; b=bm.iloc[i-w:i]; vb=(b*b).mean()-b.mean()**2
  if vb>1e-12:
   beta=(x.mul(b,axis=0).mean()-x.mean().mul(b.mean()))/vb
   f.iloc[i]=x.sum()-beta*b.sum()
 ic=[]; ns=[]; dates=[]
 for i in range(len(r)-1):
  q=pd.concat([f.iloc[i],(p.iloc[i+1]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8: ic.append(spearmanr(q.iloc[:,0],q.y).statistic); ns.append(len(q)); dates.append(f.index[i])
 x=np.array(ic); print('w',w,'dates',len(x),'meanN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),5),'ICIR',round(x.mean()/x.std(ddof=1),5),'hit',round((x>0).mean(),4),'turn',round(np.nanmean(np.abs(f.rank(axis=1,pct=True).diff()).mean(axis=1)),5))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  z=x[(pd.DatetimeIndex(dates).year>=lo)&(pd.DatetimeIndex(dates).year<=hi)]; print('reg',lo,hi,'ICIR',round(z.mean()/z.std(ddof=1),5),'n',len(z))
 # horizon decay with same signal, non-overlapping forward endpoint
 for h in [5,10]:
  a=[]
  for i in range(len(r)-h):
   q=pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.y).statistic)
  a=np.array(a);print('h',h,'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'dates',len(a))
