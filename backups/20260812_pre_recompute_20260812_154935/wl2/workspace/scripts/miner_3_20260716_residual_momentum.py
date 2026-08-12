import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U}).sort_index(); r=p.pct_change(); m=r.median(axis=1)
for w in [20,60,120]:
 f=pd.DataFrame(index=r.index,columns=U,dtype=float)
 for i in range(w,len(r)):
  x=r.iloc[i-w:i]; bm=m.iloc[i-w:i]; vb=bm.var()
  if vb>0:
   beta=x.apply(lambda z:z.cov(bm)/vb); f.iloc[i]=x.sum()-beta*bm.sum()
 for h in [1,5,10]:
  ic=[]; ns=[]; dates=[]
  for i in range(len(r)-h):
   q=pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(q)>=8: ic.append(spearmanr(q.iloc[:,0],q.y).statistic);ns.append(len(q));dates.append(f.index[i])
  x=np.array(ic); print('w',w,'h',h,'dates',len(x),'meanN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,3),'IC',round(x.mean(),5),'ICIR',round(x.mean()/x.std(ddof=1),5),'hit',round((x>0).mean(),4))
  if h==1:
   for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
    z=x[(pd.DatetimeIndex(dates).year>=lo)&(pd.DatetimeIndex(dates).year<=hi)];print('reg',lo,round(z.mean(),5),round(z.mean()/z.std(ddof=1),5),len(z))
 # rank turnover
 ranks=f.rank(axis=1,pct=True); print('turn',np.nanmean(np.abs(ranks.diff()).mean(axis=1)))
