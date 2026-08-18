import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 D[s]=x
# prior completed session intraday reversal: assets with weak close/open return tend to rebound next close
F=pd.DataFrame({s:-(D[s]['close']/D[s]['open']-1) for s in U})
# suppress tiny/noisy signals via 3-session EW mean, using only through date
F=F.ewm(span=3,min_periods=3).mean()
for h in [1,5,10]:
 Y=pd.DataFrame({s:D[s]['close'].shift(-h)/D[s]['close']-1 for s in U})
 a=[]; ns=[]; ds=[]
 for d in F.index:
  z=pd.concat([F.loc[d],Y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q);ns.append(len(z));ds.append(d)
 a=np.array(a); ds=pd.DatetimeIndex(ds)
 print('h',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
   q=a[(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi))];print('regime',lo,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('coverage',F.notna().mean().mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
print('corr momentum',F.stack().corr(pd.DataFrame({s:D[s].close.pct_change(20) for s in U}).stack()))
