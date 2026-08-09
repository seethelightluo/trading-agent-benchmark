import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut].sort_index() for s in U}
F=pd.DataFrame({s:-x.pct_change().rolling(5,min_periods=5).sum() for s,x in px.items()})
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut].sort_index(); state=(v>v.rolling(60,min_periods=40).median()).astype(float).reindex(F.index).ffill(); F=F.mul(state,axis=0)
for h in [1,5,10]:
 Y=pd.DataFrame({s:px[s].shift(-h)/px[s]-1 for s in U}); a=[];ns=[];ds=[]
 for d in F.index:
  z=pd.concat([F.loc[d],Y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q);ns.append(len(z));ds.append(d)
 a=np.array(a); ds=pd.DatetimeIndex(ds); print(f'h={h} dates={len(a)} meanN={np.mean(ns):.2f} cov={np.mean(ns)/15:.3f} IC={a.mean():.5f} ICIR={a.mean()/a.std(ddof=1):.5f} hit={np.mean(a>0):.3f}')
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
   q=a[(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi))];print('regime',lo,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'stress_frac',state.mean())
