import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:'2027-02-24']
R=P.pct_change(); f=-(P.pct_change(3).shift(1)/R.rolling(20).std().shift(1)); out={}
for h in [1,3,5]:
 yy=P.pct_change(h).shift(-h); a=[];ds=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q);ds.append(dt);ns.append(len(z))
 a=np.array(a); print('h',h,'dates',len(a),'n',np.mean(ns),'cov',np.mean(ns)/15,'ic',a.mean(),'icir',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2025-12-31'),('2026','2027-02-24')]:
   z=a[(np.array(ds)>=pd.Timestamp(lo))&(np.array(ds)<=pd.Timestamp(hi))];print(lo,len(z),z.mean() if len(z) else np.nan,z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
rnk=f.rank(axis=1,pct=True);print('turn',rnk.diff().abs().mean(axis=1).mean())
f.stack().rename('signal').to_csv('scripts/miner_1_20270225_volnorm_rev3_signal.csv',header=True)
