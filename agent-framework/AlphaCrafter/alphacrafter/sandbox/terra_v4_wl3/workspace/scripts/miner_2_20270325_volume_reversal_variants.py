import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data/'
F={}; idx=None
for s in U:
 p=base+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); F[s]=d
idx=sorted(set().union(*[set(x.index) for x in F.values()])); c=pd.DataFrame({s:F[s].close.reindex(idx) for s in F}); v=pd.DataFrame({s:F[s].volume.reindex(idx) for s in F}); r=c.pct_change(3); vr=(v/v.rolling(20,min_periods=10).median()).clip(.5,3)
for power in [.25,.5,1,1.5]:
 sig=(-r*vr.pow(power)).replace([np.inf,-np.inf],np.nan); out=[]
 for i,dt in enumerate(c.index[:-1]):
  z=pd.concat([sig.loc[dt],c.iloc[i+1]/c.iloc[i]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(out).dropna(); print('power',power,'dates',len(q),'N',sig.notna().sum(axis=1).mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
