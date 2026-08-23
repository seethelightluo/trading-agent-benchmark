import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
cut=pd.Timestamp('2031-08-20'); px=pd.concat(D,axis=1).sort_index().loc[:cut]
r=px.pct_change(); vol=r.rolling(60,min_periods=60).std(); f=-(px/px.shift(20)-1)/vol.replace(0,np.nan)
for h in [5,10,20]:
 I=[];N=[];ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],(px.shift(-h).loc[d]/px.loc[d]-1)],axis=1).dropna()
  if len(z)>=8:
   x=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(x):I.append(x);N.append(len(z));ds.append(d)
 a=np.array(I); print(h,'dates',len(a),'avg_n',round(np.mean(N),3),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',np.mean([f.loc[d].notna().sum() for d in ds])/15)
 if h==10:
  q=f.rank(axis=1,pct=True); print('turnover',q.diff().abs().mean(axis=1).loc[ds].mean())
  for y in sorted(set(d.year for d in ds)):
   b=a[[d.year==y for d in ds]];print(y,round(b.mean(),6),len(b))
