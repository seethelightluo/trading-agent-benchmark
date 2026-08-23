import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for s in U:
 p=os.path.join(base,s+'.csv')
 if os.path.exists(p):
  d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date'); px[s]=d['close'].astype(float)
prices=pd.DataFrame(px).sort_index().loc[:'2030-12-11']
ret=prices.pct_change(); r20=prices/prices.shift(20)-1
neg=ret.where(ret<0); dsv=neg.rolling(40,min_periods=20).std()*np.sqrt(252)
raw=(-r20).clip(-8,8); basef=(-r20/(dsv+1e-8)).clip(-8,8)
def run(h):
  ics=[]; ns=[]; ds=[]
  for i in range(60,len(prices)-h):
   dt=prices.index[i]; y=basef.loc[dt]; x=pd.DataFrame({'raw':raw.loc[dt],'vol':dsv.loc[dt]}); fr=prices.iloc[i+h].div(prices.iloc[i])-1
   z=pd.concat([y.rename('y'),x,fr.rename('f')],axis=1).dropna()
   if len(z)<8: continue
   X=np.column_stack([np.ones(len(z)),z.raw.rank(pct=True),z.vol.rank(pct=True)])
   res=z.y.values-X@np.linalg.lstsq(X,z.y.values,rcond=None)[0]
   ics.append(spearmanr(res,z.f).statistic); ns.append(len(z)); ds.append(dt)
  s=pd.Series(ics,index=ds).dropna(); return s,ns
for h in [5,10,20]:
 s,ns=run(h); print('H',h,'dates',len(s),'avg_n',round(np.mean(ns),3),'IC',round(s.mean(),8),'ICIR',round(s.mean()/s.std()*np.sqrt(252),5),'hit',round((s>0).mean(),4),'coverage',round(sum(ns)/(len(s)*len(U)),5))
s,ns=run(20); print('regime',{str(y):round(s[s.index.year==y].mean(),6) for y in sorted(s.index.year.unique())}); print('turnover_proxy',round(np.nanmean([np.mean(np.abs(s.iloc[i]-s.iloc[i-1])) for i in range(1,len(s))]),6))
