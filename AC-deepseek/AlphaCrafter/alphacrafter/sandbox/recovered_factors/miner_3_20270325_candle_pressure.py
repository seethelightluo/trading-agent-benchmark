import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; F={}; R={}
for s in A:
 d=get_stock_daily_data(s,5000).set_index('date'); d.index=pd.to_datetime(d.index)
 c=pd.to_numeric(d.close,errors='coerce'); h=pd.to_numeric(d.high,errors='coerce'); l=pd.to_numeric(d.low,errors='coerce'); o=pd.to_numeric(d.open,errors='coerce')
 # directional candle pressure: close location, but penalize wide ranges; smoothed and lagged
 cl=((2*c-h-l)/(h-l).replace(0,np.nan)).clip(-1,1)
 rng=(h-l)/c.replace(0,np.nan)
 F[s]=(cl/(1+rng*10)).rolling(10,min_periods=8).mean()
 R[s]=c.pct_change()
f=pd.concat(F,axis=1); ret=pd.concat(R,axis=1)
print('FACTOR candle_pressure_10 dates',len(f),'assets',len(A))
def ev(x,h):
 fw=(1+ret).rolling(h,min_periods=h).apply(np.prod,raw=True).shift(-h); q=[]; ns=[]
 for dt in x.index:
  z=pd.concat([x.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 q=pd.Series(q);return len(q),q.mean(),q.mean()/q.std(ddof=1),float((q>0).mean()),np.mean(ns)
for h in [1,5,10,20]:print('h',h,ev(f,h))
for nm,m in [('2020',f.index.year==2020),('2021_22',(f.index.year>=2021)&(f.index.year<=2022)),('2023_24',(f.index.year>=2023)&(f.index.year<=2024)),('2025_26',f.index.year>=2025)]:print('REGIME',nm,ev(f.loc[m],1))
print('coverage',f.notna().stack().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(1).mean())
for fid in ['ravmom','trend','shortrev','vol','relvol','vixrev']:
 E={}
 for s in A:
  r=R[s]
  if fid=='ravmom':E[s]=r.rolling(20).sum()
  elif fid=='trend':E[s]=r.rolling(20).sum()/r.rolling(20).std()
  elif fid=='shortrev':E[s]=-r.rolling(5).sum()/r.rolling(10).std()
  elif fid=='vol':E[s]=-r.rolling(20).std()
  elif fid=='relvol':E[s]=r.rolling(5).std()/r.rolling(20).std()
  else:E[s]=-r.rolling(1).sum()
 e=pd.concat(E,axis=1);z=pd.concat([f.stack(),e.stack()],axis=1).dropna();print('LIBCORR',fid,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
