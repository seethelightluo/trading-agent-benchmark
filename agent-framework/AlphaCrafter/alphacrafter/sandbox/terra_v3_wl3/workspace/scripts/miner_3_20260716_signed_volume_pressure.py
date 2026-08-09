import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={};V={}
for s in U:
 d=get_stock_daily_data(s,days=3000); d=d[['date','close','volume']].copy(); d.date=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date'); P[s]=d.close; V[s]=d.volume
p=pd.DataFrame(P).sort_index(); v=pd.DataFrame(V).reindex(p.index); r=p.pct_change()
# Signed volume pressure: rolling return-volume covariance normalized by volume and return vol.
vs=v.rolling(20,min_periods=15).mean(); signvol=(r*(v/vs).clip(0,10)).rolling(10,min_periods=7).sum()
fac=signvol.sub(signvol.median(axis=1),axis=0)
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in fac.index:
  a=pd.DataFrame({'f':fac.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1: vals.append(a.f.corr(a.r));ns.append(len(a))
 z=pd.Series(vals).dropna(); print('h',h,'dates',len(z),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),(z>0).mean()))
print('coverage',fac.notna().sum().sum()/fac.size,'turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'period',p.index.min(),p.index.max())
