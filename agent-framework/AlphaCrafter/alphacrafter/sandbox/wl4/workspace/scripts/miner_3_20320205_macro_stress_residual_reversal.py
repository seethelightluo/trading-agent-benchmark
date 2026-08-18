import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d['date']); px[s]=d.set_index('date')['close'].astype(float)
prices=pd.DataFrame(px).sort_index().ffill(); ret=np.log(prices/prices.shift(1))
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix['date']=pd.to_datetime(vix['date']); vix=vix.set_index('date')
vc=next(c for c in ['close','Close','value','VIX'] if c in vix.columns); v=vix[vc].astype(float).reindex(prices.index).ffill()
m=ret.mean(axis=1); res=ret.rolling(10).sum().sub(m.rolling(10).sum(),axis=0); vol=ret.rolling(40).std(); base=-res/vol
vz=(v-v.rolling(252).median())/(v.rolling(252).std()+1e-12); stress=vz.clip(-2,2)
variants={'macro_stress':base.mul(1+0.35*stress,axis=0),'macro_calm':base.mul(1-0.35*stress,axis=0),'macro_binary':base.mul((1+0.5*(stress>0).astype(float)),axis=0),'base':base}
fwd=prices.shift(-10)/prices-1
for name,x in variants.items():
 x=x.shift(1); ics=[]; turnovers=[]; ns=[]; prev=None
 for dt in x.index:
  a=x.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna(); n=len(z)
  if n>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(n)
  ranks=a.rank(pct=True)
  if prev is not None:
   q=pd.concat([ranks,prev],axis=1).dropna(); turnovers.append((q.iloc[:,0]-q.iloc[:,1]).abs().mean())
  prev=ranks
 ic=np.asarray(ics); ic=ic[np.isfinite(ic)]
 print(name,'dates',len(ic),'avgN',round(np.mean(ns),2),'coverage',round(len(ic)/(len(x.index)-1),4),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/(ic.std(ddof=1)+1e-12),6),'hit',round((ic>0).mean(),4),'turn',round(np.mean(turnovers),4))
 for days in [365,730,1095]:
  sub=ic[-days:]; print(' recent',days,round(sub.mean(),6),round(sub.mean()/(sub.std(ddof=1)+1e-12),6),len(sub))
