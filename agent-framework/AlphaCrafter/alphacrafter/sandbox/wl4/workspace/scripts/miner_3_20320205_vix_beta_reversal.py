import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d): px[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index().ffill(); r=np.log(p/p.shift(1)); m=r.mean(axis=1); res=r.rolling(10).sum().sub(m.rolling(10).sum(),axis=0); base=-res/(r.rolling(40).std()+1e-12)
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.set_index('date'); c=next(c for c in ['close','Close','value','VIX'] if c in v); vr=v[c].astype(float).reindex(p.index).ffill().pct_change()
beta=r.rolling(60).cov(vr)/(vr.rolling(60).var()+1e-12); bz=(beta-beta.mean(axis=1).values[:,None])/(beta.std(axis=1).values[:,None]+1e-12)
# cross-sectional beta tilt, sign is interpretable: favor low VIX-beta names during stress
f={'vix_beta_low':base*(1-0.25*bz),'vix_beta_high':base*(1+0.25*bz),'base':base}; fw=p.shift(-10)/p-1
for n,x in f.items():
 ic=[];ns=[]
 for dt in x.index:
  z=pd.concat([x.shift(1).loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(ic); a=a[np.isfinite(a)]; print(n,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
 for q in [365,730,1095]:
  z=a[-q:]; print(q,round(z.mean(),6),round(z.mean()/(z.std(ddof=1)+1e-12),6))
