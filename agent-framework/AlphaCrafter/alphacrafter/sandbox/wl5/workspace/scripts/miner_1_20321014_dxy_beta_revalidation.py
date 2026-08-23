import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); xs[s]=d.set_index('date').close.astype(float)
pdprice=pd.DataFrame(xs).sort_index().ffill(); r=pdprice.pct_change()
macro=pd.read_csv('../persistent/index_data/DXY.csv'); macro.date=pd.to_datetime(macro.date)
mc=macro.set_index('date').close.astype(float).sort_index().reindex(pdprice.index).ffill().pct_change()
rows=[]; artifacts=[]
for i in range(100,len(pdprice)-10):
 dt=pdprice.index[i]; rr=r.iloc[:i+1]; m=mc.iloc[:i+1].iloc[-60:]
 w=rr.iloc[-60:].copy(); valid=m.notna(); m=m[valid]; w=w.loc[m.index]
 var=m.var(); beta=w.apply(lambda x:x.cov(m)/var if var>1e-12 else np.nan)
 vol=rr.iloc[-20:].std()*np.sqrt(252); rev=-pdprice.iloc[i]/pdprice.iloc[i-10]-1
 sig=rev/vol.replace(0,np.nan); ab=beta.abs(); fac=sig*(1-(ab-ab.median())/(ab.std()+1e-9)*.25)
 fwd=pdprice.iloc[i+10]/pdprice.iloc[i]-1; z=pd.DataFrame({'f':fac,'y':fwd}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  ic=z.f.corr(z.y); rows.append((dt,ic,len(z)))
  for s,v in fac.items(): artifacts.append({'date':dt,'symbol':s,'signal':v,'ic':ic})
ics=np.array([x[1] for x in rows]); print('cutoff',pdprice.index[-11].date(),'dates',len(rows),'mean_n',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2] for x in rows])/15); print('IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',np.mean(ics>0),'turnover','not_computed')
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-10-03')]:
 q=np.array([x[1] for x in rows if pd.Timestamp(a)<=x[0]<=pd.Timestamp(b)]); print('regime',a,b,'n',len(q),'IC',q.mean() if len(q) else np.nan,'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
pd.DataFrame(artifacts).to_csv('scripts/miner_1_20321014_dxy_beta_revalidation_signal.csv',index=False)
