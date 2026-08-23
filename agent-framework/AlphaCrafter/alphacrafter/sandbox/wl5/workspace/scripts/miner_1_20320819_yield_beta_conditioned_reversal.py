import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d['date']); xs[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(xs).sort_index().ffill(); r=p.pct_change()
# US10Y yield series is an available tradable benchmark and is used only as conditioning input.
m=r['US10Y']
rows=[]; artifacts=[]
for i in range(100,len(p)-10):
 dt=p.index[i]; rr=r.iloc[:i+1]; mw=m.iloc[:i+1].iloc[-60:]
 w=rr.iloc[-60:].loc[mw.index]; mw=mw[w.index]; var=mw.var()
 beta=w.apply(lambda x:x.cov(mw)/var if var>1e-12 else np.nan)
 vol=rr.iloc[-20:].std()*np.sqrt(252)
 rev=-(p.iloc[i]/p.iloc[i-10]-1)
 base=rev/vol.replace(0,np.nan)
 ab=beta.abs(); fac=base*(1-0.25*(ab-ab.median())/(ab.std()+1e-9))
 fwd=p.iloc[i+10]/p.iloc[i]-1
 z=pd.DataFrame({'f':fac,'y':fwd}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  ic=z.f.corr(z.y); rows.append((dt,float(ic),len(z)))
  for s,v in fac.items():
   if s in z.index: artifacts.append({'date':dt,'symbol':s,'signal':float(v),'ic':float(ic)})
ics=np.array([x[1] for x in rows]); print('dates',len(rows),'mean_n',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2] for x in rows])/15)
print('IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',np.mean(ics>0))
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2032-08-18')]:
 q=np.array([x[1] for x in rows if pd.Timestamp(a)<=x[0]<=pd.Timestamp(b)])
 print(a,b,'dates',len(q),'IC',q.mean() if len(q) else np.nan,'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
out=pd.DataFrame(artifacts); out.to_csv('scripts/miner_1_20320819_yield_beta_conditioned_reversal_signal.csv',index=False)
print('artifact_rows',len(out),'last_date',out.date.max() if len(out) else None)
