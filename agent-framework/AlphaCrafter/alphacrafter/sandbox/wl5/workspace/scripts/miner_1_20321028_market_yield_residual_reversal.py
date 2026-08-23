import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change(); rows=[]; art=[]
for i in range(100,len(p)-10):
 dt=p.index[i]; rr=r.iloc[:i+1]; w=rr.iloc[-80:]; m=w.mean(axis=1); y=rr['US10Y'].iloc[-80:] if 'US10Y' in rr else pd.Series(index=w.index)
 z=pd.concat([m,y],axis=1).dropna(); X=np.column_stack([np.ones(len(z)),z.iloc[:,0],z.iloc[:,1]])
 fac={}
 for s in U:
  q=rr[s].iloc[-80:].reindex(z.index); ok=q.notna(); zz=z.loc[ok]; qq=q.loc[ok]
  if len(qq)<40: fac[s]=np.nan; continue
  X2=np.column_stack([np.ones(len(zz)),zz.iloc[:,0],zz.iloc[:,1]])
  b=np.linalg.lstsq(X2,qq.values,rcond=None)[0]; resid=qq.values-X2@b
  rv=pd.Series(resid,index=zz.index)
  rev=-rv.iloc[-20:].sum(); vol=rv.iloc[-60:].std()*np.sqrt(252)
  fac[s]=rev/vol if vol>1e-8 else np.nan
 fwd=p.iloc[i+10]/p.iloc[i]-1; q=pd.DataFrame({'f':pd.Series(fac),'y':fwd}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8:
  ic=q.f.corr(q.y); rows.append((dt,ic,len(q)))
  art += [{'date':dt,'symbol':s,'signal':v} for s,v in fac.items()]
ics=np.array([x[1] for x in rows]); print('cutoff',p.index[-11].date(),'dates',len(rows),'mean_n',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2] for x in rows])/15); print('IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',np.mean(ics>0))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-10-27')]:
 q=np.array([x[1] for x in rows if pd.Timestamp(a)<=x[0]<=pd.Timestamp(b)]); print('regime',a,b,'n',len(q),'IC',q.mean() if len(q) else np.nan,'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
pd.DataFrame(art).to_csv('scripts/miner_1_20321028_market_yield_residual_reversal_signal.csv',index=False)
