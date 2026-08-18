import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index().loc[:'2035-10-11']
ret=p.pct_change(); rollmax=p.rolling(60,min_periods=40).max(); dd=(p/rollmax-1).abs().rolling(20,min_periods=10).mean()
sig=(p.pct_change(10)/dd.replace(0,np.nan)).shift(1); rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],(p.shift(-10)/p-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); turn=sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna()
print('cutoff',p.index.max().date(),'dates',len(r),'avg_n',r.n.mean(),'coverage',len(r)/len(sig),'IC10',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean(),'turn',turn.mean())
for a,b in [('2020','2024-12-31'),('2025','2030-12-31'),('2031','2035-10-11'),('2035-04-01','2035-10-11')]:
 q=r.loc[a:b].ic; print('regime',a,b,'n',len(q),'ic',q.mean(),'icir',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [1,5,10,20]:
 rr=[]; yy=p.shift(-h)/p-1
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.nanmean(rr),len(rr))
out=sig.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_2_20351012_drawdown_recovery_signal.csv')
