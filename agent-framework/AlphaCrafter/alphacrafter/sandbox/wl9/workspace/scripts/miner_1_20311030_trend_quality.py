import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d)>260:
  d=d.copy(); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').sort_index()
c=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); r=c.pct_change()
# Trend-quality factor: medium-horizon return, rewarded for directional consistency and penalized by volatility.
ret=c.pct_change(60); consistency=(np.sign(r).rolling(60,min_periods=40).mean()).abs(); vol=r.rolling(60,min_periods=40).std()
sig=(ret*consistency/(vol+1e-8)).shift(1)
for h in [5,10,20,40,60]:
 fwd=c.shift(-h)/c-1; rows=[]
 for dt in c.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 x=pd.DataFrame(rows,columns=['date','ic','n']); q=x.ic
 print(f'H {h} dates {len(x)} avg_n {x.n.mean():.2f} coverage {x.n.mean()/len(D):.4f} IC {q.mean():.6f} ICIR {q.mean()/q.std(ddof=1):.6f} hit {(q>0).mean():.4f}')
 if h==20:
  for name,a,b in [('2024-26','2024','2026'),('2027-29','2027','2029'),('2030','2030','2030'),('2031','2031','2031')]:
   y=x[(x.date.dt.year>=int(a))&(x.date.dt.year<=int(b))].ic
   print('REG',name,len(y),y.mean(),y.mean()/y.std(ddof=1) if len(y)>1 else np.nan)
# rank turnover
z=sig.rank(axis=1,pct=True); print('TURN',z.diff().abs().mean(axis=1).mean())
out=[]
for dt in sig.index:
 for s in sig.columns:
  if pd.notna(sig.loc[dt,s]): out.append((dt,s,float(sig.loc[dt,s])))
pd.DataFrame(out,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20311030_trend_quality_signal.csv',index=False)
print('ROWS',len(out),'ASSETS',len(D))
