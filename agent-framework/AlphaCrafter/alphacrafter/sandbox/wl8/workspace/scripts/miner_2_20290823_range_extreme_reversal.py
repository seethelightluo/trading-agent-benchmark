import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s, days=4000)
 if d is None or len(d)<100: continue
 d=d.sort_values('date').copy(); c=d['close'].astype(float); h=d['high'].astype(float); l=d['low'].astype(float)
 r=c.pct_change(); hi=h.rolling(60).max().shift(1); lo=l.rolling(60).min().shift(1); pos=(c.shift(1)-lo)/(hi-lo).replace(0,np.nan)
 sig=(0.5-pos)*2/(r.rolling(20).std().shift(1).replace(0,np.nan)*np.sqrt(20)); fwd=c.shift(-5)/c-1
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'signal':sig,'fwd':fwd}).dropna())
all=pd.concat(rows,ignore_index=True); all=all[all.date<='2029-08-22']; ics=[]
for dt,g in all.groupby('date'):
 if len(g)>=8 and g.signal.nunique()>1 and g.fwd.nunique()>1: ics.append((dt,g.signal.corr(g.fwd),len(g)))
z=pd.DataFrame(ics,columns=['date','ic','n']); mu=z.ic.mean(); sd=z.ic.std(ddof=1)
print('dates',len(z),'avg_instruments',z.n.mean(),'coverage',len(all)/((all.date.nunique())*15),'ic',mu,'icir',mu/sd,'hit',(z.ic>0).mean())
for label,cut in [('2026','2026-12-31'),('2027-29','2027-01-01'),('recent180','2029-02-23')]:
 q=z[z.date>=cut]; print(label,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
all[['date','symbol','signal']].to_csv('scripts/miner_2_20290823_range_extreme_reversal_signal.csv',index=False)
