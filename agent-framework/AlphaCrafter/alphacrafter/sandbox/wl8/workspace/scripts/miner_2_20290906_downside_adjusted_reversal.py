import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<50: continue
 d=d.sort_values('date'); c=d.close.astype(float); r=c.pct_change()
 down=r.where(r<0,0).rolling(20).std().shift(1)*np.sqrt(20)
 # short-horizon overshoot, penalized by downside risk; inverted for reversal discovery
 sig=-(c.shift(1)/c.shift(11)-1)/down.replace(0,np.nan)
 fwd=c.shift(-10)/c-1
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'signal':sig,'fwd':fwd}).dropna())
a=pd.concat(rows,ignore_index=True); a=a[a.date<='2029-09-05']; z=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.signal.nunique()>1 and g.fwd.nunique()>1:z.append((dt,g.signal.corr(g.fwd),len(g)))
z=pd.DataFrame(z,columns=['date','ic','n']); m=z.ic.mean(); sd=z.ic.std(ddof=1)
print('dates',len(z),'avg_instruments',z.n.mean(),'coverage',len(a)/(a.date.nunique()*15),'ic',m,'icir',m/sd,'hit',(z.ic>0).mean())
for lab,a1,b in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026-27','2026-01-01','2027-12-31'),('2028-29','2028-01-01','2029-09-05'),('recent180','2029-02-27','2029-09-05')]:
 q=z[(z.date>=a1)&(z.date<=b)];print(lab,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
a[['date','symbol','signal']].to_csv('scripts/miner_2_20290906_downside_adjusted_reversal_signal.csv',index=False)
