import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<160: continue
 d=d.sort_values('date').copy(); c=d.close.astype(float); r=c.pct_change()
 # lagged medium-term trend, scaled by recent risk; all inputs end before decision day
 vol=r.rolling(30).std().shift(1)*np.sqrt(30)
 sig=(c.shift(1)/c.shift(121)-1)/vol.replace(0,np.nan)
 fwd=c.shift(-10)/c-1
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'signal':sig,'fwd':fwd}).dropna())
all=pd.concat(rows,ignore_index=True); all=all[all.date<='2029-09-05']
ics=[]
for dt,g in all.groupby('date'):
 if len(g)>=8 and g.signal.nunique()>1 and g.fwd.nunique()>1:
  ics.append((dt,g.signal.corr(g.fwd),len(g)))
z=pd.DataFrame(ics,columns=['date','ic','n']); mu=z.ic.mean(); sd=z.ic.std(ddof=1)
print('dates',len(z),'avg_instruments',z.n.mean(),'coverage',len(all)/(all.date.nunique()*15),'ic',mu,'icir',mu/sd,'hit',(z.ic>0).mean())
for label,a,b in [('2020-2022','2020-01-01','2022-12-31'),('2023-2025','2023-01-01','2025-12-31'),('2026-2027','2026-01-01','2027-12-31'),('2028-2029','2028-01-01','2029-09-05'),('recent180','2029-02-27','2029-09-05')]:
 q=z[(z.date>=a)&(z.date<=b)]; print(label,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
# rank signal turnover proxy
wide=all.pivot(index='date',columns='symbol',values='signal'); ranks=wide.rank(pct=True); print('turnover_proxy',ranks.diff().abs().mean().mean())
all[['date','symbol','signal']].to_csv('scripts/miner_2_20290906_voladjusted_momentum_120d_signal.csv',index=False)
