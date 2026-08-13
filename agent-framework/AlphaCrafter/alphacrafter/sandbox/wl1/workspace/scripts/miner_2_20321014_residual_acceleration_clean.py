import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=3200) for s in U}
# Clean invalid/non-positive prices before returns; retain available universe.
cols={s:d.set_index('date').close.rename(s) for s,d in D.items() if d is not None and len(d)>0}
p=pd.concat(cols,axis=1).sort_index().replace([np.inf,-np.inf],np.nan)
p=p.where(p>0).ffill(limit=5)
r=p.pct_change().replace([np.inf,-np.inf],np.nan)
m20=p.pct_change(20); m60=p.pct_change(60)
raw=m20-m60/3
res=raw.sub(raw.mean(axis=1),axis=0)
vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
f=(res/vol).replace([np.inf,-np.inf],np.nan)
fwd=(p.shift(-10)/p-1).replace([np.inf,-np.inf],np.nan)
rows=[]; sig=[]
for dt in p.index:
 z=pd.concat([f.loc[dt].rename('f'),fwd.loc[dt].rename('r')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  rows.append((dt,z.f.corr(z.r),len(z))); sig.append(f.loc[dt].rename(dt))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna()
print('clean residual acceleration 20v60; dates',len(o),'avg_n',round(o.n.mean(),2),'assets',len(U),'coverage',round(f.notna().stack().mean(),5))
print('IC',round(o.ic.mean(),8),'ICIR',round(o.ic.mean()/o.ic.std(ddof=1),8),'hit',round((o.ic>0).mean(),5))
for a,b in [('2020-01-01','2023-12-31'),('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2032-10-14')]:
 q=o.loc[a:b]; print(a[:4]+'-'+b[:4],len(q),round(q.ic.mean(),8) if len(q) else None,round(q.ic.mean()/q.ic.std(ddof=1),8) if len(q)>1 else None,round((q.ic>0).mean(),5) if len(q) else None)
ss=pd.DataFrame(sig).reindex(columns=U); print('rank_turnover',round(ss.rank(pct=True).diff().abs().mean(axis=1).mean(),8)); ss.to_csv('scripts/miner_2_20321014_residual_acceleration_clean_signal.csv')
