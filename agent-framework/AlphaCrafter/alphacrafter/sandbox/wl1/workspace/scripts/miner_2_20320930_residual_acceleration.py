import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=3000) for s in U}
p=pd.concat({s:d.set_index('date').close for s,d in D.items() if d is not None},axis=1).sort_index().ffill()
r=p.pct_change(); fwd=p.pct_change(10).shift(-10)
# residual acceleration: recent 20d return minus slow 60d return, neutralized cross-section and scaled by trailing 20d volatility
m20=p.pct_change(20); m60=p.pct_change(60)
raw=m20-m60/3
res=raw.sub(raw.mean(axis=1),axis=0)
vol=r.rolling(20).std()*np.sqrt(252)
f=res/vol
rows=[]; sig=[]
for dt in p.index:
 z=pd.concat([f.loc[dt].rename('f'),fwd.loc[dt].rename('r')],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.f.corr(z.r),len(z))); sig.append(f.loc[dt].rename(dt))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('factor residual acceleration 20v60; dates',len(o),'avg_n',o.n.mean(),'coverage',f.notna().stack().mean())
print('IC',o.ic.mean(),'ICIR',o.ic.mean()/o.ic.std(ddof=1),'hit',(o.ic>0).mean())
for a,b in [('2020-01-01','2023-12-31'),('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2032-09-30')]:
 q=o.loc[a:b]; print(a[:4]+'-'+b[:4],len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan,(q.ic>0).mean() if len(q) else np.nan)
ss=pd.DataFrame(sig); print('rank_turnover',ss.rank(pct=True).diff().abs().mean(axis=1).mean()); ss.to_csv('scripts/miner_2_20320930_residual_acceleration_signal.csv')
