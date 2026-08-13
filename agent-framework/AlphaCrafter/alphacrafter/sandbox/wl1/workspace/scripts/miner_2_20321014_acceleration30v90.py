import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=3200) for s in U}
p=pd.concat({s:d.set_index('date').close.rename(s) for s,d in D.items() if d is not None},axis=1).sort_index().replace([np.inf,-np.inf],np.nan)
p=p.where(p>0).ffill(limit=5); r=p.pct_change().replace([np.inf,-np.inf],np.nan)
raw=p.pct_change(30)-p.pct_change(90)/3
f=raw.sub(raw.mean(axis=1),axis=0)/(r.rolling(30,min_periods=20).std()*np.sqrt(252)); f=f.replace([np.inf,-np.inf],np.nan)
fw=(p.shift(-10)/p-1).replace([np.inf,-np.inf],np.nan); rows=[]; sig=[]
for dt in p.index:
 z=pd.concat([f.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.f.corr(z.r),len(z))); sig.append(f.loc[dt].rename(dt))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna()
print('30v90 residual acceleration dates',len(o),'avg_n',round(o.n.mean(),2),'coverage',round(f.notna().stack().mean(),5))
print('IC',round(o.ic.mean(),8),'ICIR',round(o.ic.mean()/o.ic.std(ddof=1),8),'hit',round((o.ic>0).mean(),5))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032')]:
 q=o.loc[a:b]; print(a+'-'+b,len(q),round(q.ic.mean(),8),round(q.ic.mean()/q.ic.std(ddof=1),8),(q.ic>0).mean())
ss=pd.DataFrame(sig); print('rank_turnover',round(ss.rank(pct=True).diff().abs().mean(axis=1).mean(),8)); ss.to_csv('scripts/miner_2_20321014_acceleration30v90_signal.csv')
