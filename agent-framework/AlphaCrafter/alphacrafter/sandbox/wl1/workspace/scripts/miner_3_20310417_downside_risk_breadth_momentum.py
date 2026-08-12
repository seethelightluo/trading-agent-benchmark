import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); P[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill(); r=np.log(p).diff(); mom=np.log(p).diff(20); rev=-np.log(p).diff(5)
# Stable downside deviation: zero-fill positive returns, then rolling RMS; avoids sparse rolling-std NaNs.
neg2=(r.clip(upper=0)**2); down=np.sqrt(neg2.rolling(40,min_periods=20).mean())
trend=(p>p.rolling(60,min_periods=40).mean()).astype(float); breadth=trend.mean(axis=1)
raw=(mom+0.35*rev)/down.replace(0,np.nan)
sig0=raw.mul(0.65+0.70*breadth,axis=0)*trend.replace(0,np.nan); sig=sig0.shift(1).rank(axis=1,pct=True)
rows=[]
for dt in sig.index:
 i=p.index.get_loc(dt)
 for h in [1,5,10,20]:
  if i+h>=len(p): continue
  y=p.iloc[i+h]/p.iloc[i]-1; z=pd.concat([sig.loc[dt].rename('x'),y.rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,z.x.corr(z.y,method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['date','h','ic','n']); o.date=pd.to_datetime(o.date)
print('dates',o.date.nunique(),'assets',p.shape[1],'avgN',o.groupby('date').n.first().mean(),'coverage',sig.notna().mean().mean(),'turnover',sig.diff().abs().mean().mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first(); print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for a,b in [(2020,2022),(2023,2025),(2026,2028),(2029,2030),(2031,2031)]:
 g=o[(o.h==10)&(o.date.dt.year>=a)&(o.date.dt.year<=b)]
 if len(g): print('period',a,b,'IC %.6f ICIR %.6f n=%d'%(g.ic.mean(),g.ic.mean()/g.ic.std(ddof=1),len(g)))
sig.to_csv('scripts/miner_3_20310417_downside_risk_breadth_momentum_signal.csv')
