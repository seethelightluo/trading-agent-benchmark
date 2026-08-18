import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,5200)
 if d is None or len(d)==0: d=get_index_daily_data(s,5200)
 if d is not None and len(d): P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=np.log(px).diff()
# Long-horizon drawdown recovery: buy assets with deep 60d drawdowns only when their
# recent 5d return has begun recovering; normalize by idiosyncratic volatility.
peak=px.rolling(60,min_periods=40).max(); dd=px/peak-1
rebound=r.rolling(5,min_periods=5).sum(); vol=r.rolling(20,min_periods=15).std()+1e-9
f=(-dd)*(rebound.clip(lower=0)+0.002)/vol
f=f.shift(1); fw={h:np.log(px.shift(-h)/px) for h in [5,10,20]}
rows=[]; turns=[]
for dt in f.index:
 a=f.loc[dt]; y=fw[10].loc[dt]; ok=a.notna()&y.notna()
 if ok.sum()>=8:
  rows.append((dt,a[ok].corr(y[ok]),ok.sum()))
  if len(rows)>1:
   prev=f.loc[rows[-2][0]]; q=ok&prev.notna()
   turns.append((a[q].rank(pct=True)-prev[q].rank(pct=True)).abs().mean())
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=z.ic.dropna()
print('dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15)
print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(ic.mean(),ic.mean()/ic.std(),(ic>0).mean(),np.nanmean(turns)))
for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 q=ic[(ic.index>=lo)&(ic.index<=hi+'-12-31')];print(lo,hi,len(q),q.mean(),q.mean()/q.std() if q.std()>0 else np.nan,(q>0).mean())
for h in [5,10,20]:
 vals=[]; y=fw[h]
 for dt in f.index:
  a=f.loc[dt];v=y.loc[dt];ok=a.notna()&v.notna()
  if ok.sum()>=8: vals.append(a[ok].corr(v[ok]))
 print('decay',h,np.nanmean(vals),len(vals))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20340428_drawdown_recovery_signal.csv',index=False)
