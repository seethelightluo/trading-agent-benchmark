import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,3200)
 if d is None or len(d)<200:d=get_index_daily_data(s,3200)
 if d is not None and len(d)>200:
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill();r=np.log(p).diff()
# Quality trend: 20-day return weighted by directional consistency, risk-normalized.
cons=(r>0).rolling(20).mean()*2-1
f=r.rolling(20).sum()*cons/(r.rolling(60).std()*np.sqrt(20)+1e-12)
f=f.replace([np.inf,-np.inf],np.nan);f=f.sub(f.mean(axis=1),axis=0)
def run(h):
 y=np.log(p.shift(-h)/p);o=[]
 for dt in f.index:
  x=f.loc[dt];z=y.loc[dt];ok=x.notna()&z.notna()
  if ok.sum()>=8:o.append((dt,x[ok].corr(z[ok],method='spearman'),ok.sum()))
 return pd.DataFrame(o,columns=['date','ic','n']).dropna()
for h in [1,3,5,10]:
 a=run(h);print('H',h,'dates',len(a),'avgN',a.n.mean(),'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
 if h==5:
  for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2029)]:
   q=a[(a.date.dt.year>=lo)&(a.date.dt.year<=hi)];print('REG',lo,hi,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
print('coverage',f.notna().sum().sum()/(f.shape[0]*15),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'assets',len(P),'dates',len(f))
f.to_csv('scripts/miner_1_20290920_consistency_trend_signal.csv',index_label='date')
