import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); P[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill(); r=np.log(p).diff(); rv=r.rolling(60,min_periods=30).std()
shock=r.rolling(3,min_periods=3).sum(); base=-shock*(shock.abs()/(rv*np.sqrt(3)))
# A persistent high-dispersion regime: 5-day mean dispersion above its trailing 60-day median.
disp=r.std(axis=1); gate=(disp.rolling(5,min_periods=5).mean()>disp.rolling(60,min_periods=30).median()).astype(float)
f=base.mul(gate,axis=0); f=f.sub(f.mean(axis=1),axis=0)
out=[]; sig=[]
for dt in f.index:
 x=f.loc[dt]; y=r.shift(-1).loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: out.append((dt,x[ok].corr(y[ok],method='spearman'),ok.sum()))
 sig += [(dt,s,float(f.loc[dt].get(s,np.nan))) for s in U]
a=pd.DataFrame(out,columns=['date','ic','n']).dropna()
print('dates',len(a),'avg_n',a.n.mean(),'avg_valid_coverage',a.n.mean()/len(U),'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2029)]:
 q=a[(a.date.dt.year>=lo)&(a.date.dt.year<=hi)]; print('regime',lo,hi,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
for h in [3,5,10]:
 z=[]; yall=np.log(p.shift(-h)/p)
 for dt in f.index:
  x=f.loc[dt]; y=yall.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:z.append(x[ok].corr(y[ok],method='spearman'))
 z=pd.Series(z).dropna(); print('horizon',h,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1))
rank=f.rank(pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean())
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20290823_persistent_dispersion_shockrev_signal.csv',index=False)
