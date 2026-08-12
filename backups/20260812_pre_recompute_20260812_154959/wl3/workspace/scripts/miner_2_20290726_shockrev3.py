import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is not None and len(d)>100:
  x=d.copy();x.date=pd.to_datetime(x.date);P[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill(); r=np.log(p).diff();
# Nonlinear shock reversal: reverse 3-day move, scaled by its absolute size relative to 60d volatility.
rv=r.rolling(60).std(); f=(-r.rolling(3).sum() * (r.rolling(3).sum().abs()/ (rv*np.sqrt(3)))).replace([np.inf,-np.inf],np.nan)
f=f.sub(f.mean(axis=1),axis=0); out=[]; sig=[]
for dt in f.index:
 z=f.loc[dt]; y=r.shift(-1).loc[dt];ok=z.notna()&y.notna()
 if ok.sum()>=8:
  out.append((dt,z[ok].corr(y[ok],method='spearman'),ok.sum()))
  sig += [(dt,s,float(z.get(s,np.nan))) for s in U]
a=pd.DataFrame(out,columns=['date','ic','n']).dropna();print('dates',len(a),'avg_n',a.n.mean(),'coverage',len(sig)/(len(f)*15));print('IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2029)]:
 q=a[(a.date.dt.year>=lo)&(a.date.dt.year<=hi)];print(lo,hi,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
for h in [3,5,10]:
 z=[];yall=np.log(p.shift(-h)/p)
 for dt in f.index:
  x=f.loc[dt];y=yall.loc[dt];ok=x.notna()&y.notna()
  if ok.sum()>=8:z.append(x[ok].corr(y[ok],method='spearman'))
 z=pd.Series(z).dropna();print('h',h,len(z),z.mean(),z.mean()/z.std(ddof=1))
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean());pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20290726_shockrev3_signal.csv',index=False)
