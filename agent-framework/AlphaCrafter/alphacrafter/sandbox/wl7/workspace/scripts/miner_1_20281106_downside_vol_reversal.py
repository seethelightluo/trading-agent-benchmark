import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=2600)
 if x is not None and len(x):
  z=x.copy(); z['date']=pd.to_datetime(z['date']); D[s]=z.set_index('date')['close'].astype(float).sort_index()
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
sig=-(p/p.shift(10)-1)/(r.where(r<0).rolling(30,min_periods=15).std()+1e-8); fwd=p.shift(-20)/p-1
ics=[]; turns=[]; cov=[]; counts=[]
for d in sig.index:
 a=sig.loc[d]; b=fwd.loc[d]; ok=a.notna()&b.notna(); n=int(ok.sum()); counts.append(n); cov.append(n/15)
 if n>=8:
  ics.append(a[ok].corr(b[ok])); prev=sig.shift(1).loc[d]; turns.append((a[ok].rank(pct=True)-prev[ok].rank(pct=True)).abs().mean())
ics=pd.Series(ics).dropna(); good=[(n,c) for n,c in zip(counts,cov) if n>=8]
print('candidate downside-vol normalized reversal 10d/20d'); print('dates',len(ics),'avg instruments',np.mean([n for n,c in good]),'coverage',np.mean([c for n,c in good])); print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(ics.mean(),ics.mean()/ics.std(ddof=1),(ics>0).mean(),np.nanmean(turns)))
for lo,hi in [('2025-01-01','2026-12-31'),('2027-01-01','2028-11-05')]:
 q=ics.loc[lo:hi]; print(lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean())
for h in [5,10,20,40]:
 z=[]
 for d in sig.index:
  a=sig.loc[d]; b=p.shift(-h).loc[d]/p.loc[d]-1; ok=a.notna()&b.notna()
  if ok.sum()>=8:z.append(a[ok].corr(b[ok]))
 z=pd.Series(z).dropna(); print('horizon',h,'n',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20281106_downside_vol_reversal_signal.csv',index=False)
