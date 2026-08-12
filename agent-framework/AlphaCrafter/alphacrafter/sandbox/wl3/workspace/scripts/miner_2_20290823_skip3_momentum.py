import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill();r=np.log(p).diff()
# Intermediate horizon momentum, skipping the last 3 sessions, scaled by recent risk.
mom=np.log(p.shift(3)/p.shift(23)); risk=r.rolling(20,min_periods=15).std()*np.sqrt(20); f=mom/risk
f=f.sub(f.mean(axis=1),axis=0); out=[];sig=[]
for dt in f.index:
 x=f.loc[dt];y=r.shift(-1).loc[dt];ok=x.notna()&y.notna()
 if ok.sum()>=8:out.append((dt,x[ok].corr(y[ok],method='spearman'),ok.sum()))
 sig += [(dt,s,float(f.loc[dt].get(s,np.nan))) for s in U]
a=pd.DataFrame(out,columns=['date','ic','n']).dropna();print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/len(U),'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2029)]:
 q=a[(a.date.dt.year>=lo)&(a.date.dt.year<=hi)];print('regime',lo,hi,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
for h in [3,5,10]:
 z=[];yy=np.log(p.shift(-h)/p)
 for dt in f.index:
  x=f.loc[dt];y=yy.loc[dt];ok=x.notna()&y.notna()
  if ok.sum()>=8:z.append(x[ok].corr(y[ok],method='spearman'))
 z=pd.Series(z).dropna();print('horizon',h,len(z),z.mean(),z.mean()/z.std(ddof=1))
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean());pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20290823_skip3_momentum_signal.csv',index=False)
