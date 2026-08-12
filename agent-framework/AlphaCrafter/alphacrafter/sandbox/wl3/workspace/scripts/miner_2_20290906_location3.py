import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill();r=np.log(p).diff()
# Three-session range-location reversal: fade recent shock, with larger exposure
# when close is near a 20-session range extreme, normalized by 20-session risk.
hi=p.rolling(20,min_periods=15).max();lo=p.rolling(20,min_periods=15).min();loc=((p-lo)/(hi-lo)-.5).replace([np.inf,-np.inf],np.nan)
ret=np.log(p/p.shift(3));rv=r.rolling(20,min_periods=15).std()*np.sqrt(3)
f=-ret.abs()*np.sign(ret)*(loc.abs()+.25)/(rv+1e-12);f=f.sub(f.mean(axis=1),axis=0)
out=[];sig=[]
for dt in f.index:
 x=f.loc[dt];y=r.shift(-1).loc[dt];ok=x.notna()&y.notna()
 if ok.sum()>=8:out.append((dt,x[ok].corr(y[ok],method='spearman'),ok.sum()))
 sig += [(dt,s,float(f.loc[dt].get(s,np.nan))) for s in U]
a=pd.DataFrame(out,columns=['date','ic','n']).dropna();print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/15,'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean());
for l,h in [(2020,2022),(2023,2025),(2026,2027),(2028,2029)]:
 q=a[(a.date.dt.year>=l)&(a.date.dt.year<=h)];print('regime',l,h,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [1,3,5,10]:
 z=[]
 for dt in f.index:
  x=f.loc[dt];y=np.log(p.shift(-h)/p).loc[dt];ok=x.notna()&y.notna()
  if ok.sum()>=8:z.append(x[ok].corr(y[ok],method='spearman'))
 z=pd.Series(z).dropna();print('decay',h,z.mean(),z.mean()/z.std(ddof=1))
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20290906_location3_signal.csv',index=False)
