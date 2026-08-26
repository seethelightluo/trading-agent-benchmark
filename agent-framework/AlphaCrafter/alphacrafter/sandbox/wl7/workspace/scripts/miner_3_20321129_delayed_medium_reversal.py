import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x)>100:
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date); D[s]=z.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); vol30=r.rolling(30).std()
# Medium-horizon delayed reversal: fade 10-session return ending three sessions before signal,
# volatility normalized. Shift keeps the latest three completed observations out of the feature.
f=(-(p.pct_change(10).shift(3))/vol30.shift(3))
rows=[]
for h in [1,5,10,20]:
 fr=p.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1])); ns.append(len(a))
 q=pd.Series(vals).dropna(); rows.append((h,len(q),np.mean(ns),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1),(q>0).mean()))
print('cutoff',p.index.max().date(),'calendar_dates',len(p),'avgN=%.2f'%f.notna().sum(axis=1).mean())
for h,n,avgn,ic,sd,ir,hit in rows: print('H%d dates=%d avgN=%.2f IC=%+.6f ICIR=%+.6f hit=%.4f'%(h,n,avgn,ic,ir,hit))
rank=f.rank(axis=1,pct=True); ch=(rank.diff().abs().sum(axis=1)/rank.notna().sum(axis=1)).dropna()
print('coverage=%.4f turnover=%.4f'%(f.notna().mean().mean(),ch.mean()))
out=f.copy(); out.index.name='date'; out.to_csv('scripts/miner_3_20321129_delayed_medium_reversal_signal.csv')
# thirds for admission horizon
h=20; fr=p.pct_change(h).shift(-h); vals=[]; dates=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1])); dates.append(dt)
q=pd.Series(vals,index=pd.to_datetime(dates)); print('H20 thirds',*[round(x.mean(),6) for x in np.array_split(q,3)])
