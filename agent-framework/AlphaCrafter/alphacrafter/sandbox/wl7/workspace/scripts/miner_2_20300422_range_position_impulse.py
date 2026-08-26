import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={}; hi={}; lo={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>120:
  d=d.copy(); d.date=pd.to_datetime(d.date); x=d.set_index('date'); cl[s]=x.close.astype(float); hi[s]=x.high.astype(float); lo[s]=x.low.astype(float)
p=pd.concat(cl,axis=1).sort_index().ffill(); H=pd.concat(hi,axis=1).reindex(p.index).ffill(); L=pd.concat(lo,axis=1).reindex(p.index).ffill()
r=np.log(p).diff(); atr=((H-L)/p).rolling(20).mean(); vol=r.rolling(20).std()
# Directional range-expansion continuation: recent return, scaled by volatility, weighted by close location in its 20d range.
ret=np.log(p/p.shift(10)); rng=(p-p.rolling(60).min())/(p.rolling(60).max()-p.rolling(60).min())
f=(ret/vol)*(2*rng-1)
f=f.shift(1)
for h in [1,5,10,20,40]:
 fr=np.log(p.shift(-h)/p); z=[]
 for dt in p.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(a.iloc[:,0].corr(a.iloc[:,1]))
 q=pd.Series(z).dropna(); print('h',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean())
fr=np.log(p.shift(-10)/p); z=[]
for dt in p.index:
 a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8:z.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
q=pd.DataFrame(z,columns=['date','ic','n']).set_index('date')
for name,a in [('early',q.iloc[:len(q)//3]),('mid',q.iloc[len(q)//3:2*len(q)//3]),('late',q.iloc[2*len(q)//3:])]:print(name,len(a),a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1)*np.sqrt(len(a)),a.n.mean())
print('assets',len(cl),'calendar_dates',len(p),'coverage',f.notna().sum(axis=1).div(len(cl)).mean(),'rank_turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
sig=f.stack().rename('signal').reset_index();sig.columns=['date','symbol','signal'];sig.to_csv('scripts/miner_2_20300422_range_position_impulse_signal.csv',index=False)
q.reset_index().to_csv('scripts/miner_2_20300422_range_position_impulse_ic.csv',index=False)
