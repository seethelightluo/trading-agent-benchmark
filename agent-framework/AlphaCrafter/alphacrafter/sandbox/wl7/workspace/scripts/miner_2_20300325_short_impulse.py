import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 try:
  d=get_stock_daily_data(s,days=4000)
  if d is not None and len(d)>100:
   d=d.copy(); d['date']=pd.to_datetime(d.date); frames[s]=d.set_index('date').close.astype(float)
 except Exception as e: print('skip',s,str(e))
p=pd.concat(frames,axis=1).sort_index().ffill(); r=np.log(p).diff(); vol=r.rolling(20).std(); f=(r.rolling(5).sum()/vol).shift(1)
out=[]
for h in [1,5,10,20,40]:
 fr=np.log(p.shift(-h)/p); vals=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(vals).dropna(); out.append((h,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean()))
print('assets',len(frames),'dates',len(p),'last',p.index[-1].date()); print('RESULT',out)
h=10; fr=np.log(p.shift(-h)/p); vals=[]
for dt in p.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
for label,sub in [('early',q.iloc[:len(q)//3]),('mid',q.iloc[len(q)//3:2*len(q)//3]),('late',q.iloc[2*len(q)//3:])]: print(label,len(sub),sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1)*np.sqrt(len(sub)),sub.n.mean())
print('coverage',f.notna().sum(axis=1).div(len(frames)).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
sig=f.stack().rename('signal').reset_index(); sig.columns=['date','symbol','signal']; sig.to_csv('scripts/miner_2_20300325_short_impulse_signal.csv',index=False); q.reset_index().to_csv('scripts/miner_2_20300325_short_impulse_ic.csv',index=False)
