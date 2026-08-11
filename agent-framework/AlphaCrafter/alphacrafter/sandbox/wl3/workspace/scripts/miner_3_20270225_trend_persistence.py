import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; xs=[]
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<150:d=get_index_daily_data(s,2600)
 if d is not None:xs.append(d[['date','close']].assign(symbol=s))
w=pd.concat(xs).pivot(index='date',columns='symbol',values='close').sort_index();r=w.pct_change()
ret=w.pct_change(60);dv=r.where(r<0).rolling(40,min_periods=20).std();dd=w/w.rolling(120,min_periods=60).max()-1
f=ret/(dv*np.sqrt(60)+1e-12)+.35*w.pct_change(20)/(r.rolling(20,min_periods=12).std()*np.sqrt(20)+1e-12)+.20*dd
# row-wise winsorization via explicit masking (safe alignment)
lo=f.quantile(.05,axis=1); hi=f.quantile(.95,axis=1)
f=f.where(f.ge(lo,axis=0),lo,axis=0).where(lambda x:x.le(hi,axis=0),hi,axis=0)
print('cutoff',w.index.max().date(),'dates',len(w),'assets',len(w.columns))
for h in [1,3,5,10]:
 ic=[];ns=[];fr=w.shift(-h)/w-1
 for dt in w.index:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:ic.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 q=pd.Series(ic).dropna();print('H',h,'n',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),6),'hit',round((q>0).mean(),4))
print('coverage',round(f.notna().mean().mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
f.stack().rename('signal').reset_index().to_csv('scripts/miner_3_20270225_trend_persistence_signal.csv',index=False)
for name,lo_,hi_ in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025+','2025','2099')]:
 z=[]
 for dt in f.loc[lo_:hi_].index:
  q=pd.concat([f.loc[dt],(w.shift(-1)/w-1).loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 z=pd.Series(z).dropna();print('REG',name,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1)*np.sqrt(len(z)),6) if len(z)>1 else None)
