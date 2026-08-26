import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; xs=[]
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d): xs.append(d[['date','close']].set_index('date').rename(columns={'close':s}))
p=pd.concat(xs,axis=1).sort_index().ffill(); r=p.pct_change(); x=r.rolling(3,min_periods=3).sum(); disp=x.std(axis=1); gate=(disp>disp.rolling(60,min_periods=30).median()).astype(float)
f=(-x/r.rolling(20,min_periods=15).std()).shift(1).mul(gate.shift(1),axis=0); f=f.sub(f.median(axis=1),axis=0)
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[];ns=[];ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): vals.append(c);ns.append(len(z));ds.append(dt)
 q=pd.Series(vals,index=ds); print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round((q>0).mean(),4))
 if h==1: print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6),'segments',[round(y.mean(),6) for y in np.array_split(q,3)])
out=f.copy();out.index=out.index.astype(str);out.to_csv('scripts/miner_2_20310728_dispersion_shock_signal.csv')
