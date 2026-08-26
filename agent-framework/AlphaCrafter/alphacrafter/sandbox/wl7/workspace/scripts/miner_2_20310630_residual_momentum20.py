import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames=[]
for s in U:
 d=get_stock_daily_data(s, days=5000)
 if d is not None and len(d):
  x=d[['date','close']].copy().set_index('date'); x.columns=[s]; frames.append(x)
p=pd.concat(frames,axis=1).sort_index().ffill(); r=p.pct_change()
mom=p.pct_change(20); vol=r.rolling(40).std()*np.sqrt(252)
cs=mom.sub(mom.median(axis=1),axis=0); f=(cs/vol).shift(1)
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; dates=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
 q=pd.Series(vals,index=dates).dropna()
 print('H',h,'dates',len(q),'avgN',round(float(np.mean(ns)),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round((q>0).mean(),4))
 if h==1:
  print('coverage',round(f.notna().mean().mean(),4),'turnover',round(float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),6),'segments',[round(x.mean(),6) for x in np.array_split(q,3)])
out=f.copy(); out.index=out.index.astype(str); out.to_csv('scripts/miner_2_20310630_residual_momentum20_signal.csv')
