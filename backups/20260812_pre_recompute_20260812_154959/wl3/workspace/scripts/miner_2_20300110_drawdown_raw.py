import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; xs={}
for s in U:
 try:
  d=get_stock_daily_data(s,days=3000)
  if d is not None and len(d)>100: xs[s]=d.set_index('date')['close'].astype(float)
 except FileNotFoundError: pass
p=pd.DataFrame(xs).sort_index(); dd=p/p.rolling(60).max()-1; f=(-dd).shift(1)
for h in [1,3,5,10]:
 fr=np.log(p).shift(-h)-np.log(p); z=[]; ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); ns.append(len(a))
 z=pd.Series(z).dropna(); print(h,len(z),round(np.mean(ns),2),round(z.mean(),6),round(z.mean()/z.std(),6),round((z>0).mean(),4))
print('coverage',f.notna().sum().sum()/f.size,'dates',len(p),'names',len(xs))
f.stack().rename('signal').to_csv('scripts/miner_2_20300110_drawdown_raw_signal.csv')
