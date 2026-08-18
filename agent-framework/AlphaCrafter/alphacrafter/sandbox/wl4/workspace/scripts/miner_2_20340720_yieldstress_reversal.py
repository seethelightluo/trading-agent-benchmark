import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=3600)
 if x is not None:
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index(); R=P.pct_change()
# cross-asset stress: breadth of negative 5d returns; in broad stress, favor recent losers (rebound)
r5=P/P.shift(5)-1
bread=(r5<0).mean(axis=1)
f=(-r5).rank(axis=1,pct=True).mul((bread-0.5).abs()+0.5,axis=0).shift(1)
fr=P.shift(-10)/P-1
ics=[]; ns=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1]));ns.append(len(z))
ic=np.array(ics);print('dates',len(ic),'avg_instruments',np.mean(ns));print('mean_ic %.9f icir %.9f hit %.4f'%(np.mean(ic),np.mean(ic)/np.std(ic,ddof=1),np.mean(ic>0)))
for n in [120,260,520,780,1200]:
 q=ic[-n:];print('window',n,'ic %.9f icir %.9f'%(np.mean(q),np.mean(q)/np.std(q,ddof=1)))
print('coverage',P.notna().mean().mean())
