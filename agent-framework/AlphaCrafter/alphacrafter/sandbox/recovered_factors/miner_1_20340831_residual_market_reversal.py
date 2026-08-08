import pandas as pd, numpy as np
from scipy.stats import spearmanr
import glob, os
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
p=pd.DataFrame(px).sort_index().ffill(); r=np.log(p).diff()
# residual short-horizon reversal: 5d asset return less rolling beta to equal-weight market times market return
m=r.mean(axis=1); out=[]
for w in [5,10,20]:
 beta=r.rolling(60,min_periods=30).cov(m).div(m.rolling(60,min_periods=30).var(),axis=0)
 resid=r.rolling(w).sum().sub(beta.mul(m.rolling(w).sum(),axis=0))
 f=-resid.shift(1)
 for h in [1,5,10,20]:
  fr=r.rolling(h).sum().shift(-h)
  ics=[]; ns=[]; turnovers=[]
  dates=f.index
  for d in dates:
   x=f.loc[d]; y=fr.loc[d]; z=pd.concat([x,y],axis=1).dropna()
   if len(z)>=8:
    ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  ic=np.nanmean(ics); sd=np.nanstd(ics,ddof=1); print('w',w,'h',h,'dates',len(ics),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.3f'%(ic,ic/sd if sd else 0,np.mean(np.array(ics)>0)))
 print('coverage',f.notna().mean().mean(),'turn10',f.rank(axis=1).diff(10).abs().mean().mean()/len(assets))
