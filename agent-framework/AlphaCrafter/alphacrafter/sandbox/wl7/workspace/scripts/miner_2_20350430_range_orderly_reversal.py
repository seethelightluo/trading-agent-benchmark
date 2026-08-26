import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}; F={}; R={}
for s,d in D.items():
 x=d.set_index('date').sort_index(); r=x.close.pct_change(); R[s]=r
 rv=r.rolling(20).std(); intr=(x.high-x.low)/x.close
 ratio=intr.rolling(5).mean()/(intr.rolling(60).mean().shift(5)+1e-12)
 bonus=(1.25-.5*ratio).clip(.5,1.25)
 F[s]=(-r.rolling(5).sum()/rv*bonus)
# align panel; t signal uses close through t, forward starts t+1
panel=pd.concat(F,axis=1); rp=pd.concat(R,axis=1).reindex(panel.index)
for h in [1,5,10,20]:
  vals=[]
  for s in U:
   fut=(1+rp[s].shift(-1)).rolling(h).apply(np.prod,raw=True).shift(-(h-1))-1
   vals.append(fut)
  fr=pd.concat(vals,axis=1); fr.columns=U
  ics=[]; ns=[]
  for t in panel.index:
   a=panel.loc[t]; b=fr.loc[t]; ok=a.notna()&b.notna()
   if ok.sum()>=8:ics.append(a[ok].corr(b[ok]));ns.append(ok.sum())
  z=pd.Series(ics).dropna(); print('H',h,'ic',z.mean(),'icir',z.mean()/z.std(ddof=1),'dates',len(z),'avg_n',np.mean(ns))
# signal artifact
long=panel.stack().rename('signal').reset_index();long.columns=['date','symbol','signal'];long.to_csv('scripts/miner_2_20350430_range_orderly_reversal_signal.csv',index=False)
# coverage and turnover top rank
print('coverage',panel.notna().sum().sum()/(panel.shape[0]*15),'period',panel.index.min(),panel.index.max())
rank=panel.rank(axis=1,pct=True); print('rank turnover', (rank.diff().abs().mean().mean()))
