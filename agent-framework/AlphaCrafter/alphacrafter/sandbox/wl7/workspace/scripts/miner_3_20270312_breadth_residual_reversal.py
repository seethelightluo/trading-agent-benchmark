import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 try:
  d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d['date']=pd.to_datetime(d.date); px[a]=d.set_index('date').close
 except Exception as e: print('missing',a,e)
p=pd.DataFrame(px).sort_index().loc[:'2027-03-12']; r=p.pct_change()
med=r.rolling(20).median().shift(1); breadth=(med>0).mean(axis=1)
res10=p.pct_change(10).sub(p.pct_change(10).median(axis=1),axis=0); vol=r.rolling(20).std().shift(1)
activation=(1+np.clip((0.5-breadth)*2,-0.5,0.8))
signal=(-res10.shift(1)/vol)*activation.shift(1)
for h in [1,5,10,20]:
 fr=p.pct_change(h).shift(-h); vals=[]; dates=[]; ns=[]
 for dt in signal.index:
  z=pd.concat([signal.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 q=pd.Series(vals,index=dates); print('h',h,'dates',len(q),'avgN',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std()*np.sqrt(252),'hit',(q>0).mean())
rank=signal.rank(axis=1,pct=True); print('coverage',signal.notna().mean().mean(),'turnover',rank.diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2027')]:
 q=[]
 for dt in signal.loc[lo:hi].index:
  z=pd.concat([signal.loc[dt],p.pct_change().shift(-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(lo,len(q),np.mean(q) if q else np.nan)
signal.to_csv('scripts/miner_3_20270312_breadth_residual_reversal_signal.csv')
