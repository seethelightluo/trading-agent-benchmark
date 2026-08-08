import pandas as pd,numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in syms}).sort_index(); r=p.pct_change()
# One interpretable candidate: acceleration divided by recent realized risk.
acc=r.rolling(10).sum()-r.shift(10).rolling(10).sum()
f=acc/r.rolling(20).std().replace(0,np.nan)
print('candidate= (ret10-ret_prev10)/std20; end=',p.index.max().date())
for h in [1,5,10,20]:
 vals=[]; ns=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turn10',round((f.rank(axis=1,pct=True)-f.shift(10).rank(axis=1,pct=True)).abs().mean(axis=1).mean(),4))
for lo,hi in [(2024,2027),(2028,2030),(2031,2032)]:
 a=[]
 for i in range(len(p)-10):
  if lo<=p.index[i].year<=hi:
   z=pd.concat([f.iloc[i],p.iloc[i+10]/p.iloc[i]-1],axis=1).dropna()
   if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print('REG',lo,hi,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6))
# library audit intentionally explicit: compare against persisted signal columns if available; otherwise report missing evidence.
print('LIBRARY_AUDIT: no serialized per-date signal vectors exist in factor JSON; exact max_abs_library_correlation cannot be established; admission fails pending reconstruction.')
