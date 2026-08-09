import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 D[a]=d
# common date panel, factor: lagged negative intraday return, smoothed 5d to reduce noise
close=pd.DataFrame({a:D[a].close for a in assets}); op=pd.DataFrame({a:D[a].open for a in assets})
r=close/op-1
f=-(r.rolling(5,min_periods=3).mean()).shift(1)
for h in [1,5,10,20]:
  ic=[]; ns=[]; turnovers=[]
  for t in close.index:
   if t not in f.index: continue
   j=close.index.get_loc(t)+h
   if j>=len(close): continue
   x=f.loc[t]; y=close.iloc[j]/close.loc[t]-1
   z=pd.concat([x,y],axis=1).dropna()
   if len(z)<8: continue
   ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  ic=np.array(ic); print('H',h,'dates',len(ic),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(np.mean(ic),np.mean(ic)/np.std(ic,ddof=1),np.mean(ic>0)))
# coverage and rank turnover on decision dates
valid=f.notna().sum(axis=1); print('coverage',valid.mean()/15,'cells',valid.sum(),'turnover proxy',np.mean((f.rank(axis=1,pct=True).diff().abs().sum(axis=1)/2).dropna()))
# regime blocks
for lo,hi in [('2020','2023'),('2024','2026'),('2027','2028'),('2029','2030')]:
 q=[]
 for t in close.index:
  if not (t.strftime('%Y')>=lo and t.strftime('%Y')<=hi): continue
  j=close.index.get_loc(t)+5
  if j>=len(close):continue
  z=pd.concat([f.loc[t],close.iloc[j]/close.loc[t]-1],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(lo,hi,len(q),np.mean(q) if q else np.nan, np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
