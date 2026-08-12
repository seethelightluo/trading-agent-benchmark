import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<150: d=get_index_daily_data(s,2600)
 if d is not None: rows.append(d[['date','close']].assign(symbol=s))
w=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=w.pct_change()
# Shock-conditioned short reversal: reverse a recent 3-day move, with stronger signal on
# cross-sectional high-dispersion days, and normalize by each asset's recent risk.
risk=r.rolling(20,min_periods=12).std(); disp=r.std(axis=1).rolling(20,min_periods=12).mean(); ds=(disp-disp.rolling(120,min_periods=40).mean())/(disp.rolling(120,min_periods=40).std()+1e-12)
f=(-w.pct_change(3)/(risk+1e-12)).mul((1+0.35*np.tanh(ds)).clip(0.65,1.35), axis=0)
f=f.replace([np.inf,-np.inf],np.nan).clip(-8,8)
fr=w.shift(-1)/w-1
for h in [1,3,5,10]:
 qs=[]; ns=[]
 for dt in w.index:
  z=pd.concat([f.loc[dt],(w.shift(-h)/w-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):qs.append(q);ns.append(len(z))
 q=pd.Series(qs); print('H',h,'n',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),6),'hit',round((q>0).mean(),4))
print('cutoff',w.index.max().date(),'dates',len(w),'assets',len(w.columns),'coverage',round(f.notna().mean().mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
f.stack().rename('signal').reset_index().to_csv('scripts/miner_2_20270520_dispersion_conditioned_reversal_signal.csv',index=False)
for name,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025+','2025-01-01','2099-12-31')]:
 q=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(q).dropna(); print('REG',name,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),6))
print('max_abs_library_correlation',None)
