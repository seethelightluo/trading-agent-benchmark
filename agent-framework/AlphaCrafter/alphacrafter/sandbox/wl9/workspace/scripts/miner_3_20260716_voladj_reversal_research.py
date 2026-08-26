import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:end] for s in U}
# 5d reversal normalized by trailing 20d realized volatility; uses only t close and predicts k observations ahead.
F={}; R={}
for s,x in D.items():
 r=x.close.pct_change(); R[s]=r
 F[s]=-(x.close.pct_change(5)/(r.rolling(20,min_periods=15).std()*np.sqrt(5)+1e-12))
for k in [1,5,10]:
  vals=[]; yr={}; ns=[]
  dates=sorted(set().union(*[set(x.index) for x in D.values()]))
  for dt in dates:
   xs=[];ys=[]
   for s,x in D.items():
    if dt not in x.index or pd.isna(F[s].get(dt)): continue
    i=x.index.get_loc(dt)
    if i+k<len(x): xs.append(F[s].loc[dt]); ys.append(x.close.iloc[i+k]/x.close.iloc[i]-1)
   if len(xs)>=8 and len(set(xs))>1:
    q=spearmanr(xs,ys).statistic
    if pd.notna(q): vals.append(q);ns.append(len(xs));yr.setdefault(dt.year,[]).append(q)
  a=np.array(vals); print('H',k,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round(np.mean(a>0),4),'years',{z:round(np.mean(v),5) for z,v in yr.items()})
# signal coverage and turnover over active cross sections
allf=pd.DataFrame({s:pd.Series(F[s]) for s in U}); print('coverage',round(allf.notna().sum(axis=1).mean()/15,4),'turnover',round(allf.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2,4))
