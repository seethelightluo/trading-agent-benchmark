import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
O={}; C={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 O[s]=d.open; C[s]=d.close
op=pd.concat(O,axis=1).reindex(columns=U); cl=pd.concat(C,axis=1).reindex(columns=U)
intr=cl/op-1
# Remove the completed day's equal-weight common intraday move, then reverse idiosyncratic pressure.
common=intr.mean(axis=1)
res=intr.sub(common,axis=0)
for w in [1,3,5]:
 f=-res.rolling(w,min_periods=w).mean()
 for h in [1,3,5]:
  y=cl.pct_change(h).shift(-h)
  vals=[]; dates=[]; ns=[]
  for i in range(len(cl)-h):
   q=pd.concat([f.iloc[i],y.iloc[i].rename('y')],axis=1).dropna()
   if len(q)>=8 and q.iloc[:,0].nunique()>1:
    r=spearmanr(q.iloc[:,0],q.y).statistic
    if np.isfinite(r): vals.append(r); dates.append(cl.index[i]); ns.append(len(q))
  a=np.array(vals); print('w',w,'h',h,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
   aa=a[(np.array(dates)>=pd.Timestamp(lo))&(np.array(dates)<=pd.Timestamp(hi))]
   if len(aa): print(' regime',lo,'n',len(aa),'ic',round(aa.mean(),6),'icir',round(aa.mean()/aa.std(ddof=1),6))
 # turnover/coverage for daily signal
 print(' turnover',round(np.nanmean(np.abs(f.rank(pct=True).diff()).mean(axis=1)),5),'coverage',round(f.notna().mean().mean(),5))
