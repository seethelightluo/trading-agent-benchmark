"""Miner 3: volatility-normalized three-day reversal."""
import pandas as pd,numpy as np,glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2035-11-22')
def rd(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return pd.to_numeric(d.loc[d.index<=END,'close'],errors='coerce')
C=pd.DataFrame({a:rd(a) for a in A}); R=C.pct_change(fill_method=None); vol=R.rolling(20,min_periods=15).std()
F=(-R.rolling(3,min_periods=3).sum()/vol).shift(1).clip(-10,10)
def calc(h):
 vals=[]; turns=[]; breadth=[]; prev=None; cells=0
 for i in range(len(C)-h):
  q=pd.concat([F.iloc[i],R.shift(-h).iloc[i]],axis=1).dropna()
  if len(q)>=8:
   vals.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank(),method='spearman')); breadth.append(len(q));cells+=len(q)
   s=F.iloc[i].rank(); turns.append((s!=prev).mean() if prev is not None else np.nan);prev=s
 v=np.array(vals); v=v[np.isfinite(v)]
 return np.mean(v),np.mean(v)/(np.std(v,ddof=1)+1e-12),len(v),np.mean(breadth),np.nanmean(turns),np.mean(v>0),cells
for h in [1,5,10,20]: print('horizon',h,calc(h))
for lo,hi in [('2020','2025-12-31'),('2026','2030-12-31'),('2031','2035-11-22')]:
 vals=[]
 for i in range(len(C)-1):
  if not(C.index[i]>=pd.Timestamp(lo) and C.index[i]<=pd.Timestamp(hi)):continue
  q=pd.concat([F.iloc[i],R.shift(-1).iloc[i]],axis=1).dropna()
  if len(q)>=8: vals.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank(),method='spearman'))
 v=np.array(vals); print('regime',lo,len(v),np.nanmean(v),np.nanmean(v)/(np.nanstd(v,ddof=1)+1e-12))
print('coverage',F.notna().sum().sum()/(len(F)*15),'dates',len(C),'assets',15,'library',len(glob.glob('factors/*.json')))
