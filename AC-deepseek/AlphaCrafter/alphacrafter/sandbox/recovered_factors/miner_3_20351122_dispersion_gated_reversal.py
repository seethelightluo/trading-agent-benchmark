"""Miner 3: dispersion-gated short-horizon reversal."""
import pandas as pd,numpy as np, json,glob,os
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2035-11-22')
def rd(a,col):
 p='../persistent/stock_data/'+a+'.csv'; d=pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return pd.to_numeric(d.loc[d.index<=END,col],errors='coerce')
C=pd.DataFrame({a:rd(a,'close') for a in A}); R=C.pct_change(fill_method=None)
# Lagged reversal, activated only when cross-sectional dispersion is unusually high.
csdisp=R.std(axis=1,min_count=8); gate=csdisp/(csdisp.rolling(60,min_periods=30).median()+1e-12)
F=(-R.rolling(3,min_periods=3).sum().mul(gate,axis=0)).shift(1).clip(-.25,.25)
def calc(h):
 vals=[]; turns=[]; breadth=[]; prev=None; cells=0
 for i in range(len(C)-h):
  q=pd.concat([F.iloc[i],R.shift(-h).iloc[i]],axis=1).dropna()
  if len(q)>=8:
   x=q.iloc[:,0].rank(); y=q.iloc[:,1].rank(); vals.append(x.corr(y,method='spearman')); breadth.append(len(q));cells+=len(q)
   s=F.iloc[i].rank(); turns.append((s!=prev).mean() if prev is not None else np.nan);prev=s
 v=np.array(vals,float); v=v[np.isfinite(v)]
 return float(np.mean(v)),float(np.mean(v)/(np.std(v,ddof=1)+1e-12)),len(v),float(np.mean(breadth)),float(np.nanmean(turns)),float(np.mean(v>0)),cells
for h in [1,5,10,20]: print('horizon',h,calc(h))
for lo,hi in [('2020','2025-12-31'),('2026','2030-12-31'),('2031','2035-11-22')]:
 vals=[]
 for i in range(len(C)-1):
  if not (C.index[i]>=pd.Timestamp(lo) and C.index[i]<=pd.Timestamp(hi)):continue
  q=pd.concat([F.iloc[i],R.shift(-1).iloc[i]],axis=1).dropna()
  if len(q)>=8: vals.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank(),method='spearman'))
 v=np.array(vals,float); print('regime',lo,len(v),float(np.nanmean(v)),float(np.nanmean(v)/(np.nanstd(v,ddof=1)+1e-12)))
print('signal coverage',float(F.notna().sum().sum()/(len(F)*15)),'dates',len(C),'assets',15)
# correlation against admitted factor signal histories, using same-date common cells
print('library audit: inspect only if gates pass')
files=glob.glob('factors/*.json'); print('library files',len(files))
