import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in S:
 f='../persistent/stock_data/'+s+'.csv'; d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); P[s]=d.close.astype(float)
p=pd.DataFrame(P).loc[:'2029-10-18']; r=p.pct_change(); v=r.rolling(20,min_periods=15).std()
# fast/medium trend blend, risk adjusted and lagged
sig=(0.6*p.pct_change(5).shift(1)/v.shift(1)+0.4*p.pct_change(20).shift(1)/v.shift(1))
def run(h):
 f=p.shift(-h)/p-1; a=[]; ns=[]; prev=None; to=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z)); q=sig.loc[dt].rank(pct=True)
   if prev is not None: to.append(np.mean(abs(q-prev)))
   prev=q
 a=np.array(a); return len(a),np.mean(ns),np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0),np.mean(to)
print('dates period',len(sig.loc[sig.first_valid_index():]),sig.first_valid_index(),sig.last_valid_index())
for h in [1,5,10,20]: print(h,run(h))
for lo,hi in [('2026-07-01','2027-12-31'),('2028-01-01','2029-10-18')]:
 x=sig.loc[lo:hi]; f=p.shift(-10)/p-1;a=[]
 for dt in x.index:
  z=pd.concat([x.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print(lo,len(a),a.mean(),a.mean()/a.std(ddof=1))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20291018_blended_trend_signal.csv',index=False)
