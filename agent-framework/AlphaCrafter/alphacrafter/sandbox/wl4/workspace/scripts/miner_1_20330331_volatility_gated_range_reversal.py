import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')
  P[s]=d['close'].astype(float)
P=pd.DataFrame(P).sort_index()
r=P.pct_change()
# Candidate: 120-day range-location reversal, strengthened only in high
# cross-sectional volatility regimes. Every ingredient is lagged before use.
hi=P.shift(1).rolling(120,min_periods=60).max()
lo=P.shift(1).rolling(120,min_periods=60).min()
loc=(P.shift(1)-lo)/(hi-lo).replace(0,np.nan)
base=1-loc
avol=r.shift(1).rolling(20,min_periods=10).std()
csvol=avol.median(axis=1)
# high-volatility gate is continuous and cross-sectional-neutral
mult=(avol.div(csvol,axis=0)).clip(0.5,2.0)
sig=base*mult
# artifact required for deterministic audit
os.makedirs('scripts/artifacts',exist_ok=True)
sig.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv('scripts/artifacts/miner_1_20330331_volatility_gated_range_reversal_signal.csv',index=False)
for H in [5,10,20,30]:
 y=P.shift(-H)/P-1; vals=[]; ns=[]; dates=[]
 for d in sig.index:
  q=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q)); dates.append(d)
 x=np.asarray(vals,float)
 print('H',H,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(x),6),'ICIR_daily',round(np.nanmean(x)/(np.nanstd(x,ddof=1)+1e-12),6),'ICIR_ann',round(np.nanmean(x)/(np.nanstd(x,ddof=1)+1e-12)*np.sqrt(252),6),'hit',round(np.mean(x>0),4))
 for label,lo_dt in [('recent365','2032-04-01'),('recent730','2031-04-01'),('recent1095','2030-04-01')]:
  z=x[np.array(dates)>=pd.Timestamp(lo_dt)]
  print(label,'dates',len(z),'IC',round(np.nanmean(z),6) if len(z) else np.nan,'ICIR_daily',round(np.nanmean(z)/(np.nanstd(z,ddof=1)+1e-12),6) if len(z)>1 else np.nan)
print('coverage',round(sig.notna().mean().mean(),6),'avg_valid',round(sig.notna().sum(axis=1).mean(),3),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
