import pandas as pd,numpy as np
from scipy.stats import spearmanr
symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}; P={}
for s in symbols:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:'2027-02-25']
 P[s]=d; r=d.pct_change(); F[s]=(d.shift(5)/d.shift(1)-1) # skip latest 1 day, 5d lagged momentum
px=pd.DataFrame(P); fac=pd.DataFrame(F)
for h in [1,5,10]:
 fwd=px.shift(-h)/px-1; vals=[]; ns=[]; tr=[]
 for dt in fac.index:
  a=fac.loc[dt];b=fwd.loc[dt];ok=a.notna()&b.notna()
  if ok.sum()>=8:
   vals.append(spearmanr(a[ok],b[ok]).statistic);ns.append(ok.sum())
   old=fac.shift(10).loc[dt];ko=ok&old.notna()
   if ko.sum()>=8:tr.append(np.mean(np.sign(a[ko])!=np.sign(old[ko])))
 x=np.array(vals);print(h,len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0),np.mean(tr))
for y in [2020,2021,2022,2023,2024,2025,2026,2027]:
 fwd=px.shift(-10)/px-1;x=[]
 for dt in fac.index:
  if dt.year!=y:continue
  a=fac.loc[dt];b=fwd.loc[dt];ok=a.notna()&b.notna()
  if ok.sum()>=8:x.append(spearmanr(a[ok],b[ok]).statistic)
 print(y,len(x),np.mean(x) if x else np.nan)
out=fac.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_3_20270225_skip5_mom.csv',index=False)
print('coverage',fac.notna().mean().mean(),'matrix',fac.notna().sum().sum()/fac.size)
