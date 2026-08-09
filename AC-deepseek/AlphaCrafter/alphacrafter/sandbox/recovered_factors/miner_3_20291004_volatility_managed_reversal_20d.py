import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; root='../persistent/stock_data'
p=pd.DataFrame({a:pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index(); r=p.pct_change()
# Contrarian 20-day return, scaled by recent risk; t-1 prevents look-ahead
f=-(p/p.shift(20)-1)/(r.rolling(20).std()*np.sqrt(20)); f=f.shift(1)
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; z=[]; ns=[]
 for d in f.index:
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[d,ok],y.loc[d,ok]).statistic);ns.append(ok.sum())
 z=np.array(z); print(h,len(z),round(np.mean(ns),2),round(z.mean(),5),round(z.mean()/z.std(ddof=1),5),round((z>0).mean(),4))
for label,lo,hi in [('2025-27','2025','2028'),('2028-29','2028','2030')]:
 y=p.shift(-10)/p-1;z=[]
 for d in f.index:
  if not (pd.Timestamp(lo+'-01-01')<=d<pd.Timestamp(hi+'-01-01')):continue
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[d,ok],y.loc[d,ok]).statistic)
 print(label,len(z),round(np.mean(z),5),round(np.mean(z)/np.std(z,ddof=1),5))
print('coverage',round(f.notna().mean().mean(),4),'rows',len(p),'assets',len(A))
