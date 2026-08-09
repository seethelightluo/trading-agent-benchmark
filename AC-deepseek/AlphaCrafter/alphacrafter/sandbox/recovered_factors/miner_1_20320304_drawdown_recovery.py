import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
from pathlib import Path
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={Path(f).stem:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}; A=[a for a in A if a in D]
cl=pd.DataFrame({a:D[a].close for a in A}).sort_index().ffill()
# Drawdown recovery quality: distance above rolling 60-session low, scaled by 60-session high-low.
lo=cl.rolling(60,min_periods=40).min(); hi=cl.rolling(60,min_periods=40).max()
f=((cl-lo)/(hi-lo)).replace([np.inf,-np.inf],np.nan)
print('assets',len(A),'dates',len(cl),'signal_cells',int(f.notna().sum().sum()),'coverage',round(f.notna().mean().mean(),4))
for h in [1,5,10,20]:
 fw=cl.shift(-h)/cl-1; z=[];ns=[]
 for d in f.index:
  ok=f.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:
   q=spearmanr(f.loc[d][ok],fw.loc[d][ok]).statistic
   if np.isfinite(q):z.append(q);ns.append(ok.sum())
 s=pd.Series(z);print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'recent120',round(s.tail(120).mean(),6),round(s.tail(120).mean()/s.tail(120).std(ddof=1),6))
print('turnover10',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
for label,ix in [('2020-23',(0,1000)),('2024-27',(1000,2000)),('2028-30',(2000,3000)),('2031+', (3000,len(f)))]:
 fw=cl.shift(-10)/cl-1;s=[]
 for d in f.index[ix[0]:ix[1]]:
  ok=f.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:s.append(spearmanr(f.loc[d][ok],fw.loc[d][ok]).statistic)
 print('REG',label,'n',len(s),'IC',round(np.mean(s),6),'ICIR',round(np.mean(s)/np.std(s,ddof=1),6))
