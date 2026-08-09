import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
from pathlib import Path
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={Path(f).stem:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}; A=[a for a in A if a in D]
# Close-location persistence: average signed location of close within daily range,
# a price-action pressure measure distinct from return momentum.
lo=pd.DataFrame({a:D[a].low for a in A}).sort_index().ffill(); hi=pd.DataFrame({a:D[a].high for a in A}).sort_index().ffill(); cl=pd.DataFrame({a:D[a].close for a in A}).sort_index().ffill(); op=pd.DataFrame({a:D[a].open for a in A}).sort_index().ffill()
rng=(hi-lo).replace(0,np.nan); loc=(2*cl-hi-lo)/rng
f=loc.rolling(20,min_periods=15).mean()
# use only completed prior row: evaluation naturally aligns signal date to future close
print('assets',len(A),'dates',len(cl),'signal_cells',int(f.notna().sum().sum()),'coverage',round(f.notna().mean().mean(),4))
for h in [1,5,10,20]:
 fw=cl.shift(-h)/cl-1; z=[]; ns=[]
 for d in f.index:
  ok=f.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:
   q=spearmanr(f.loc[d][ok],fw.loc[d][ok]).statistic
   if np.isfinite(q): z.append(q);ns.append(ok.sum())
 s=pd.Series(z); print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'recent120',round(s.tail(120).mean(),6),round(s.tail(120).mean()/s.tail(120).std(ddof=1),6))
print('turnover10',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
for p in [(0, len(f)),(0,1000),(1000,2000),(2000,3000),(3000,len(f))]:
 s=[]; fw=cl.shift(-10)/cl-1
 for d in f.index[p[0]:p[1]]:
  ok=f.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:s.append(spearmanr(f.loc[d][ok],fw.loc[d][ok]).statistic)
 print('REG',p,'n',len(s),'IC',round(np.mean(s),6) if s else None,'ICIR',round(np.mean(s)/np.std(s,ddof=1),6) if len(s)>1 else None)
