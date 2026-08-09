import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
from pathlib import Path
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.DataFrame({Path(f).stem:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in glob.glob('../persistent/stock_data/*.csv')}).sort_index().ffill(); px=px[[a for a in A if a in px]]
macro=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(px.index).ffill()
r=px.pct_change(); mv=macro.pct_change()
# Residual 30d return after rolling asset sensitivity to VIX moves; winsorized cross section.
window=60; horizon=20
cov=r.rolling(window,min_periods=40).cov(mv); var=mv.rolling(window,min_periods=40).var(); beta=cov.div(var,axis=0)
ret=px.pct_change(30); vix30=macro.pct_change(30)
f=ret-beta.mul(vix30,axis=0)
f=f.clip(lower=f.quantile(.05,axis=1),upper=f.quantile(.95,axis=1),axis=0)
print('idea=vix_residual_relative_momentum','instruments',len(f.columns),'rows',len(f),'coverage',round(f.notna().mean().mean(),4))
for h in [1,5,10,20,40]:
 fw=px.shift(-h)/px-1; z=[]; ds=[]; ns=[]
 for d in f.index:
  ok=f.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:
   q=spearmanr(f.loc[d][ok],fw.loc[d][ok]).statistic
   if np.isfinite(q): z.append(q);ds.append(d);ns.append(ok.sum())
 z=pd.Series(z,index=ds)
 print('H',h,'dates',len(z),'meanN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4),'latest120',round(z.tail(120).mean(),6),round(z.tail(120).mean()/z.tail(120).std(ddof=1),6))
print('turnover10',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2031')]:
 q=[]; fw=px.shift(-20)/px-1
 for d in f.loc[lo:hi].index:
  ok=f.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:q.append(spearmanr(f.loc[d][ok],fw.loc[d][ok]).statistic)
 q=pd.Series(q).dropna();print('regime',lo,hi,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
