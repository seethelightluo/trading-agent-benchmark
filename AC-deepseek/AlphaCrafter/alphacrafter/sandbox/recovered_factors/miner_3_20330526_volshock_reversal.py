import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for a in assets:
 q=pd.read_csv('../persistent/stock_data/'+a+'.csv'); q.date=pd.to_datetime(q.date); d[a]=q.set_index('date').close
p=pd.DataFrame(d).sort_index(); r=p.pct_change()
# Volatility-shock reversal: fade recent 5d move only when current volatility is elevated
v20=r.rolling(20,min_periods=15).std(); baseline=v20.rolling(120,min_periods=80).median()
shock=(v20/baseline-1).clip(lower=0)
sig=(-p.pct_change(5)/v20*shock).shift(1)
print('range',p.index.min().date(),p.index.max().date(),'assets',len(p.columns))
print('candidate volatility_shock_reversal; valid cells',int(sig.notna().sum().sum()))
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; vals=[]; ns=[]; dates=[]
 for dt in p.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 a=np.asarray(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
# coverage and 10day rank turnover
print('coverage',round(sig.notna().mean().mean(),4),'turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4),'mean_valid',round(sig.notna().sum(axis=1).mean(),2))
for y in range(2020,2034):
 vals=[]
 for dt in p.index[p.index.year==y]:
  z=pd.concat([sig.loc[dt],(p.shift(-1)/p-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 if len(vals)>20: print('YEAR',y,len(vals),round(np.mean(vals),6),round(np.mean(vals)/np.std(vals,ddof=1),4))
