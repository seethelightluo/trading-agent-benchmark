import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv');d.date=pd.to_datetime(d.date);p[a]=d.set_index('date').close
p=pd.DataFrame(p).sort_index(); r=p.pct_change()
# Trend consistency: signed 20d momentum scaled by fraction of positive days, lagged.
win=(r>0).rolling(20,min_periods=15).mean(); mom=p.pct_change(20)
f=(mom*(2*win-1)).shift(1)
for h in [1,5,10,20]:
 vals=[]; ns=[]; fw=p.shift(-h)/p-1
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=pd.Series(vals);print(h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for n in [60,120,250]:
 vals=[]
 for dt in f.index[-n:]:
  z=pd.concat([f.loc[dt],(p.shift(-1)/p-1).loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 s=pd.Series(vals);print('recent',n,round(s.mean(),6),round(s.mean()/s.std(ddof=1),6),len(s))
print('coverage',round(f.notna().mean().mean(),4),'avg valid',round(f.notna().sum(axis=1).mean(),2))
ics=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],(p.shift(-1)/p-1).loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:ics.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
q=pd.DataFrame(ics,columns=['date','ic']).set_index('date');print(q.groupby(q.index.year).agg(['mean','count']).round(5).to_string())
# rank turnover proxy
print('rank turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),5))
