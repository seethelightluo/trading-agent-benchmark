import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(p):
 d=pd.read_csv(p);d.date=pd.to_datetime(d.date);return d.set_index('date').close.astype(float)
px=pd.concat({s:load(Path('../persistent/stock_data')/(s+'.csv')) for s in U},axis=1).sort_index().loc[:'2026-11-04']
v=load(Path('../persistent/index_data/VIX.csv')).reindex(px.index).ffill()
# VIX falling (risk-on) boosts momentum; VIX rising suppresses it
reg=-(v.pct_change(5)).clip(-.5,.5).fillna(0); f=px.pct_change(20).mul(1+reg,axis=0)
for h in [1,5,10]:
 y=px.pct_change(h).shift(-h);a=[];ds=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(d);ns.append(len(z))
 s=pd.Series(a,index=pd.DatetimeIndex(ds));print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4))
 if h==1:print('annual',[(yr,round(s[s.index.year==yr].mean(),5),len(s[s.index.year==yr])) for yr in range(2020,2027)])
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'period',px.index.min(),px.index.max())
