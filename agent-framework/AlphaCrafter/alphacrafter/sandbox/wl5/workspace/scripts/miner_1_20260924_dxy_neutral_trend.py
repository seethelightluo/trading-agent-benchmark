import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def read(s,macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+s+'.csv'
 return pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index().close
px=pd.concat({s:read(s) for s in U},axis=1).sort_index().loc[:'2026-09-23']
r=px.pct_change(); d=read('DXY',True).reindex(px.index).ffill().pct_change()
# DXY-neutral trend: trailing 20d asset return minus rolling beta to DXY times DXY return, scaled by idiosyncratic vol
cov=r.rolling(60,min_periods=40).cov(d); var=d.rolling(60,min_periods=40).var(); beta=cov.div(var,axis=0)
idio=r.sub(beta.mul(d,axis=0)); f=(idio.rolling(20,min_periods=15).sum()).div(idio.rolling(20,min_periods=15).std())
for h in [1,5,10]:
 vals=[]; ns=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(q)>=8:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 a=np.array(vals); print('DXY-neutral idio trend',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',f.notna().mean().mean())
