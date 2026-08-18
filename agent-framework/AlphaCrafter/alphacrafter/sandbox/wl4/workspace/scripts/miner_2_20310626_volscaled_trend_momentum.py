import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p): p='../persistent/index_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x['date']=pd.to_datetime(x['date']); x=x.sort_values('date').set_index('date'); D[s]=x['close'].astype(float)
px=pd.DataFrame(D).sort_index(); r=px.pct_change()
# one interpretable signal: lagged 20d risk-adjusted momentum, confirmed by 60d trend
f=(px.pct_change(20).shift(1)/r.rolling(20).std().shift(1))*np.sign(px.pct_change(60).shift(1))
# forward returns
for h in [5,10,20]:
 ic=[]; ns=[]; dates=[]
 fr=px.shift(-h)/px-1
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   ic.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic); ns.append(len(a)); dates.append(dt)
 z=pd.Series(ic,index=dates); print('H',h,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'dates',len(z),'avgN',np.mean(ns),'recent365',z[z.index>=z.index.max()-pd.Timedelta(days=365)].mean(),'recent730',z[z.index>=z.index.max()-pd.Timedelta(days=730)].mean())
# coverage and rank turnover
valid=f.notna().sum(axis=1); print('coverage',valid.mean()/15,'avgN',valid.mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
