import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=Path('../persistent/stock_data')/(s+'.csv'); x=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
 D[s]=x.loc[:'2028-10-18','close'].astype(float)
px=pd.concat(D,axis=1).dropna(how='all'); rets=px.pct_change()
# One interpretable idea: distance from prior 60-session high, with a 20d trend confirmation.
# Negative distance is a pullback; score favors assets near/above breakout while requiring positive trend.
high=px.rolling(60,min_periods=50).max().shift(1)
trend=px.pct_change(20).shift(1)
bread=(rets.rolling(20,min_periods=15).mean().shift(1)>0).astype(float)
f=(px.shift(1)/high-1) * (0.5+0.5*bread) # breakout distance, lagged
# Actually px.shift(1)/high where high already shifted: no future data
for h in [1,5,10,20]:
 vals=[]; dates=[]; ns=[]
 fr=px.pct_change(h).shift(-h)
 for dt in f.index:
  a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
 q=pd.Series(vals,index=dates).dropna(); recent=q.tail(250)
 print(h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5),'hit',round((q>0).mean(),3),'recent250',round(recent.mean(),5),round(recent.mean()/recent.std(ddof=1),5),'coverage',round(np.mean(ns)/15,4))
# signal turnover as rank ordering changes
r=f.rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
