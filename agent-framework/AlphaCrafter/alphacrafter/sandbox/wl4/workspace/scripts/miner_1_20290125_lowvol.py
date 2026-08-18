import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 d[s]=x['close'].astype(float)
p=pd.DataFrame(d).sort_index(); r=p.pct_change()
# low-vol factor: inverse trailing realized volatility, lagged one day
f=1/r.rolling(20,min_periods=15).std().shift(1)
for h in [1,5,10,20]:
 vals=[]; dates=[]; ns=[]
 for i in range(len(p)-h):
  y=p.iloc[i+h]/p.iloc[i]-1
  z=f.iloc[i]
  q=pd.concat([z,y],axis=1).dropna()
  if len(q)>=8:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); dates.append(p.index[i]); ns.append(len(q))
 a=np.array(vals); print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),5),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),5),'hit',round(np.mean(a>0),4))
# turnover rank changes
ranks=f.rank(axis=1,pct=True); ch=(ranks.diff().abs().mean(axis=1)).dropna(); print('coverage',round(f.notna().mean().mean(),4),'turnover',round(ch.mean(),5))
