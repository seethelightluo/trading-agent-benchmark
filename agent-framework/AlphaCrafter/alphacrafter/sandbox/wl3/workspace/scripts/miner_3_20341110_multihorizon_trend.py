import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f); d.date=pd.to_datetime(d.date); P[s]=d.set_index('date').close
P=pd.DataFrame(P).sort_index(); ret=P.pct_change()
# Multi-horizon trend alignment: combine 20d and 60d returns after volatility normalization.
# Every input is lagged one session before measuring forward returns.
vol=ret.rolling(20).std()*np.sqrt(252)
m20=P.pct_change(20).div(vol)
m60=P.pct_change(60).div(vol)
F=(0.6*m20+0.4*m60).shift(1)
print('assets',len(P.columns),'dates',len(P),'signal_coverage',F.notna().mean().mean())
for h in [1,3,5,10,20]:
 vals=[]; ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(vals); print('horizon',h,'dates',len(a),'avgN',np.mean(ns),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(len(a)),'hit',np.mean(a>0))
 if h==10:
  for n in [120,252,756,1260]:
   q=a[-n:]; print('recent',n,'IC',np.nanmean(q),'ICIR',np.nanmean(q)/np.nanstd(q,ddof=1)*np.sqrt(len(q)))
rank=F.rank(axis=1,pct=True); print('turnover',np.nanmean(np.abs(rank-rank.shift(1)).mean(axis=1)))
