import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f);d.date=pd.to_datetime(d.date);D[s]=d.set_index('date').close
P=pd.DataFrame(D).sort_index(); r=P.pct_change(); v=r.rolling(20).std()
# Trend-exhaustion: medium trend plus short pullback, risk normalized, lagged one session.
F=(0.7*P.pct_change(20).div(v)+0.3*(-P.pct_change(5).div(v))).shift(1)
F.to_csv('scripts/miner_3_20341110_trend_exhaustion_signal.csv',index_label='date')
print('assets',len(P.columns),'dates',len(P),'coverage',F.notna().mean().mean())
for h in [1,3,5,10,20]:
 a=[]; ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.asarray(a);print('horizon',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(len(a)),'hit',np.mean(a>0))
 if h==10:
  for n in [120,252,756,1260]:
   q=a[-n:];print('recent',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)))
print('turnover',np.nanmean(np.abs(F.rank(axis=1,pct=True)-F.rank(axis=1,pct=True).shift(1)).mean(axis=1)))
