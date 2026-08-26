import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={}; vol={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).set_index('date')
  cl[a]=d['close'].replace(0,np.nan); vol[a]=d['volume'].replace(0,np.nan)
P=pd.DataFrame(cl).sort_index(); V=pd.DataFrame(vol).reindex(P.index)
r=P.pct_change(); lag=P.shift(1)
r5=lag.pct_change(5); v20=r.shift(1).rolling(20).std()
rv=V.shift(1)/V.shift(1).rolling(20).median()-1
raw=-r5/v20*(1+rv.clip(-0.5,1.0))
sig=raw.rank(axis=1,pct=True)-0.5
rows=[]
for dt in sig.index:
 fwd=P.shift(-10).loc[dt]/P.loc[dt]-1
 z=pd.concat([sig.loc[dt],fwd],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avgN',round(q.n.mean(),3),'coverage',round(q.n.sum()/(len(q)*15),4))
for direction in [1,-1]:
 x=q.ic*direction
 print('direction',direction,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 for w in [365,750,1260]:
  z=x.tail(w); print('recent',w,round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
for h in [1,5,10,20]:
 yy=P.shift(-h)/P-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,round(np.nanmean(rr),6),'n',len(rr))
print('turnover',round(sig.diff().abs().mean().mean(),6))
sig.tail(700).to_csv('scripts/miner_1_20341109_volume_confirmed_reversal_signal.csv')
q.to_csv('scripts/miner_1_20341109_volume_confirmed_reversal_ic.csv')
