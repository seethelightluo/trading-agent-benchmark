import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index()
  px[s]=d
p=pd.DataFrame(px).sort_index().ffill()
r=p.pct_change()
# one interpretable idea: risk-adjusted multi-horizon momentum, with a short-term reversal penalty
mom=(p.shift(1).pct_change(20)+p.shift(1).pct_change(60))/2
vol=r.shift(1).rolling(20).std()*np.sqrt(20)
f=mom/vol
# damp the most recent 3d move to avoid chasing sharp spikes
f=f - 0.35*r.shift(1).rolling(3).sum()/vol
rows=[]; artifact=[]
for dt in f.index:
 x=f.loc[dt]; y=r.shift(-1).loc[dt]
 z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  rows.append((dt,ic,len(z)))
  for s,v in x.items(): artifact.append((dt,s,v))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
cut=pd.Timestamp('2025-01-01'); q=a[a.index>=cut]
mean=q.ic.mean(); sd=q.ic.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
print('dates',len(q),'all_dates',len(a),'avg_n',q.n.mean(),'IC',mean,'ICIR',icir,'hit',(q.ic>0).mean(),'coverage',q.n.mean()/15)
print('early_late',q.iloc[:len(q)//2].ic.mean(),q.iloc[len(q)//2:].ic.mean())
for h in [1,5,10,20]:
 yy=p.shift(-h).pct_change(h)
 vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 vals=pd.Series(vals).dropna(); print('decay',h,vals.mean(),len(vals))
out=pd.DataFrame(artifact,columns=['date','symbol','signal']);out.to_csv('scripts/miner_1_20270308_risk_adjusted_momentum_signal.csv',index=False)
