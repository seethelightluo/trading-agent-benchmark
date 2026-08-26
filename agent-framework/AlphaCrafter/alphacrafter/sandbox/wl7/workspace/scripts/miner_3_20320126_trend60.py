import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2032-01-25'); D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);D[s]=x.sort_values('date').set_index('date')
R={1:[],5:[],10:[],20:[]}; cov=[];turn=[];prev=None;rows=[]
for t in sorted(set().union(*[set(x.index[x.index<=cutoff]) for x in D.values()])):
 a={};F={h:{} for h in R}
 for s,x in D.items():
  if t not in x.index:continue
  z=x.loc[:t];c=z.close.astype(float)
  if len(z)<125:continue
  r60=c.iloc[-1]/c.iloc[-61]-1; vol60=np.log(c/c.shift(1)).rolling(60).std().iloc[-1]
  r20=c.iloc[-1]/c.iloc[-21]-1
  if not np.all(np.isfinite([r60,vol60,r20])) or vol60<=0:continue
  # long trend with recent confirmation, volatility normalized
  a[s]=(r60/vol60)*(1+0.25*np.tanh(r20/(vol60*np.sqrt(20))))
  rows.append({'date':str(t.date()),'symbol':s,'signal':float(a[s])})
  fut=x.loc[x.index>t].close
  for h in R:
   if len(fut)>=h:F[h][s]=fut.iloc[h-1]/c.iloc[-1]-1
 if len(a)>=8:
  cov.append(len(a)/15); rr=pd.Series(a).rank(pct=True);turn.append(0 if prev is None else (rr-prev.reindex(rr.index).fillna(.5)).abs().mean());prev=rr
  for h in R:
   ss=[s for s in a if s in F[h]]
   if len(ss)>=8:
    q=spearmanr([a[s] for s in ss],[F[h][s] for s in ss]).statistic
    if np.isfinite(q):R[h].append(q)
print('cutoff',cutoff.date(),'coverage',np.mean(cov),'turnover',np.mean(turn),'dates',len(R[10]),'avgN',len(U))
for h,q0 in R.items():
 q=pd.Series(q0);n=len(q);print('H',h,'dates',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'thirds',[q.iloc[i*n//3:(i+1)*n//3].mean() for i in range(3)])
pd.DataFrame(rows).to_csv('scripts/miner_3_20320126_trend60_signal.csv',index=False)
