import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2032-01-25'); D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').set_index('date')
# Candidate: volatility-normalized 30d trend, confirmed by agreement of 10d and 60d trends
R={h:[] for h in [1,5,10,20]}; cov=[];turn=[];prev=None; rows=[]
Ds=sorted(set().union(*[set(x.index[x.index<=cutoff]) for x in D.values()]))
for t in Ds:
 a={}; f={h:{} for h in R}
 for s,x in D.items():
  if t not in x.index: continue
  z=x.loc[:t]; c=z.close.astype(float)
  if len(z)<75: continue
  r10=c.iloc[-1]/c.iloc[-11]-1; r30=c.iloc[-1]/c.iloc[-31]-1; r60=c.iloc[-1]/c.iloc[-61]-1
  vol40=np.log(c/c.shift(1)).rolling(40).std().iloc[-1]
  if not all(np.isfinite([r10,r30,r60,vol40])) or vol40<=0: continue
  # smooth agreement multiplier, bounded, avoids hard sign discontinuity
  agree=np.tanh(2.0*r10/(vol40*np.sqrt(10)))+0.5*np.tanh(2.0*r60/(vol40*np.sqrt(60)))
  a[s]=(r30/vol40)*(1+0.30*agree)
  rows.append({'date':str(t.date()),'symbol':s,'signal':float(a[s])})
  fut=x.loc[x.index>t].close
  for h in R:
   if len(fut)>=h: f[h][s]=fut.iloc[h-1]/c.iloc[-1]-1
 if len(a)>=8:
  cov.append(len(a)/15); rr=pd.Series(a).rank(pct=True); turn.append(0 if prev is None else (rr-prev.reindex(rr.index).fillna(.5)).abs().mean()); prev=rr
  for h in R:
   com=[s for s in a if s in f[h] and np.isfinite(f[h][s])]
   if len(com)>=8:
    q=spearmanr([a[s] for s in com],[f[h][s] for s in com]).statistic
    if np.isfinite(q): R[h].append((t,q,len(com)))
print('cutoff',cutoff.date(),'dates',len(R[10]),'avgN',np.mean([z[2] for z in R[10]]),'coverage',np.mean(cov),'turnover',np.mean(turn))
for h,a in R.items():
 q=pd.Series([z[1] for z in a]); n=len(q); thirds=[q.iloc[i*n//3:(i+1)*n//3].mean() for i in range(3)]
 print('H',h,'dates',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'thirds',thirds)
pd.DataFrame(rows).to_csv('scripts/miner_3_20320126_consistent_trend30_signal.csv',index=False)
