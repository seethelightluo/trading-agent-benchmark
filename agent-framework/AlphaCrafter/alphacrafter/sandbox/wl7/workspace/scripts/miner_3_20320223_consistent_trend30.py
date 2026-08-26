import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').set_index('date')
cutoff=min(pd.Timestamp('2032-02-20'), max(x.index.max() for x in D.values()))
R={h:[] for h in [1,5,10,20]}; cov=[]; turn=[]; prev=None; rows=[]
Ds=sorted(set().union(*[set(x.index[x.index<=cutoff]) for x in D.values()]))
for t in Ds:
 a={}; futs={h:{} for h in R}
 for s,x in D.items():
  if t not in x.index: continue
  z=x.loc[:t]; c=z.close.astype(float)
  if len(c)<75: continue
  r10=c.iloc[-1]/c.iloc[-11]-1; r30=c.iloc[-1]/c.iloc[-31]-1; r60=c.iloc[-1]/c.iloc[-61]-1
  v=np.log(c/c.shift(1)).rolling(40).std().iloc[-1]
  if not np.isfinite(v) or v<=0: continue
  # continuous confirmation: medium trend weighted by agreement of short and long trends
  agree=np.tanh(2*r10/(v*np.sqrt(10)))+0.5*np.tanh(2*r60/(v*np.sqrt(60)))
  a[s]=(r30/v)*(1+0.30*agree)
  rows.append({'date':t.date(),'symbol':s,'signal':a[s]})
  for h in R:
   f=x.loc[x.index>t].close
   if len(f)>=h: futs[h][s]=f.iloc[h-1]/c.iloc[-1]-1
 if len(a)>=8:
  cov.append(len(a)/15); rr=pd.Series(a).rank(pct=True)
  turn.append(0 if prev is None else (rr-prev.reindex(rr.index).fillna(.5)).abs().mean()); prev=rr
  for h in R:
   com=[s for s in a if s in futs[h]]
   if len(com)>=8:
    q=spearmanr([a[s] for s in com],[futs[h][s] for s in com]).statistic
    if np.isfinite(q): R[h].append((t,q,len(com)))
print('cutoff',cutoff.date())
print('coverage',np.mean(cov),'turnover',np.mean(turn),'dates',len(R[10]),'avgN',np.mean([v[2] for v in R[10]]))
for h,a in R.items():
 q=pd.Series([v[1] for v in a]); n=len(q); thirds=[q.iloc[i*n//3:(i+1)*n//3].mean() for i in range(3)]
 print('H',h,'dates',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'thirds',thirds)
pd.DataFrame(rows).to_csv('scripts/miner_3_20320223_consistent_trend30_signal.csv',index=False)
