import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').set_index('date')
cutoff=pd.Timestamp('2032-03-05')
H=[1,5,10,20]; R={h:[] for h in H}; cov=[]; turn=[]; prev=None; rows=[]
Ds=sorted(set().union(*[set(x.index[x.index<=cutoff]) for x in D.values()]))
for t in Ds:
 a={}; fut={h:{} for h in H}
 for s,x in D.items():
  if t not in x.index: continue
  z=x.loc[:t]; c=z.close.astype(float)
  if len(c)<65: continue
  ret=c.pct_change().iloc[-20:]
  vol=ret.std()
  if not np.isfinite(vol) or vol<=0: continue
  # interpretable path-efficiency momentum: net 20d return divided by path length,
  # with a mild long-term trend confirmation, all information known at t
  net=c.iloc[-1]/c.iloc[-21]-1
  path=ret.abs().sum()
  longtrend=c.iloc[-1]/c.iloc[-61]-1
  sig=(net/path)*(1+0.25*np.tanh(longtrend/(vol*np.sqrt(20))))
  a[s]=sig; rows.append({'date':t.date(),'symbol':s,'signal':sig})
  for h in H:
   f=x.loc[x.index>t].close
   if len(f)>=h: fut[h][s]=f.iloc[h-1]/c.iloc[-1]-1
 if len(a)>=8:
  cov.append(len(a)/15); rr=pd.Series(a).rank(pct=True)
  turn.append(0 if prev is None else (rr-prev.reindex(rr.index).fillna(.5)).abs().mean()); prev=rr
  for h in H:
   com=[s for s in a if s in fut[h]]
   if len(com)>=8:
    q=spearmanr([a[s] for s in com],[fut[h][s] for s in com]).statistic
    if np.isfinite(q): R[h].append((t,q,len(com)))
print('cutoff',cutoff.date(),'universe',len(U))
print('coverage',np.mean(cov),'turnover',np.mean(turn),'dates',len(R[10]),'avgN',np.mean([v[2] for v in R[10]]))
for h,a in R.items():
 q=pd.Series([v[1] for v in a]); n=len(q)
 print('H',h,'dates',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
pd.DataFrame(rows).to_csv('scripts/miner_1_20320308_path_efficiency_confirmed_signal.csv',index=False)
