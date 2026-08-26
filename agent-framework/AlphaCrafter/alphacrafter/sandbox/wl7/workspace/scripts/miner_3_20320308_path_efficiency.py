import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').set_index('date')
cutoff=pd.Timestamp('2032-03-05')
Hs=[1,5,10,20]; R={h:[] for h in Hs}; cov=[]; turn=[]; prev=None; rows=[]
Ds=sorted(set().union(*[set(x.index[x.index<=cutoff]) for x in D.values()]))
for t in Ds:
 a={}; fut={h:{} for h in Hs}
 for s,x in D.items():
  if t not in x.index: continue
  z=x.loc[:t]; c=z.close.astype(float)
  if len(c)<65: continue
  lr=np.log(c/c.shift(1)).dropna(); r20=c.iloc[-1]/c.iloc[-21]-1
  path=lr.iloc[-20:].abs().sum(); vol=lr.iloc[-40:].std()
  if not np.isfinite(path) or path<=0 or not np.isfinite(vol) or vol<=0: continue
  # directional path efficiency, volatility normalized; lag-safe
  a[s]=(r20/path)/(vol*np.sqrt(20))
  rows.append({'date':t.date(),'symbol':s,'signal':a[s]})
  for h in Hs:
   f=x.loc[x.index>t].close
   if len(f)>=h: fut[h][s]=f.iloc[h-1]/c.iloc[-1]-1
 if len(a)>=8:
  cov.append(len(a)/15); rr=pd.Series(a).rank(pct=True)
  turn.append(0 if prev is None else (rr-prev.reindex(rr.index).fillna(.5)).abs().mean()); prev=rr
  for h in Hs:
   qv=[s for s in a if s in fut[h]]
   if len(qv)>=8:
    q=spearmanr([a[s] for s in qv],[fut[h][s] for s in qv]).statistic
    if np.isfinite(q): R[h].append((t,q,len(qv)))
print('cutoff',cutoff.date(),'universe',len(U))
print('coverage',np.mean(cov),'turnover',np.mean(turn),'H10_dates',len(R[10]),'H10_avgN',np.mean([v[2] for v in R[10]]))
for h,aa in R.items():
 q=pd.Series([v[1] for v in aa]); n=len(q); thirds=[q.iloc[i*n//3:(i+1)*n//3].mean() for i in range(3)]
 print('H',h,'dates',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'thirds',thirds)
pd.DataFrame(rows).to_csv('scripts/miner_3_20320308_path_efficiency_signal.csv',index=False)
