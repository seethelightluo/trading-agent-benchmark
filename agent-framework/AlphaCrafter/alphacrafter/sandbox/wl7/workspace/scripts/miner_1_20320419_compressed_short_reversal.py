import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').set_index('date')
cutoff=pd.Timestamp('2032-04-16'); Hs=[1,5,10,20]; R={h:[] for h in Hs}; cov=[]; turns=[]; prev=None; rows=[]
Ds=sorted(set().union(*[set(x.index[x.index<=cutoff]) for x in D.values()]))
for t in Ds:
 a={}; F={h:{} for h in Hs}
 for s,x in D.items():
  if t not in x.index: continue
  z=x.loc[:t]; c=z.close.astype(float); lr=np.log(c).diff()
  if len(c)<75: continue
  r5=c.iloc[-1]/c.iloc[-6]-1; v20=lr.tail(20).std(); v60=lr.tail(60).std()
  if not np.isfinite(v20) or not np.isfinite(v60) or v20<1e-8 or v60<1e-8: continue
  compression=np.clip(v60/v20,0.5,2.0); a[s]=-r5/v20*compression
  rows.append({'date':t.date(),'symbol':s,'signal':a[s]})
  for h in Hs:
   f=x.loc[x.index>t].close
   if len(f)>=h: F[h][s]=f.iloc[h-1]/c.iloc[-1]-1
 if len(a)>=8:
  cov.append(len(a)/15); rr=pd.Series(a).rank(pct=True)
  turns.append(0 if prev is None else (rr-prev.reindex(rr.index).fillna(.5)).abs().mean()); prev=rr
  for h in Hs:
   com=[s for s in a if s in F[h]]
   if len(com)>=8:
    q=spearmanr([a[s] for s in com],[F[h][s] for s in com]).statistic
    if np.isfinite(q): R[h].append((t,q,len(com)))
print('cutoff',cutoff.date(),'universe',len(U))
print('coverage',np.mean(cov),'turnover',np.mean(turns))
for h,a in R.items():
 q=pd.Series([v[1] for v in a]); n=len(q); thirds=[q.iloc[i*n//3:(i+1)*n//3].mean() for i in range(3)]
 print('H',h,'dates',n,'avgN',np.mean([v[2] for v in a]),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'thirds',thirds)
pd.DataFrame(rows).to_csv('scripts/miner_1_20320419_compressed_short_reversal_signal.csv',index=False)
