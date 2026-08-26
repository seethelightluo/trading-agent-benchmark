import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').set_index('date')
cutoff=pd.Timestamp('2032-04-02'); Hs=[1,5,10,20]; R={h:[] for h in Hs}; cov=[]; turn=[]; prev=None
Ds=sorted(set().union(*[set(x.index[x.index<=cutoff]) for x in D.values()]))
for t in Ds:
 a={}; F={h:{} for h in Hs}
 for s,x in D.items():
  if t not in x.index: continue
  z=x.loc[:t]; c=z.close.astype(float); lr=np.log(c).diff().dropna()
  if len(c)<100: continue
  r20=c.iloc[-1]/c.iloc[-21]-1; r60=c.iloc[-1]/c.iloc[-61]-1
  v=lr.tail(60).std(); sig=(0.65*r20+0.35*r60)/(v*np.sqrt(252)+1e-12)
  if np.isfinite(sig):
   a[s]=sig
   for h in Hs:
    f=x.loc[x.index>t].close
    if len(f)>=h: F[h][s]=f.iloc[h-1]/c.iloc[-1]-1
 if len(a)>=8:
  cov.append(len(a)/15); rr=pd.Series(a).rank(pct=True)
  turn.append(0 if prev is None else (rr-prev.reindex(rr.index).fillna(.5)).abs().mean()); prev=rr
  for h in Hs:
   com=[s for s in a if s in F[h]]
   if len(com)>=8:
    q=spearmanr([a[s] for s in com],[F[h][s] for s in com]).statistic
    if np.isfinite(q): R[h].append(q)
print('cutoff',cutoff.date(),'universe',len(U),'coverage',round(np.mean(cov),6),'turnover',round(np.mean(turn),6))
for h,a in R.items():
 q=pd.Series(a); n=len(q); thirds=[q.iloc[i*n//3:(i+1)*n//3].mean() for i in range(3)]
 print('H',h,'dates',n,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'thirds',[round(x,6) for x in thirds])
