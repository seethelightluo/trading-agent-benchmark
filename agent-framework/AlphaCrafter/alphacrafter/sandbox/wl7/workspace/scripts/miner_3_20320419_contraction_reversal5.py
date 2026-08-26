import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
cutoff=pd.Timestamp('2032-04-15'); Hs=[1,5,10,20]; R={h:[] for h in Hs}; cov=[]; turns=[]; prev=None; rows=[]
Ds=sorted(set().union(*[set(x.index[x.index<=cutoff]) for x in D.values()]))
for t in Ds:
 a={}; F={h:{} for h in Hs}
 for s,x in D.items():
  if t not in x.index: continue
  z=x.loc[:t]; c=z.close.astype(float); lr=np.log(c).diff()
  if len(c)<75: continue
  v20=lr.tail(20).std(); v60=lr.tail(60).std()
  if not np.isfinite(v20+v60) or v60<=1e-8: continue
  # short-horizon reversal, strengthened after volatility contraction; lagged through t
  r5=c.iloc[-1]/c.iloc[-6]-1
  a[s]=-r5*np.clip(v60/(v20+1e-8),0.5,2.0)
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
print('coverage',np.mean(cov),'turnover',np.mean(turns),'h10_dates',len(R[10]),'avgN',np.mean([v[2] for v in R[10]]))
for h,a in R.items():
 q=pd.Series([v[1] for v in a]); n=len(q); thirds=[q.iloc[i*n//3:(i+1)*n//3].mean() for i in range(3)]
 print('H',h,'dates',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'thirds',thirds)
pd.DataFrame(rows).to_csv('scripts/miner_3_20320419_contraction_reversal5_signal.csv',index=False)
