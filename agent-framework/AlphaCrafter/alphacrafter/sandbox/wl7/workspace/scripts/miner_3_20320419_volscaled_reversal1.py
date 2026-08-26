import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
cutoff=pd.Timestamp('2032-04-15'); R={1:[],5:[],10:[]}; cov=[]; turns=[]; prev=None; rows=[]
Ds=sorted(set().union(*[set(x.index[x.index<=cutoff]) for x in D.values()]))
for t in Ds:
 a={}; F={h:{} for h in R}
 for s,x in D.items():
  if t not in x.index: continue
  z=x.loc[:t]; c=z.close.astype(float); lr=np.log(c).diff()
  if len(c)<65: continue
  v20=lr.tail(20).std(); v60=lr.tail(60).std()
  if not np.isfinite(v20+v60) or v20<=1e-8: continue
  a[s]=-lr.iloc[-1]/v20*np.clip(v60/v20,0.5,2.0)
  rows.append({'date':t.date(),'symbol':s,'signal':a[s]})
  for h in R:
   f=x.loc[x.index>t].close
   if len(f)>=h: F[h][s]=f.iloc[h-1]/c.iloc[-1]-1
 if len(a)>=8:
  cov.append(len(a)/15); rr=pd.Series(a).rank(pct=True); turns.append(0 if prev is None else (rr-prev.reindex(rr.index).fillna(.5)).abs().mean()); prev=rr
  for h in R:
   com=[s for s in a if s in F[h]]
   if len(com)>=8:
    q=spearmanr([a[s] for s in com],[F[h][s] for s in com]).statistic
    if np.isfinite(q): R[h].append(q)
print('cutoff',cutoff.date(),'universe',len(U),'coverage',np.mean(cov),'turnover',np.mean(turns))
for h,a in R.items():
 q=pd.Series(a); print('H',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
pd.DataFrame(rows).to_csv('scripts/miner_3_20320419_volscaled_reversal1_signal.csv',index=False)
