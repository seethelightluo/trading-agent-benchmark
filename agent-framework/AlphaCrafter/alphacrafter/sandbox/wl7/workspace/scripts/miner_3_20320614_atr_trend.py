import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
cutoff=pd.Timestamp('2032-06-13'); H=[1,5,10,20]; R={h:[] for h in H}; cov=[]; turns=[]; prev=None; rows=[]
all_dates=sorted(set().union(*[set(D[s].index[D[s].index<=cutoff]) for s in U]))
for t in all_dates:
 a={}; F={h:{} for h in H}
 for s,x in D.items():
  if t not in x.index: continue
  q=x.loc[:t].astype(float)
  if len(q)<65: continue
  c=q.close; logret=np.log(c).diff(); tr=np.maximum(q.high-q.low,np.maximum(abs(q.high-c.shift(1)),abs(q.low-c.shift(1))))
  atr=tr.tail(20).mean();
  if not np.isfinite(atr) or atr<=0: continue
  # Smooth directional persistence, normalized by recent true range.
  sig=(c.iloc[-1]/c.iloc[-21]-1)/(atr/c.tail(20).mean())
  if not np.isfinite(sig): continue
  a[s]=sig; rows.append({'date':t.date(),'symbol':s,'signal':sig})
  f=x.loc[x.index>t].close
  for h in H:
   if len(f)>=h: F[h][s]=f.iloc[h-1]/c.iloc[-1]-1
 if len(a)>=8:
  rr=pd.Series(a).rank(pct=True); cov.append(len(a)/15); turns.append(0 if prev is None else (rr-prev.reindex(rr.index).fillna(.5)).abs().mean()); prev=rr
  for h in H:
   z=[s for s in a if s in F[h]]
   if len(z)>=8:
    ic=spearmanr([a[s] for s in z],[F[h][s] for s in z]).statistic
    if np.isfinite(ic): R[h].append(ic)
print('cutoff',cutoff.date(),'universe',len(U),'dates',len(all_dates),'coverage',round(np.mean(cov),5),'turnover',round(np.mean(turns),5))
for h,a in R.items():
 z=pd.Series(a); print('H',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
pd.DataFrame(rows).to_csv('scripts/miner_3_20320614_atr_trend_signal.csv',index=False)
