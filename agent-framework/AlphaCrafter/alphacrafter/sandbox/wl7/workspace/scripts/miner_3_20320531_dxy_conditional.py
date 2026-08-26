import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index()['close']
cutoff=pd.Timestamp('2032-05-30'); H=[1,5,10,20]; R={h:[] for h in H}; cov=[]; turns=[]; prev=None; rows=[]
all_dates=sorted(set().union(*[set(D[s].index[D[s].index<=cutoff]) for s in U]))
for t in all_dates:
 if t not in dxy.index: continue
 dx=dxy.loc[:t].dropna()
 if len(dx)<65: continue
 # Strong dollar regime: use short-term reversal; otherwise medium-term momentum.
 # All inputs end at t, and forward returns start after t.
 strong=dx.iloc[-1]/dx.iloc[-21]-1 > 0
 a={}; F={h:{} for h in H}
 for s,x in D.items():
  if t not in x.index: continue
  c=x.loc[:t].close.astype(float); r=np.log(c).diff()
  if len(c)<65: continue
  vol=r.tail(20).std()
  if not np.isfinite(vol) or vol<=1e-8: continue
  sig=((-r.tail(5).sum()) if strong else r.tail(20).sum())/vol
  a[s]=sig; rows.append({'date':t.date(),'symbol':s,'signal':sig,'strong_dxy':int(strong)})
  f=x.loc[x.index>t].close
  for h in H:
   if len(f)>=h: F[h][s]=f.iloc[h-1]/c.iloc[-1]-1
 if len(a)>=8:
  rr=pd.Series(a).rank(pct=True); cov.append(len(a)/15); turns.append(0 if prev is None else (rr-prev.reindex(rr.index).fillna(.5)).abs().mean()); prev=rr
  for h in H:
   z=[s for s in a if s in F[h]]
   if len(z)>=8:
    q=spearmanr([a[s] for s in z],[F[h][s] for s in z]).statistic
    if np.isfinite(q): R[h].append(q)
print('cutoff',cutoff.date(),'universe',len(U),'dates',len(all_dates),'coverage',round(np.mean(cov),5),'turnover',round(np.mean(turns),5))
for h,a in R.items():
 q=pd.Series(a); print('H',h,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
pd.DataFrame(rows).to_csv('scripts/miner_3_20320531_dxy_conditional_signal.csv',index=False)
