import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
cutoff=pd.Timestamp('2032-04-29'); horizons=[1,5,10,20]; R={h:[] for h in horizons}; cov=[]; turns=[]; prev=None; rows=[]
# common daily cross-asset return, formed only from each date's close observations
all_dates=sorted(set().union(*[set(x.index[x.index<=cutoff]) for x in D.values()]))
P=pd.DataFrame({s:D[s].close for s in U}); lr=np.log(P).diff(); common=lr.median(axis=1)
for t in all_dates:
 a={}; F={h:{} for h in horizons}
 for s,x in D.items():
  if t not in x.index: continue
  c=x.loc[:t].close.astype(float); r=np.log(c).diff()
  if len(c)<65: continue
  ix=common.index[(common.index<=t)].intersection(r.index)
  if len(ix)<60: continue
  y=r.reindex(ix).tail(60); m=common.reindex(ix).tail(60)
  ok=y.notna()&m.notna()
  if ok.sum()<40: continue
  xx=m[ok].values; yy=y[ok].values; beta=np.cov(xx,yy,ddof=1)[0,1]/max(np.var(xx,ddof=1),1e-12)
  resid=y-beta*m
  rv=resid.tail(20).std()
  if not np.isfinite(rv) or rv<=1e-8: continue
  # residual 5d shock reversal, volatility normalized
  sig=-resid.tail(5).sum()/rv
  a[s]=sig
  rows.append({'date':t.date(),'symbol':s,'signal':sig})
  for h in horizons:
   f=x.loc[x.index>t].close
   if len(f)>=h: F[h][s]=f.iloc[h-1]/c.iloc[-1]-1
 if len(a)>=8:
  rr=pd.Series(a).rank(pct=True); cov.append(len(a)/15)
  turns.append(0 if prev is None else (rr-prev.reindex(rr.index).fillna(.5)).abs().mean()); prev=rr
  for h in horizons:
   com=[s for s in a if s in F[h]]
   if len(com)>=8:
    q=spearmanr([a[s] for s in com],[F[h][s] for s in com]).statistic
    if np.isfinite(q): R[h].append(q)
print('cutoff',cutoff.date(),'universe',len(U),'coverage',round(np.mean(cov),5),'turnover',round(np.mean(turns),5))
for h,a in R.items():
 q=pd.Series(a); print('H',h,'dates',len(q),'avgN',round(np.mean([len(a)]),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
pd.DataFrame(rows).to_csv('scripts/miner_3_20320503_beta_neutral_residual_reversal5_signal.csv',index=False)
