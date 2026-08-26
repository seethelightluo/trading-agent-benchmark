import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
cutoff=pd.Timestamp('2032-05-16'); horizons=[1,5,10,20]; R={h:[] for h in horizons}; dates={h:[] for h in horizons}; cov=[]; turns=[]; prev=None; rows=[]
P=pd.DataFrame({s:D[s].close for s in U}); lr=np.log(P).diff()
all_dates=sorted(set().union(*[set(x.index[x.index<=cutoff]) for x in D.values()]))
for t in all_dates:
 a={}; F={h:{} for h in horizons}
 for s,x in D.items():
  if t not in x.index: continue
  c=x.loc[:t].close.astype(float); r=np.log(c).diff()
  if len(c)<65: continue
  v20=r.tail(20).std(); v60=r.tail(60).std()
  if not np.isfinite(v20) or not np.isfinite(v60) or v20<=1e-8: continue
  # volatility-compression-adjusted trend: recent momentum favored when stable, penalized by noisy path
  sig=r.tail(10).sum()/v20*np.sqrt(10)*np.sqrt(np.clip(v60/v20,0.5,2.0))
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
    if np.isfinite(q): R[h].append(q); dates[h].append(t)
print('cutoff',cutoff.date(),'universe',len(U),'available_dates',len(all_dates),'coverage',round(np.mean(cov),5),'turnover',round(np.mean(turns),5))
for h,a in R.items():
 q=pd.Series(a); print('H',h,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if len(q)>=30:
  n=len(q)//3; print(' thirds',*[round(q.iloc[i*n:(i+1)*n].mean(),5) for i in range(3)])
pd.DataFrame(rows).to_csv('scripts/miner_3_20320517_compression_trend_signal.csv',index=False)
