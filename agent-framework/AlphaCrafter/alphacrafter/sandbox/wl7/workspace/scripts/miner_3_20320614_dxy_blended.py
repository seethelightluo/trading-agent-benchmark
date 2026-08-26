import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index()['close']
cutoff=pd.Timestamp('2032-06-13'); H=[1,5,10,20]; R={h:[] for h in H}; cov=[]; turns=[]; prev=None; rows=[]
all_dates=sorted(set().union(*[set(D[s].index[D[s].index<=cutoff]) for s in U]))
for t in all_dates:
 if t not in dxy.index: continue
 dx=dxy.loc[:t].dropna()
 if len(dx)<125: continue
 # Continuous DXY trend interaction: blend 20d momentum with 5d reversal.
 d20=dx.iloc[-1]/dx.iloc[-21]-1; hist=dx.pct_change(20).dropna().tail(100)
 if len(hist)<60: continue
 z=(d20-hist.mean())/(hist.std()+1e-12); w=float(np.clip((z+0.5)/2.5,0,1))
 a={}; F={h:{} for h in H}
 for s,x in D.items():
  if t not in x.index: continue
  c=x.loc[:t].close.astype(float); r=np.log(c).diff()
  if len(c)<65: continue
  v=r.tail(40).std()
  if not np.isfinite(v) or v<=1e-8: continue
  sig=((1-w)*r.tail(20).sum()+w*(-r.tail(5).sum()))/v
  a[s]=sig; rows.append({'date':t.date(),'symbol':s,'signal':sig,'dxy_z':z,'blend_reversal':w})
  f=x.loc[x.index>t].close
  for h in H:
   if len(f)>=h: F[h][s]=f.iloc[h-1]/c.iloc[-1]-1
 if len(a)>=8:
  rr=pd.Series(a).rank(pct=True); cov.append(len(a)/15); turns.append(0 if prev is None else (rr-prev.reindex(rr.index).fillna(.5)).abs().mean()); prev=rr
  for h in H:
   zset=[s for s in a if s in F[h]]
   if len(zset)>=8:
    q=spearmanr([a[s] for s in zset],[F[h][s] for s in zset]).statistic
    if np.isfinite(q): R[h].append(q)
print('cutoff',cutoff.date(),'universe',len(U),'dates',len(all_dates),'coverage',round(np.mean(cov),5),'turnover',round(np.mean(turns),5))
for h,a in R.items():
 q=pd.Series(a); print('H',h,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
pd.DataFrame(rows).to_csv('scripts/miner_3_20320614_dxy_blended_signal.csv',index=False)
