import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').set_index('date')
cut=pd.Timestamp('2032-03-19'); H=[1,5,10,20]; R={h:[] for h in H}; cov=[]; turns=[]; prev=None; rows=[]
Ds=sorted(set().union(*[set(x.index[x.index<=cut]) for x in D.values()]))
for t in Ds:
 ret={}; vol={}; px={}
 for s,x in D.items():
  if t not in x.index: continue
  c=x.loc[:t].close.astype(float)
  if len(c)<65: continue
  r=c.pct_change(); v=r.iloc[-61:].std()
  if np.isfinite(v) and v>0: ret[s]=c.iloc[-1]/c.iloc[-21]-1; vol[s]=v; px[s]=c.iloc[-1]
 if len(ret)<8: continue
 med=np.median(list(ret.values())); sig={s:-(ret[s]-med)/(vol[s]+1e-12) for s in ret}
 rank=pd.Series(sig).rank(pct=True); cov.append(len(sig)/15); turns.append(0 if prev is None else (rank-prev.reindex(rank.index).fillna(.5)).abs().mean()); prev=rank
 for s in sig: rows.append({'date':t,'symbol':s,'signal':sig[s]})
 for h in H:
  z=[]
  for s in sig:
   f=D[s].loc[D[s].index>t].close
   if len(f)>=h: z.append((sig[s],f.iloc[h-1]/px[s]-1))
  if len(z)>=8:
   q=spearmanr(*zip(*z)).statistic
   if np.isfinite(q): R[h].append(q)
print('cutoff',cut.date(),'universe',15,'coverage',round(np.mean(cov),4),'turnover',round(np.mean(turns),4),'dates',len(R[1]))
for h in H:
 q=pd.Series(R[h]); print('H',h,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
pd.DataFrame(rows).to_csv('scripts/miner_1_20320322_residual20_signal.csv',index=False)
