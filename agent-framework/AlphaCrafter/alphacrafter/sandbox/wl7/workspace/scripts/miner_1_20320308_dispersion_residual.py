import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').set_index('date')
cut=pd.Timestamp('2032-03-05'); H=[1,5,10,20]; R={h:[] for h in H}; cov=[]; turn=[]; prev=None; rows=[]
Ds=sorted(set().union(*[set(x.index[x.index<=cut]) for x in D.values()]))
for t in Ds:
 ret={}; vol={}; sig={}
 for s,x in D.items():
  if t not in x.index: continue
  c=x.loc[:t].close.astype(float)
  if len(c)<45: continue
  r=c.pct_change(); v=r.iloc[-21:].std()
  if not np.isfinite(v) or v<=0: continue
  ret[s]=c.iloc[-1]/c.iloc[-6]-1; vol[s]=v
 if len(ret)<8: continue
 disp=np.std(list(ret.values()))
 # conditional residual reversal: stronger when cross-asset dispersion is elevated
 med=np.median(list(ret.values()))
 for s in ret:
  sig[s]=-(ret[s]-med)/vol[s] * (1+0.8*np.tanh(disp/np.median(list(vol.values()))-1))
 fut={h:{} for h in H}
 for s,x in D.items():
  if s not in sig: continue
  c=x.loc[:t].close.astype(float); f=x.loc[x.index>t].close
  for h in H:
   if len(f)>=h: fut[h][s]=f.iloc[h-1]/c.iloc[-1]-1
  rows.append({'date':t.date(),'symbol':s,'signal':sig[s]})
 rr=pd.Series(sig).rank(pct=True); cov.append(len(sig)/15); turn.append(0 if prev is None else (rr-prev.reindex(rr.index).fillna(.5)).abs().mean()); prev=rr
 for h in H:
  com=[s for s in sig if s in fut[h]]
  if len(com)>=8:
   q=spearmanr([sig[s] for s in com],[fut[h][s] for s in com]).statistic
   if np.isfinite(q): R[h].append(q)
print('cutoff',cut.date(),'universe',15,'coverage',np.mean(cov),'turnover',np.mean(turn),'dates',len(R[10]),'avgN',np.mean([len(sig) for _ in R[10]]))
for h in H:
 q=pd.Series(R[h]); print('H',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
pd.DataFrame(rows).to_csv('scripts/miner_1_20320308_dispersion_residual_signal.csv',index=False)
