import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').set_index('date')
cut=pd.Timestamp('2032-03-19'); H=[1,5,10,20]; R={h:[] for h in H}; cov=[]; turns=[]; prev=None; rows=[]
Ds=sorted(set().union(*[set(x.index[x.index<=cut]) for x in D.values()]))
for t in Ds:
 sig={}; prices={}; rets={}; vols={}
 for s,x in D.items():
  if t not in x.index: continue
  c=x.loc[:t].close.astype(float)
  if len(c)<65: continue
  r=c.pct_change(); v=r.iloc[-41:].std()
  if not np.isfinite(v) or v<=0: continue
  prices[s]=c.iloc[-1]; rets[s]=c.iloc[-4]/c.iloc[-9]-1; vols[s]=v
 if len(rets)<8: continue
 disp=np.std(list(rets.values())); med=np.median(list(rets.values())); crossvol=np.median(list(vols.values()))
 # Smoothed 5-day residual reversal, selectively amplified in high-dispersion days.
 gate=1.0+0.7*np.tanh(disp/(crossvol+1e-12)-1.0)
 for s in rets: sig[s]=-(rets[s]-med)/(vols[s]+1e-12)*gate
 ranks=pd.Series(sig).rank(pct=True); cov.append(len(sig)/15)
 turns.append(0 if prev is None else (ranks-prev.reindex(ranks.index).fillna(.5)).abs().mean()); prev=ranks
 for s in sig: rows.append({'date':t,'symbol':s,'signal':sig[s]})
 for h in H:
  vals=[]
  for s in sig:
   x=D[s]; f=x.loc[x.index>t].close
   if len(f)>=h: vals.append((sig[s],f.iloc[h-1]/prices[s]-1))
  if len(vals)>=8:
   q=spearmanr(*zip(*vals)).statistic
   if np.isfinite(q): R[h].append(q)
print('cutoff',cut.date(),'universe',15,'dates',len(R[1]),'coverage',round(np.mean(cov),4),'turnover',round(np.mean(turns),4),'avgN',round(np.mean([len([s for s in U if s in D and t in D[s].index]) for t in Ds if t<=cut]),2))
for h in H:
 q=pd.Series(R[h]); print('H',h,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
pd.DataFrame(rows).to_csv('scripts/miner_1_20320322_shock_reversal_signal.csv',index=False)
