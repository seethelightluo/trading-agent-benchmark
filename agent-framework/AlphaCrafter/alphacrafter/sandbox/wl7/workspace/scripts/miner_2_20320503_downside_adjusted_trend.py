import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
cutoff=pd.Timestamp('2032-04-30'); horizons=[1,5,10,20]; out={h:[] for h in horizons}; ns=[]; turns=[]; prev=None; rows=[]
Ds=sorted(set().union(*[set(x.index[x.index<=cutoff]) for x in D.values()]))
for t in Ds:
 sig={}; fwd={h:{} for h in horizons}
 for s,x in D.items():
  if t not in x.index: continue
  z=x.loc[:t]; c=z.close.astype(float); r=np.log(c).diff()
  if len(c)<65: continue
  # Downside deviation over the trailing 40 sessions; explicit min count avoids NaN collapse.
  neg=r.tail(40).where(r.tail(40)<0).dropna()
  if len(neg)<5: continue
  down=np.sqrt(np.mean(neg**2))
  mom=np.log(c.iloc[-1]/c.iloc[-21]) if len(c)>=21 else np.nan
  if not np.isfinite(down+mom) or down<=1e-8: continue
  sig[s]=mom/down
  rows.append({'date':t.date(),'symbol':s,'signal':sig[s]})
  future=x.loc[x.index>t].close
  for h in horizons:
   if len(future)>=h: fwd[h][s]=future.iloc[h-1]/c.iloc[-1]-1
 if len(sig)>=8:
  ns.append(len(sig)/15); ranks=pd.Series(sig).rank(pct=True)
  turns.append(0 if prev is None else (ranks-prev.reindex(ranks.index).fillna(.5)).abs().mean()); prev=ranks
  for h in horizons:
   common=[s for s in sig if s in fwd[h]]
   if len(common)>=8:
    q=spearmanr([sig[s] for s in common],[fwd[h][s] for s in common]).statistic
    if np.isfinite(q): out[h].append(q)
print('cutoff',cutoff.date(),'universe',len(U),'dates',len(Ds),'coverage',round(float(np.mean(ns)),4),'turnover',round(float(np.mean(turns)),4))
for h,v in out.items():
 q=pd.Series(v); print('H',h,'valid_dates',len(q),'avgN>=8 dates','IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
pd.DataFrame(rows).to_csv('scripts/miner_2_20320503_downside_adjusted_trend_signal.csv',index=False)
