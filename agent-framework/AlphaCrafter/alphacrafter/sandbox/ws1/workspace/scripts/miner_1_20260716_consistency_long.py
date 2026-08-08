import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
px={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/index_data/'+s+'.csv'
 d=pd.read_csv(f,parse_dates=['date']).sort_values('date'); d=d[d.date<=end][['date','close']].dropna().drop_duplicates('date'); px[s]=d.set_index('date').close
D=sorted(set().union(*[set(x.index) for x in px.values()]))
for w in [40,60]:
 out={h:[] for h in [1,5,10]}; ns=[]; turns=[]; prev=None
 for t in D:
  fac={}; fw={h:{} for h in [1,5,10]}
  for s,p in px.items():
   if t not in p.index: continue
   hist=p.loc[:t].tail(w+1)
   if len(hist)<w+1: continue
   fac[s]=float((hist.pct_change().dropna()>0).mean())
   fut=p.loc[p.index>t]
   for h in fw:
    if len(fut)>=h: fw[h][s]=float(fut.iloc[h-1]/p.loc[t]-1)
  ns.append(len(fac)); ranks=pd.Series(fac).rank(pct=True)
  if prev is not None:
   c=set(prev.index)&set(ranks.index)
   if len(c)>=8: turns.append(np.mean([abs(ranks[x]-prev[x]) for x in c]))
  prev=ranks
  for h in out:
   q=pd.Series(fw[h]); z=pd.Series(fac); c=z.index.intersection(q.index)
   if len(c)>=8: out[h].append(spearmanr(z[c],q[c]).statistic)
 print('W',w,'dates',len(D),'meanN',np.mean(ns),'coverage',np.mean(ns)/15,'turn',np.mean(turns),'turnN',len(turns))
 for h,a in out.items():
  a=np.array(a); print(' H',h,'N',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
 print('period',D[0],D[-1])
