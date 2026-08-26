import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
cut=pd.Timestamp('2032-05-15'); P=pd.DataFrame({s:D[s].close for s in U}); lr=np.log(P).diff()
horizons=[1,5,10,20]; R={h:[] for h in horizons}; ns={h:[] for h in horizons}; cov=[]; turn=[]; prev=None; rows=[]
for t in sorted(set().union(*[set(x.index[x.index<=cut]) for x in D.values()])):
  # cross-sectional median is formed at t, then signal uses only trailing completed data
  r3=lr.loc[:t].tail(3).sum(); vol=lr.loc[:t].tail(20).std()
  med=r3.median(); sig=(-(r3-med)/vol).replace([np.inf,-np.inf],np.nan).dropna()
  sig=sig[(sig.index.isin(U))]
  if len(sig)<8: continue
  cov.append(len(sig)/15); rank=sig.rank(pct=True)
  turn.append(0 if prev is None else (rank-prev.reindex(rank.index).fillna(.5)).abs().mean()); prev=rank
  for s,v in sig.items(): rows.append({'date':t.date(),'symbol':s,'signal':v})
  for h in horizons:
    yy={}
    for s in sig.index:
      x=D[s]; fut=x.loc[x.index>t].close
      if len(fut)>=h:
        base=x.loc[:t].close.iloc[-1]; yy[s]=fut.iloc[h-1]/base-1
    com=list(set(sig.index)&set(yy))
    if len(com)>=8:
      q=spearmanr(sig[com],pd.Series(yy)[com]).statistic
      if np.isfinite(q): R[h].append(q); ns[h].append(len(com))
print('cutoff',cut.date(),'universe',len(U),'valid_dates',len(cov),'coverage',round(np.mean(cov),5),'turnover',round(np.mean(turn),5))
for h in horizons:
 q=pd.Series(R[h]); print('H',h,'dates',len(q),'avgN',round(np.mean(ns[h]),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
pd.DataFrame(rows).to_csv('scripts/miner_2_20320517_relative_reversal3_signal.csv',index=False)
