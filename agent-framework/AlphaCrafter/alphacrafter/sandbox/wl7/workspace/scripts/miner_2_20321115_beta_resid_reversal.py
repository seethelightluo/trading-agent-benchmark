import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
cut=pd.Timestamp('2032-11-14'); P=pd.DataFrame({s:D[s].close for s in U}); lr=np.log(P).diff(); m=lr[U].mean(axis=1)
ics={h:[] for h in [1,5,10,20]}; ns={h:[] for h in ics}; prev=None; turns=[]; rows=[]
for i,t in enumerate(P.index[P.index<=cut]):
 if i<65: continue
 hist=P.index[P.index<=t]; end=hist[-2]; x=lr.loc[:end].tail(60); mm=m.loc[x.index]
 var=mm.var(); beta=x.apply(lambda z:z.cov(mm)/var if var>0 else np.nan)
 r5=lr.loc[:end].tail(5).sum(); mr=mm.tail(5).sum(); resid=r5-beta*mr
 vol=lr.loc[:end].tail(30).std(); sig=(-resid/vol).replace([np.inf,-np.inf],np.nan).dropna(); sig=sig-sig.median(); rank=sig.rank(pct=True)
 turns.append(0 if prev is None else (rank-prev.reindex(rank.index).fillna(.5)).abs().mean()); prev=rank
 for h in ics:
  yy={}
  for s in sig.index:
   fut=D[s].loc[D[s].index>t].close
   if len(fut)>=h: yy[s]=fut.iloc[h-1]/P.loc[t,s]-1
  com=list(set(sig.index)&set(yy))
  if len(com)>=8:
   q=pd.Series(yy)[com]; z=sig[com]; ok=np.isfinite(z.values)&np.isfinite(q.values); a=spearmanr(z.values[ok],q.values[ok]).statistic if ok.sum()>=8 else np.nan; ics[h].append(a); ns[h].append(len(com)); rows.extend([{'date':t.date(),'symbol':s,'signal':float(sig[s])} for s in com])
for h,a0 in ics.items():
 a=np.array(a0); a=a[np.isfinite(a)]; print(h,'dates',len(a),'avgN',round(np.mean(ns[h]),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a),6),'hit',round(np.mean(a>0),4))
print('turnover',round(np.mean(turns),6),'coverage',round(np.mean(ns[10])/15,6))
pd.DataFrame(rows).drop_duplicates(['date','symbol']).to_csv('scripts/miner_2_20321115_beta_resid_reversal_signal.csv',index=False)
