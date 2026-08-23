import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-06-02'); W=60; H=5; D={}
for s in U:
 x=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date').set_index('date')
 x=x[x.index<=END]; x['r1']=x.close.pct_change(); x['r5']=x.close.pct_change(H); x['f1']=x.close.shift(-1)/x.close-1; D[s]=x
# Equal-weight cross-asset daily return, with only contemporaneous completed data
allr=pd.concat({s:x.r1 for s,x in D.items()},axis=1).sort_index(); mkt=allr.mean(axis=1,skipna=True); rows=[]
for s,x in D.items():
 beta=x.r1.rolling(W,min_periods=40).cov(mkt.reindex(x.index))/mkt.reindex(x.index).rolling(W,min_periods=40).var()
 # factor is 5d residual momentum: asset 5d return less beta times market 5d return
 mr=mkt.rolling(H).sum().reindex(x.index)
 x['sig']=x.r5-beta*mr
 D[s]=x
for d in sorted(set().union(*[set(x.index) for x in D.values()])):
 a=[]
 for s,x in D.items():
  if d in x.index:
   q=x.loc[d]
   if np.isfinite([q.sig,q.f1]).all(): a.append((s,q.sig,q.f1))
 if len(a)>=8:
  z=pd.DataFrame(a,columns=['s','sig','f']); ic=spearmanr(z.sig,z.f).statistic
  if np.isfinite(ic): rows.append((d,ic,len(a)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('factor beta-neutral 5d momentum | end',END.date())
print('dates',len(r),'rows',int(r.n.sum()),'avg_names',round(r.n.mean(),2),'coverage',round(r.n.sum()/(len(r)*15),4))
print('daily IC',round(r.ic.mean(),6),'ICIR',round(r.ic.mean()/r.ic.std(ddof=1),6),'hit',round((r.ic>0).mean(),4))
for y,g in r.groupby(r.index.year): print('regime',y,'dates',len(g),'IC',round(g.ic.mean(),6),'ICIR',round(g.ic.mean()/g.ic.std(ddof=1),6))
for h in [252,504]:
 g=r.tail(h); print('recent',h,'dates',len(g),'IC',round(g.ic.mean(),6),'ICIR',round(g.ic.mean()/g.ic.std(ddof=1),6))
# lagged signal turnover using rank correlation on common names
prev=None; ts=[]
for d in r.index:
 cur={s:D[s].loc[d,'sig'] for s in U if d in D[s].index and np.isfinite(D[s].loc[d,'sig'])}
 if prev is not None:
  common=set(cur)&set(prev)
  if len(common)>=8: ts.append(1-spearmanr([cur[s] for s in common],[prev[s] for s in common]).statistic)
 prev=cur
print('turnover_proxy_1d',round(float(np.nanmean(ts)),4),'turnover_obs',len(ts))
