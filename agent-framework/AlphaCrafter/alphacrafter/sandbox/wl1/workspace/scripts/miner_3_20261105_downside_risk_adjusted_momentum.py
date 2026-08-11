import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
idx=sorted(set().union(*[set(v.index) for v in P.values()])); p=pd.DataFrame({s:v.reindex(idx) for s,v in P.items()}).ffill(); r=p.pct_change()
ret20=p/p.shift(20)-1
# Downside deviation uses all observations, assigning zero to non-negative returns.
down=np.sqrt((r.clip(upper=0)**2).rolling(40,min_periods=20).mean())
f=(ret20/(down*np.sqrt(252)+1e-8)).shift(1)
lo=f.quantile(.05,axis=1); hi=f.quantile(.95,axis=1); f=f.clip(lo,hi,axis=0)
print('universe',len(U),'dates',len(p),'cutoff',p.index.max().date())
for h in [5,10,20]:
 I=[];Ns=[];ds=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): I.append(z);Ns.append(len(q));ds.append(p.index[i])
 a=np.array(I); print('h',h,'valid_dates',len(a),'avgN',round(np.mean(Ns),2),'coverage',round(np.mean(Ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==20: print('annual20',{y:round(a[[d.year==y for d in ds]].mean(),6) for y in sorted(set(d.year for d in ds))})
rank=f.rank(axis=1,pct=True); turn=(rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna()
print('turnover',round(turn.mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
