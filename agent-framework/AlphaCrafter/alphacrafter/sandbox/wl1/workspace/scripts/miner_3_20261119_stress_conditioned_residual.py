import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut]
idx=sorted(set().union(*[set(v.index) for v in P.values()])|set(vix.index)); p=pd.DataFrame({s:v.reindex(idx) for s,v in P.items()}).ffill(); r=p.pct_change(); v=vix.reindex(idx).ffill()
eq=p[['SPX','NDX','SX5E','000300.SH','HSI','N225']].mean(axis=1); er=eq.pct_change();
# Explicit rolling covariance/beta avoids cross-column alignment ambiguity.
mu=r.rolling(60,min_periods=40).mean(); mue=er.rolling(60,min_periods=40).mean();
cov=r.mul(er,axis=0).rolling(60,min_periods=40).mean()-mu.mul(mue,axis=0)
var=er.pow(2).rolling(60,min_periods=40).mean()-mue.pow(2); beta=cov.div(var.replace(0,np.nan),axis=0)
ret20=p/p.shift(20)-1; eq20=eq/eq.shift(20)-1; resid=ret20-beta.mul(eq20,axis=0)
vz=(v-v.rolling(120,min_periods=60).mean())/(v.rolling(120,min_periods=60).std()+1e-8)
gate=vz.clip(lower=0)+0.25*v.pct_change(10).clip(lower=0).fillna(0); f=resid.mul(gate,axis=0).shift(1)
lo=f.quantile(.05,axis=1); hi=f.quantile(.95,axis=1); f=f.clip(lo,hi,axis=0)
print('universe',len(U),'dates',len(p),'cutoff',p.index.max().date(),'macro_valid',v.notna().sum())
for h in [5,10,20]:
 I=[];Ns=[];ds=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): I.append(z);Ns.append(len(q));ds.append(p.index[i])
 a=np.array(I); print('h',h,'valid_dates',len(a),'avgN',round(np.mean(Ns),2),'coverage',round(np.mean(Ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==20: print('annual20',{y:round(a[[d.year==y for d in ds]].mean(),6) for y in sorted(set(d.year for d in ds))})
rank=f.rank(axis=1,pct=True); turn=(rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna(); print('turnover',round(turn.mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
