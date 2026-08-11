import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-03-25')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut]
ix=sorted(set().union(*[set(x.index) for x in P.values()]) | set(v.index))
p=pd.DataFrame({s:x.reindex(ix) for s,x in P.items()}).ffill(); v=v.reindex(ix).ffill()
r=p.pct_change(); vr=v.pct_change()
# VIX-beta penalized momentum: lagged 20d trend, penalizing assets that rise/fall with volatility shocks.
# Stress intensity is lagged VIX percentile above its 120d median; all inputs are lagged before use.
beta=r.rolling(60,min_periods=40).cov(vr).div(vr.rolling(60,min_periods=40).var(),axis=0)
stress=((v-v.rolling(120,min_periods=80).median())/v.rolling(120,min_periods=80).median()).clip(lower=0)
trend=np.log(p/p.shift(20)); f=(trend - 0.75*beta*stress.values[:,None]).shift(1)
print('universe',len(U),'dates',len(p),'cutoff',p.index.max().date())
for h in [5,10,20]:
 I=[];Ns=[];ds=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic; I.append(z); Ns.append(len(q)); ds.append(p.index[i])
 a=np.array(I); print('h',h,'valid_dates',len(a),'avgN',round(np.mean(Ns),2),'coverage',round(np.mean(Ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 print(' annual', {y:round(a[[d.year==y for d in ds]].mean(),6) for y in sorted(set(d.year for d in ds))})
rank=f.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
print('decay: VIX-beta penalized log 20d momentum, stress-weight 0.75, 60d beta/120d median')
