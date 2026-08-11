import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-11-18')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
p=pd.DataFrame(P).ffill(); r=p.pct_change()
# Volatility-compression breakout: medium-horizon trend scaled by long-run risk,
# rewarded when recent realized volatility is compressed versus its 60d baseline.
ret20=p/p.shift(20)-1
rv20=r.rolling(20,min_periods=15).std(); rv60=r.rolling(60,min_periods=40).std()
f=(ret20/(rv60+1e-8)*(rv60/(rv20+1e-8)).clip(0.5,2.0)).shift(1)
print('universe',len(U),'dates',len(p),'cutoff',p.index.max().date())
for h in [5,10,20]:
 I=[];Ns=[];ds=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): I.append(z);Ns.append(len(q));ds.append(p.index[i])
 a=np.array(I); print('h',h,'valid_dates',len(a),'avgN',round(np.mean(Ns),2),'coverage',round(np.mean(Ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 print('annual',{y:round(a[[d.year==y for d in ds]].mean(),6) for y in sorted(set(d.year for d in ds))})
rank=f.rank(axis=1,pct=True); turn=(rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna()
print('turnover',round(turn.mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
