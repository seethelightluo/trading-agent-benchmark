import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-06-03')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
ix=sorted(set().union(*[set(x.index) for x in P.values()])); p=pd.DataFrame({s:x.reindex(ix) for s,x in P.items()}).ffill(); r=p.pct_change()
# Drawdown recovery quality: closeness to 60d high, penalized by downside deviation over prior 20d; lagged one day.
high=p.rolling(60,min_periods=40).max(); recovery=(p/high-1)
down=r.where(r<0,0).rolling(20,min_periods=15).std()
f=(recovery/(down*np.sqrt(20)).replace(0,np.nan)).shift(1)
print('universe',len(U),'dates',len(p),'cutoff',p.index.max().date())
for h in [5,10,20]:
 I=[];Ns=[];ds=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);Ns.append(len(q));ds.append(p.index[i])
 a=np.array(I); print('h',h,'valid_dates',len(a),'avgN',round(np.mean(Ns),2),'coverage',round(np.mean(Ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==20:
  print('annual20',{y:round(a[[d.year==y for d in ds]].mean(),6) for y in sorted(set(d.year for d in ds))})
rank=f.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
for name,lo,hi in [('2024',pd.Timestamp('2024-01-01'),pd.Timestamp('2024-12-31')),('2025',pd.Timestamp('2025-01-01'),pd.Timestamp('2025-12-31')),('2026+',pd.Timestamp('2026-01-01'),cut)]:
 I=[]
 for i in range(len(p)-20):
  if not(lo<=p.index[i]<=hi):continue
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+20]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic)
 a=np.array(I);print(name,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6))
