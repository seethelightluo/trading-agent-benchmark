import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2029-06-13')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change(); f=(-px.pct_change(60)*(2*(r.gt(0).rolling(20,min_periods=15).mean()-.5))/(r.rolling(40,min_periods=25).std()*np.sqrt(252)+1e-8)).shift(1)
print('factor existing_contrarian revalidation universe',len(U),'dates',len(px),'cutoff',px.index.max().date())
for start in [pd.Timestamp('2028-01-01'),pd.Timestamp('2029-01-01')]:
 for h in [5,10,20]:
  I=[];N=[]
  for i in range(len(px)-h):
   if px.index[i]<start: continue
   q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);N.append(len(q))
  a=np.array(I); print(start.date(),h,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(252),6),'hit',round((a>0).mean(),4))
