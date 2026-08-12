import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-06-27')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
# Recovery velocity: medium-term return rewarded, but penalize assets still deeply below their 120d peak; lagged one day.
high=px.rolling(120,min_periods=60).max(); dd=(px/high-1).clip(upper=0)
vol=r.rolling(30,min_periods=20).std()*np.sqrt(252)
f=((px.pct_change(30)/(vol+1e-8))*(1+dd)).shift(1)
print('factor recovery_velocity_30_120 universe',len(U),'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20]:
 I=[];N=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);N.append(len(q))
 a=np.array(I); print(h,'dates',len(a),'avgN',round(np.mean(N),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(252),6),'hit',round((a>0).mean(),4))
 for label,mask in [('2020-25',pd.Series(px.index.year<=2025,index=px.index)),('2026+',pd.Series(px.index.year>=2026,index=px.index)),('2028+',pd.Series(px.index.year>=2028,index=px.index)),('2029YTD',pd.Series((px.index.year==2029),index=px.index))]:
  z=[]
  for i in range(len(px)-h):
   if not mask.iloc[i]: continue
   q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:z.append(spearmanr(q.f,q.y).statistic)
  z=np.array(z); print(' ',label,len(z),round(z.mean(),6) if len(z) else None,round(z.mean()/(z.std(ddof=1)+1e-12)*np.sqrt(252),6) if len(z)>1 else None)
rr=f.rank(axis=1,pct=True); print('coverage',round(f.notna().sum().sum()/f.size,6),'turnover',round(rr.diff().abs().mean(axis=1).mean(),6))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20290628_recovery_velocity_signal.csv',index=False)
