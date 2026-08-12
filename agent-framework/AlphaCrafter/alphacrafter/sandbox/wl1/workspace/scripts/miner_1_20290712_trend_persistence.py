import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2029-07-11')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}; idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
# Trend persistence: 40d return multiplied by fraction of positive days, scaled by realized volatility; lagged one day.
ret=px.pct_change(40); persist=(r.gt(0).rolling(40,min_periods=30).mean()-0.5)*2; vol=r.rolling(40,min_periods=30).std()*np.sqrt(252); f=(ret*persist/(vol+1e-8)).shift(1)
print('factor trend_persistence_volscaled_40 universe',len(U),'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20]:
 I=[];N=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);N.append(len(q))
 a=np.array(I); print(h,'dates',len(a),'avgN',round(np.mean(N),2),'IC',round(a.mean(),6),'ICIR_daily',round(a.mean()/(a.std(ddof=1)+1e-12),6),'ICIR_ann',round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(252),6),'hit',round((a>0).mean(),4))
rr=f.rank(axis=1,pct=True); print('coverage',round(f.notna().sum().sum()/f.size,6),'turnover',round(rr.diff().abs().mean(axis=1).mean(),6)); out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20290712_trend_persistence_signal.csv',index=False)
