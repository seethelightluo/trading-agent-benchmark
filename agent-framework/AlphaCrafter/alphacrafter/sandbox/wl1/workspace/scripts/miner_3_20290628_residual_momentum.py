import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2029-06-27')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
# Residual momentum: asset 20d return minus equal-weight cross-asset benchmark return, divided by idiosyncratic 20d residual volatility; lag one day.
bench=r.mean(axis=1); ar=px.pct_change(20); br=bench.rolling(20,min_periods=12).sum(); resid=ar.sub(br,axis=0); rv=(r.sub(bench,axis=0).rolling(20,min_periods=12).std()*np.sqrt(252)); f=(resid/(rv+1e-8)).shift(1)
print('factor residual_momentum_volscaled_20 universe',len(U),'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20]:
 I=[];N=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);N.append(len(q))
 a=np.array(I); print(h,'dates',len(a),'avgN',round(np.mean(N),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(252),6),'hit',round((a>0).mean(),4))
rr=f.rank(axis=1,pct=True); print('coverage',round(f.notna().sum().sum()/f.size,6),'turnover',round(rr.diff().abs().mean(axis=1).mean(),6))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20290628_residual_momentum_signal.csv',index=False)
