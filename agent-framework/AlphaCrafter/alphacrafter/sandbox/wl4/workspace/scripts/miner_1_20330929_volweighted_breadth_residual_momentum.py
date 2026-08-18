import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for a in assets:
 f=f'../persistent/stock_data/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); P[a]=d.close.astype(float)
P=pd.DataFrame(P).sort_index().loc[:'2033-09-28']; R=P.pct_change(); n=len(P.columns)
r30=P.pct_change(30); vol60=R.rolling(60,min_periods=40).std(); invvol=(1/vol60).replace([np.inf,-np.inf],np.nan)
resid=r30.sub(r30.mean(axis=1),axis='index'); base=resid/(vol60*np.sqrt(30))
r10=P.pct_change(10); wb=(invvol*r10.gt(0)).sum(axis=1)/invvol.where(r10.notna()).sum(axis=1)
condition=(wb-0.5).rolling(10,min_periods=10).mean(); F=(base.mul(condition,axis='index')).shift(1)
rows=[]
for dt in F.index:
 fut=P.shift(-10).loc[dt]/P.loc[dt]-1; z=pd.concat([F.loc[dt],fut],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min(),r.date.max(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',n)
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),6))
for k in [120,260,520,780]:
 q=s.tail(k); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
print('coverage',round(F.notna().sum(axis=1).mean()/n,6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for h in [1,5,10,20,30]:
 rows2=[]
 for dt in F.index:
  fut=P.shift(-h).loc[dt]/P.loc[dt]-1; z=pd.concat([F.loc[dt],fut],axis=1).dropna()
  if len(z)>=8: rows2.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,round(np.nanmean(rows2),6),len(rows2))
os.makedirs('scripts/artifacts',exist_ok=True)
r.to_csv('scripts/artifacts/miner_1_20330929_volweighted_breadth_residual_momentum_ic.csv',index=False)
F.to_csv('scripts/artifacts/miner_1_20330929_volweighted_breadth_residual_momentum_signal.csv')
