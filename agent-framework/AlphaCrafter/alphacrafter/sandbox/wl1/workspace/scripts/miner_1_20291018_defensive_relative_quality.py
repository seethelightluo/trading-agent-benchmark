import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D=['XAU','US10Y','CN10Y']; cut=pd.Timestamp('2029-10-17')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change(); dr=r[D].mean(axis=1)
# Asset 20d return relative to defensive basket, scaled by asset downside volatility; lag one day.
rel=r.rolling(20,min_periods=15).sum().sub(dr.rolling(20,min_periods=15).sum(),axis=0); dv=r.clip(upper=0).rolling(40,min_periods=30).std(); f=(rel/(dv*np.sqrt(20)+1e-8)).shift(1)
for h in [10,20]:
 I=[];ds=[];ns=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);ds.append(px.index[i]);ns.append(len(q))
 a=np.array(I);d=pd.DatetimeIndex(ds);print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
 for lab,m in [('2020-25',d<pd.Timestamp('2026-01-01')),('2026+',d>=pd.Timestamp('2026-01-01')),('2028+',d>=pd.Timestamp('2028-01-01')),('2029YTD',d>=pd.Timestamp('2029-01-01'))]:
  z=a[m];print(lab,len(z),z.mean(),z.mean()/z.std(ddof=1))
 if h==20:f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20291018_defensive_relative_quality_signal.csv',index=False)
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'assets',len(U),'dates',len(px))
