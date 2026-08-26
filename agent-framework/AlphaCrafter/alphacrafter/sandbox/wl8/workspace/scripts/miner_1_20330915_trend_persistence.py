import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2033-09-14']
r=np.log(C/C.shift(1)); vol=r.rolling(20,min_periods=15).std()
# Trend persistence: risk-adjusted intermediate return, weighted by directional consistency.
ret20=np.log(C/C.shift(20)); persistence=r.gt(0).rolling(20,min_periods=15).mean()*2-1
f=(ret20/vol)*persistence
fw={h:np.log(C.shift(-h)/C) for h in [1,5,10,20]}
def obs(q, dates):
 a=[]; ns=[]
 for d in dates:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 return pd.Series(a),pd.Series(ns)
all_dates=f.index
for h in [1,5,10,20]:
 i,n=obs(fw[h],all_dates)
 print('horizon',h,'dates',len(i),'avgN',round(n.mean(),3),'IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4))
i,n=obs(fw[10],all_dates); j,m=obs(fw[10],all_dates[::10])
print('coverage',round(n.mean()/15,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 q=i.tail(w);print('recent',w,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
print('nonoverlap dates',len(j),'avgN',round(m.mean(),3),'IC',round(j.mean(),6),'ICIR',round(j.mean()/j.std(ddof=1),6),'hit',round((j>0).mean(),4))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20330915_trend_persistence_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_1_20330915_trend_persistence_ic.csv')
