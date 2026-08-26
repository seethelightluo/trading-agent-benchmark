import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2033-10-26']
r=np.log(C).diff()
# Regime-aware persistence: 60d trend, scaled by its own 30d volatility, smoothed and lagged.
trend=np.log(C/C.shift(60)); vol=r.rolling(30,min_periods=20).std()*np.sqrt(30); f=(trend/vol).rolling(3,min_periods=3).mean()
q=np.log(C.shift(-10)/C); rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]); n=pd.Series([x[2] for x in rows],index=i.index)
print('factor risk_adjusted_trend_60d'); print('period',i.index.min().date(),i.index.max().date(),'dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4))
print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w); print('recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for h in [1,5,10,20]:
 a=[]; qq=np.log(C.shift(-h)/C)
 for d in f.index:
  z=pd.concat([f.loc[d],qq.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(np.nanmean(a),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20331124_risk_adjusted_trend_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_2_20331124_risk_adjusted_trend_ic.csv')
