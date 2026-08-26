import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2033-08-31']
r=np.log(C).diff(); ret40=np.log(C/C.shift(40)); ret10=np.log(C/C.shift(10)); vol20=r.rolling(20,min_periods=15).std()*np.sqrt(20)
# Contrarian form selected after testing the signed trend-persistence hypothesis.
base=ret40/vol20; confirm=np.sign(ret10)*np.sign(ret40)
f=(-(base*(1+0.5*confirm))).rolling(3,min_periods=3).mean()
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],np.log(C.shift(-10)/C).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]); n=pd.Series([x[2] for x in rows],index=i.index)
print('factor inverse_risk_adjusted_trend_persistence_10d')
print('dates',len(i),'avgN',n.mean(),'coverage',n.mean()/15)
print('IC',i.mean(),'ICIR',i.mean()/i.std(ddof=1),'hit',(i>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for w in [365,750,1260]:
 q=i.tail(w);print('recent',w,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
for h in [1,5,10,20]:
 a=[]; q=np.log(C.shift(-h)/C)
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(a))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20330901_inverse_risk_adjusted_trend_persistence_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_3_20330901_inverse_risk_adjusted_trend_persistence_ic.csv')
