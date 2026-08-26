import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2034-02-15']
r=np.log(C).diff()
# Risk-adjusted intermediate trend: lagged 40-session return divided by lagged 20-session realized volatility.
# Cross-sectional ranks reduce scale differences across asset classes.
ret40=np.log(C/C.shift(40)).shift(1)
vol20=r.rolling(20,min_periods=15).std().shift(1)*np.sqrt(20)
f=(ret40/vol20).rank(axis=1,pct=True)
# smooth ranks to limit rebalance noise
f=f.rolling(3,min_periods=3).mean()
fr=np.log(C.shift(-10)/C)
ics=[]; ns=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
i=pd.Series(ics);n=pd.Series(ns)
print('candidate risk_adjusted_trend_40d_10d dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4))
print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4),'turnover',round(f.diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w);print('recent',w,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
for h in [1,5,20]:
 x=[]
 fh=np.log(C.shift(-h)/C)
 for d in f.index:
  z=pd.concat([f.loc[d],fh.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(pd.Series(x).mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340216_risk_adjusted_trend_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_1_20340216_risk_adjusted_trend_ic.csv')
