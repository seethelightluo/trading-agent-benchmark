import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2034-01-04']
r=np.log(C).diff()
# Trend signal: lagged 20-day return, rewarded when volatility is compressed versus its own 120-day history.
ret20=r.rolling(20,min_periods=15).sum().shift(1)
vol20=r.rolling(20,min_periods=15).std().shift(1)
base=ret20/(vol20+1e-8)
volbase=vol20.rolling(120,min_periods=60).median()
compression=(volbase/(vol20+1e-8)).clip(.5,2.0)
f=(base*compression).rolling(3,min_periods=3).mean()
q=lambda h: np.log(C.shift(-h)/C)
def calc(x):
 a=[]; ns=[]; dates=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));dates.append(d)
 return pd.Series(a,index=dates),pd.Series(ns,index=dates)
i,n=calc(q(10))
print('dates',len(i),'avgN',round(n.mean(),2),'instruments',15,'coverage',round(n.mean()/15,4),'IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w);print('recent',w,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
for h in [1,5,10,20]:print('decay',h,round(calc(q(h))[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340105_volcompression_trend_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_1_20340105_volcompression_trend_ic.csv')
