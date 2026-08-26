import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2033-12-21']
r=np.log(C).diff(); m=r.mean(axis=1); e=r.sub(m,axis=0)
disp=r.rolling(20,min_periods=15).std().mean(axis=1)
# lagged cross-asset stress gate, residual 10-day reversal, smoothed
base=-e.rolling(10,min_periods=8).sum().shift(1)
gate=(disp.shift(1)/disp.shift(1).rolling(60,min_periods=30).mean()).clip(.5,2.)
f=base.mul(gate,axis=0).rolling(3,min_periods=3).mean()
def q(h): return np.log(C.shift(-h)/C)
def calc(x):
 a=[];ns=[];ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(d)
 return pd.Series(a,index=ds),pd.Series(ns,index=ds)
i,n=calc(q(10));
print('dates',len(i),'avgN',n.mean(),'coverage',n.mean()/15,'IC',i.mean(),'ICIR',i.mean()/i.std(ddof=1),'hit',(i>0).mean())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for w in [365,750,1260]:
 x=i.tail(w);print('recent',w,x.mean(),x.mean()/x.std(ddof=1))
for h in [1,5,10,20]: print('decay',h,calc(q(h))[0].mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20331222_residual_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_1_20331222_residual_ic.csv')
print('last',f.tail(1).T.to_string(header=False))
