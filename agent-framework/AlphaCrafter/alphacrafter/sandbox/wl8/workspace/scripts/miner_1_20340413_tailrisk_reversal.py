import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2034-04-12']
r=np.log(C).diff(); market=r.mean(axis=1); resid=r.sub(market,axis=0)
# Tail-event reversal: recent residual shock, scaled by robust tail-risk frequency; lagged inputs.
shock=resid.rolling(5,min_periods=4).sum().shift(1)
q=resid.rolling(60,min_periods=30).quantile(.1).shift(1)
tail=(resid.lt(q)).rolling(40,min_periods=20).mean().shift(1)
scale=(1+5*tail).clip(1,3)
f=(-shock/(resid.rolling(40,min_periods=20).std().shift(1))).mul(scale)
f=f.rank(axis=1,pct=True); f=f.sub(f.mean(axis=1),axis=0)
def qret(h):return np.log(C.shift(-h)/C)
def calc(x):
 a=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 return pd.Series(a),pd.Series(ns)
i,n=calc(qret(10));print('period',C.index.min().date(),C.index.max().date(),'dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4));print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4));print('turnover',round(f.diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w);print('recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
for h in [1,5,20]:print('decay',h,round(calc(qret(h))[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340413_tailrisk_reversal_signal.csv',index=False);i.rename('ic').to_csv('scripts/miner_1_20340413_tailrisk_reversal_ic.csv')
