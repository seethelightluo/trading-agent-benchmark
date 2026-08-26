import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2034-04-26']
r=np.log(C).diff(); cs=r.mean(axis=1); idio=r.sub(cs,axis=0)
# Adaptive tail reversal: reverse 10d lagged residual move, scale by downside risk,
# and increase signal only when recent downside-tail breadth is elevated.
raw=-idio.rolling(10,min_periods=7).sum().shift(1)
down=idio.where(idio<0)
risk=down.rolling(40,min_periods=20).std().shift(1)
tail=(idio < -idio.rolling(40,min_periods=20).std().shift(1)).sum(axis=1)/C.notna().sum(axis=1)
bread=tail.rolling(5,min_periods=3).mean().shift(1)
base=raw/risk.replace(0,np.nan)
gate=(1+2*bread.clip(0,.8)).clip(1,2.6)
base=base.mul(gate,axis=0)
rank=base.rank(axis=1,pct=True)
f=rank.rolling(3,min_periods=2).mean(); f=f.sub(f.mean(axis=1),axis=0)
def q(h): return np.log(C.shift(-h)/C)
def calc(x):
 a=[];ns=[];ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(d)
 return pd.Series(a,index=ds),pd.Series(ns,index=ds)
i,n=calc(q(10)); print('end',C.index.max().date(),'dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4)); print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4)); print('turnover',round(f.diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w); print('recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
for h in [1,5,20]: print('decay',h,round(calc(q(h))[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340427_adaptive_tail_reversal_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_1_20340427_adaptive_tail_reversal_ic.csv')
