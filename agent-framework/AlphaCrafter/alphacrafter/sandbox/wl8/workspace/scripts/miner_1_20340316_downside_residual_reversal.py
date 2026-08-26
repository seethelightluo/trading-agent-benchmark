import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2034-03-15']
r=np.log(C).diff()
# Residual reversal: reverse recent 10d cross-sectional idiosyncratic move,
# scaled by downside-only idiosyncratic risk over the prior 40 sessions.
mu=r.mean(axis=1)
res=r.sub(mu,axis=0)
raw=-res.rolling(10,min_periods=7).sum().shift(1)
down=res.where(res<0).rolling(40,min_periods=20).std().shift(1)
f=raw/down.replace(0,np.nan)
f=f.clip(f.quantile(.05,axis=1),f.quantile(.95,axis=1),axis=0)
def fw(h): return np.log(C.shift(-h)/C)
def calc(x):
 vals=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 return pd.Series(vals),pd.Series(ns)
i,n=calc(fw(10))
print('candidate downside_residual_reversal_10d')
print('dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,5))
print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),5))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w);print('recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),5))
for h in [1,5,20]: print('decay',h,round(calc(fw(h))[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340316_downside_residual_reversal_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_1_20340316_downside_residual_reversal_ic.csv')
