import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2034-02-15']
r=np.log(C).diff(); e=r.sub(r.mean(axis=1),axis=0)
# Lagged 10-session residual reversal; amplify only when cross-asset return dispersion is high.
disp=r.rolling(20,min_periods=15).std().mean(axis=1).shift(1)
med=disp.rolling(252,min_periods=100).median().shift(1)
gate=(disp/med).clip(0.5,2.0)
base=(-e.rolling(10,min_periods=8).sum().shift(1))
f=base.mul(gate,axis=0).rolling(3,min_periods=3).mean()
def q(h): return np.log(C.shift(-h)/C)
def calc(x):
 a=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 return pd.Series(a),pd.Series(ns)
i,n=calc(q(10)); print('factor dispersion_conditioned_residual_reversal_10d','dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4))
print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w);print('recent',w,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
for h in [1,5,20]:print('decay',h,round(calc(q(h))[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20340216_dispersion_conditioned_reversal_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_2_20340216_dispersion_conditioned_reversal_ic.csv')
