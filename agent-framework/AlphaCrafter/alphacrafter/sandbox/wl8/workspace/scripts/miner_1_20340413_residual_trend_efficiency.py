import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2034-04-12']
r=np.log(C).diff(); m=r.mean(axis=1); resid=r.sub(m,axis=0)
# Intermediate-horizon residual trend, risk normalized; all observations lagged one session.
trend=resid.rolling(20,min_periods=15).sum().shift(1)
vol=resid.rolling(40,min_periods=25).std().shift(1)
raw=trend/vol.replace(0,np.nan)
# reward broad, persistent trends and suppress isolated cross-asset moves
breadth=(resid>0).rolling(10,min_periods=7).mean().mean(axis=1).shift(1)
gate=((breadth-0.5).abs()*2).clip(0.25,1.0)
f=raw.mul(gate,axis=0)
f=f.rank(axis=1,pct=True)
# mild two-day signal smoothing to control turnover
f=(f+f.shift(1))/2
f=f.sub(f.mean(axis=1),axis=0)
def q(h): return np.log(C.shift(-h)/C)
def calc(x):
 a=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 return pd.Series(a),pd.Series(ns)
i,n=calc(q(10))
print('period',C.index.min().date(),C.index.max().date(),'dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4))
print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4))
print('turnover',round(f.diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w);print('recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
for h in [1,5,20]:print('decay',h,round(calc(q(h))[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340413_residual_trend_efficiency_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_1_20340413_residual_trend_efficiency_ic.csv')
