import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2033-12-07']
r=np.log(C).diff(); market=r.mean(axis=1); resid=r.sub(market,axis=0)
# Idiosyncratic risk-normalized residual reversal: fade 10d residual move,
# scaled by trailing 30d residual volatility; all inputs are lagged by one day.
idvol=resid.rolling(30,min_periods=20).std().shift(1)
f=(-resid.rolling(10,min_periods=8).sum().shift(1)/idvol.replace(0,np.nan)).rolling(3,min_periods=3).mean()
def future(h): return np.log(C.shift(-h)/C)
def calc(x):
 rows=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 i=pd.Series([a[1] for a in rows],index=[a[0] for a in rows]); n=pd.Series([a[2] for a in rows],index=i.index)
 return i,n
i,n=calc(future(10)); print('factor idio_volnorm_residual_reversal_10d')
print('period',i.index.min().date(),i.index.max().date(),'dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4))
print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w); print('recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for h in [1,5,10,20]:
 a,_=calc(future(h)); print('decay',h,round(a.mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20331208_idio_volnorm_residual_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_3_20331208_idio_volnorm_residual_ic.csv')
