import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float).replace(0,np.nan) for s in U}).sort_index().loc[:'2033-07-06']
lr=np.log(p).diff(); r5=np.log(p/p.shift(5)); vol20=lr.rolling(20,min_periods=15).std(); atr= (p.diff().abs()/p.shift(1)).rolling(20,min_periods=15).mean()
# short-horizon reversal, scaled by recent volatility and smoothed; negative forecast direction
raw=r5/(vol20+atr+1e-12)
lo=raw.quantile(.10,axis=1); hi=raw.quantile(.90,axis=1); raw=raw.clip(lo,hi,axis=0)
f=raw.rank(axis=1,pct=True).sub(.5).rolling(3,min_periods=3).mean()
rows=[]
for d in f.index:
 q=np.log(p.shift(-10)/p).loc[d]; z=pd.concat([f.loc[d],q],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]); ns=np.array([x[2] for x in rows])
print('dates',len(i),'avgN',round(ns.mean(),3),'coverage',round(ns.mean()/15,6))
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for h in [1,5,10,20]:
 a=[]
 for d in f.index:
  z=pd.concat([f.loc[d],np.log(p.shift(-h)/p).loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 x=pd.Series(a);print('decay',h,round(x.mean(),6),len(x))
for n in [365,750,1260]:
 x=i.tail(n);print('recent',n,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),len(x))
# save artifact for provenance
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20330707_volatility_scaled_short_reversal_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_1_20330707_volatility_scaled_short_reversal_ic.csv')
