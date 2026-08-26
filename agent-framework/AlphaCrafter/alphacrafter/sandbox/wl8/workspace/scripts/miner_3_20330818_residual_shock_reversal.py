import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2033-08-17']
r=np.log(C/C.shift(5))
# Equal-weight contemporaneous cross-asset move; residual shock is asset move minus peer average.
bench=r.mean(axis=1)
res=r.sub(bench,axis=0)
# Reversal is stronger after idiosyncratic 5d shocks; smooth to reduce one-day noise.
f=(-res).rolling(3,min_periods=3).mean()
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],np.log(C.shift(-10)/C).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]); n=pd.Series([x[2] for x in rows],index=i.index)
print('dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4),'start',i.index.min(),'end',i.index.max())
print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 q=i.tail(w);print('recent',w,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
for h in [1,5,10,20]:
 q=np.log(C.shift(-h)/C); a=[]
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(np.nanmean(a),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20330818_residual_shock_reversal_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_3_20330818_residual_shock_reversal_ic.csv')
