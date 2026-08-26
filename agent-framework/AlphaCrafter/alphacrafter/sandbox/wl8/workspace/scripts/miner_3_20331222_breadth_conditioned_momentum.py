import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2033-12-21']
r=np.log(C).diff()
# Cross-asset breadth conditions the sign and strength of medium-horizon trend.
mom=np.log(C/C.shift(20)).shift(1)
breadth=(r.rolling(20,min_periods=15).sum()>0).mean(axis=1).shift(1)
f=mom.mul((2*breadth-1),axis=0)

def q(h): return np.log(C.shift(-h)/C)
def calc(x):
 a=[]; n=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));n.append(len(z))
 return pd.Series(a),pd.Series(n)
i,n=calc(q(10)); print('factor breadth_conditioned_momentum_20d','dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4))
print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w); print('recent',w,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
for h in [1,5,20]: print('decay',h,round(calc(q(h))[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20331222_breadth_conditioned_momentum_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_3_20331222_breadth_conditioned_momentum_ic.csv')
