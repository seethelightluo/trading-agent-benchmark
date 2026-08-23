import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
P=pd.DataFrame(P).sort_index().loc[:'2035-06-20']; r=P.pct_change(); vol=r.rolling(10,min_periods=7).std(); mom=P.pct_change(10)
F=(-mom/vol).sub((-mom/vol).mean(axis=1),axis=0).shift(1)
def evalh(h):
 rows=[]
 for dt in F.index:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): rows.append((dt,len(q),v))
 a=pd.DataFrame(rows,columns=['date','n','ic'])
 return a
A10=evalh(10)
print('dates',len(A10),'avgN',round(A10.n.mean(),2),'coverage',round(A10.n.sum()/(len(A10)*15),4))
print('full',*[round(x,6) for x in [A10.ic.mean(),A10.ic.mean()/A10.ic.std(ddof=1),(A10.ic>0).mean()]])
for k in [120,260,520,780]:
 q=A10.tail(k); print('recent',k,round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),6),round((q.ic>0).mean(),4))
for h in [1,5,10,20]:
 z=evalh(h).ic; print('decay',h,round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
os.makedirs('scripts/artifacts',exist_ok=True); F.to_csv('scripts/artifacts/miner_2_20350621_short_reversal_10d_signal.csv',index_label='date'); A10.to_csv('scripts/artifacts/miner_2_20350621_short_reversal_10d_ic.csv',index=False)
