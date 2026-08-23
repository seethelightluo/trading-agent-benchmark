import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
P=pd.DataFrame(P).sort_index().loc[:'2035-06-06']; r=P.pct_change(); vol=r.rolling(20,min_periods=15).std(); mom=P.pct_change(30)
F=(-mom/vol).sub((-mom/vol).mean(axis=1),axis=0).shift(1)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8:
  ic=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
  if np.isfinite(ic): rows.append((dt,len(q),ic))
a=pd.DataFrame(rows,columns=['date','n','ic'])
print('dates',len(a),'avgN',round(a.n.mean(),2),'coverage',round(a.n.sum()/(len(a)*15),4)); print('full',*[round(x,6) for x in [a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean()]])
for k in [120,260,520,780]:
 q=a.tail(k); print('recent',k,round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),6),round((q.ic>0).mean(),4))
for h in [1,5,10,20]:
 z=[]
 for dt in F.index:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): z.append(v)
 z=pd.Series(z); print('decay',h,round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
os.makedirs('scripts/artifacts',exist_ok=True); F.to_csv('scripts/artifacts/miner_2_20350607_crossasset_reversal_30d_signal.csv',index_label='date'); a.to_csv('scripts/artifacts/miner_2_20350607_crossasset_reversal_30d_ic.csv',index=False)
