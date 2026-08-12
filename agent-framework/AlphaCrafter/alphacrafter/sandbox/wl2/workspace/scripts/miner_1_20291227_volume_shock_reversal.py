import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={};V={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None:d=get_index_daily_data(s,3000)
 if d is not None:
  d=d.copy();d.date=pd.to_datetime(d.date);d=d.drop_duplicates('date').set_index('date');P[s]=d.close;V[s]=d.volume
P=pd.DataFrame(P).sort_index();V=pd.DataFrame(V).reindex(P.index);R=P.pct_change(); vr=V/V.rolling(20,min_periods=10).mean(); f=-R.rolling(5,min_periods=5).sum()*np.log1p(vr.clip(lower=0))
rows=[]
for i in range(len(P)-1):
 z=pd.concat([f.iloc[i],R.iloc[i+1]],axis=1).dropna()
 if len(z)>=8:rows.append((P.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
d=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for c in [None,'2028-01-01','2029-01-01','2029-07-01']:
 q=d if c is None else d.loc[c:];print(c or 'full','dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.mean()/15,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean());f.index.name='date';f.reset_index().to_csv('scripts/miner_1_20291227_volume_shock_reversal_signal.csv',index=False)
