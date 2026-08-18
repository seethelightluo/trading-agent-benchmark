import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-01-21'); D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date)
 D[s]=x[x.date<=end].sort_values('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); mom=p.pct_change(20); short=p.pct_change(5)
# Residualized medium-term momentum: remove the cross-sectional component explained by recent 5d return.
f=pd.DataFrame(index=p.index,columns=p.columns,dtype=float)
for dt in p.index:
 z=pd.concat([mom.loc[dt],short.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  x=z.iloc[:,1].values; y=z.iloc[:,0].values
  b=np.cov(x,y,ddof=1)[0,1]/(np.var(x,ddof=1)+1e-12); f.loc[dt,z.index]=y-b*x
for h in [1,5,10,20]:
 q=p.shift(-h)/p-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 print('h',h,'dates',len(a),'mean_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/a.ic.std(ddof=1),6),'hit',round((a.ic>0).mean(),4))
 for label,x in [('2020-22',a.loc['2020':'2022']),('2023-25',a.loc['2023':'2025']),('2026-28',a.loc['2026':'2028'])]:
  if len(x): print(label,len(x),round(x.ic.mean(),6),round(x.ic.mean()/x.ic.std(ddof=1),6))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20280121_residualized_momentum_signal.csv',index=False)
