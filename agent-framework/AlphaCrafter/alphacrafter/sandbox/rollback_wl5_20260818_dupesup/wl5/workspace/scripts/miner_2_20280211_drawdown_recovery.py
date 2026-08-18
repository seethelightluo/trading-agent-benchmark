import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-02-11'); D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=end].sort_values('date').set_index('date'); D[s]=x.close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change();
# Drawdown recovery: contrarian score combining distance from 60d low and recent 5d reversal.
# Higher score favors assets near their medium-term low but already showing a short rebound.
low=p.rolling(60,min_periods=40).min(); dist=p/low-1
f=0.5*(-dist.rank(axis=1,pct=True))+0.5*(-r.rolling(5,min_periods=4).sum()).rank(axis=1,pct=True)
# rank terms make scale comparable; higher = contrarian recovery candidate
for h in [1,5,10,20]:
 q=p.shift(-h)/p-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 print('h',h,'dates',len(a),'mean_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/a.ic.std(ddof=1),6),'hit',round((a.ic>0).mean(),4))
 for label,x in [('2020-25',a.loc[:'2025']),('2026-27',a.loc['2026':'2027']),('2027-28',a.loc['2027':'2028'])]:
  if len(x): print(label,len(x),round(x.ic.mean(),6),round(x.ic.mean()/x.ic.std(ddof=1),6))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20280211_drawdown_recovery_signal.csv',index=False)
