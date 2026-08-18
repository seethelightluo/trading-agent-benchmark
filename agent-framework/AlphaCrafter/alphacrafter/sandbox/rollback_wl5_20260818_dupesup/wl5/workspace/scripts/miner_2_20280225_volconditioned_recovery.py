import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-02-25'); D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=end].sort_values('date').set_index('date'); D[s]=x.close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# Volatility-conditioned drawdown recovery: favor assets near 60d lows,
# with recent rebound pressure, while preferring lower realized 20d risk.
low=p.rolling(60,min_periods=40).min(); dist=p/low-1
rev=(-r.rolling(5,min_periods=4).sum()).rank(axis=1,pct=True)
near=(-dist).rank(axis=1,pct=True)
lowvol=(-r.rolling(20,min_periods=15).std()).rank(axis=1,pct=True)
f=(near+rev+lowvol)/3
metrics=[]
for h in [1,5,10,20]:
 q=p.shift(-h)/p-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 ic=a.ic.mean(); ir=ic/a.ic.std(ddof=1)
 print('h',h,'dates',len(a),'mean_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((a.ic>0).mean(),4))
 for label,x in [('2020-25',a.loc[:'2025']),('2026-27',a.loc['2026':'2027']),('2027-28',a.loc['2027':'2028'])]:
  if len(x): print(label,len(x),round(x.ic.mean(),6),round(x.ic.mean()/x.ic.std(ddof=1),6))
rank=f.rank(axis=1,pct=True); print('turnover',round(rank.diff().abs().mean(axis=1).mean(),6))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20280225_volconditioned_recovery_signal.csv',index=False)
