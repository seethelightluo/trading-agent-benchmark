import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in A}
P=pd.DataFrame({a:d.close.astype(float) for a,d in D.items()}).sort_index().loc[:'2035-11-07']
H=pd.DataFrame({a:d.high.astype(float) for a,d in D.items()}).reindex(P.index); L=pd.DataFrame({a:d.low.astype(float) for a,d in D.items()}).reindex(P.index)
# Reversal of close location in the recent 20-day trading range, lagged one session.
hi=H.rolling(20,min_periods=15).max(); lo=L.rolling(20,min_periods=15).min()
F=(0.5-(P-lo)/(hi-lo).replace(0,np.nan)).shift(1)
F=F.sub(F.mean(axis=1),axis=0).div(F.std(axis=1).replace(0,np.nan),axis=0)
rows=[]
for dt in F.index:
 q=pd.concat([F.loc[dt],P.shift(-10).loc[dt]/P.loc[dt]-1],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min().date(),r.date.max().date(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(A))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True); r.to_csv('scripts/artifacts/miner_2_20351108_range_reversal_20d_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_2_20351108_range_reversal_20d_signal.csv')
