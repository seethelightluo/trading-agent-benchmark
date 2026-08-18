import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in A}
P=pd.DataFrame({a:D[a].close.astype(float) for a in A}).sort_index().loc[:'2034-06-07']
R=P.pct_change(); med=R.rolling(20,min_periods=15).sum().median(axis=1)
# Range-adjusted relative reversal: lagged negative relative 20d return divided by 20d average true-range proxy.
rel=R.rolling(20,min_periods=15).sum().sub(med,axis=0)
tr=pd.DataFrame(index=P.index,columns=A,dtype=float)
for a in A:
 d=D[a].reindex(P.index); prev=d.close.shift(1)
 tr[a]=pd.concat([d.high-prev,d.low-prev],axis=1).abs().max(axis=1)/prev
atr=tr.rolling(20,min_periods=15).mean()
F=(-rel/atr).shift(1)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min(),r.date.max(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(A))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True)
r.to_csv('scripts/artifacts/miner_2_20340608_range_adjusted_reversal_ic.csv',index=False)
F.to_csv('scripts/artifacts/miner_2_20340608_range_adjusted_reversal_signal.csv')
