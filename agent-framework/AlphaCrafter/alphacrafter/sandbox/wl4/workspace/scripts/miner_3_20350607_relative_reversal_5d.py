import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
P=pd.DataFrame(P).sort_index().loc[:'2035-06-06']; R=P.pct_change()
# Relative short-term reversal: subtract the cross-asset median 5-day return,
# then invert; lagged one completed session to avoid look-ahead.
r5=R.rolling(5,min_periods=5).sum(); F=(-(r5.sub(r5.median(axis=1),axis=0))).shift(1)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('data_last',P.index.max().date(),'period',r.date.min().date(),r.date.max().date(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(A))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
import os; os.makedirs('scripts/artifacts',exist_ok=True)
r.to_csv('scripts/artifacts/miner_3_20350607_relative_reversal_5d_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_3_20350607_relative_reversal_5d_signal.csv')
