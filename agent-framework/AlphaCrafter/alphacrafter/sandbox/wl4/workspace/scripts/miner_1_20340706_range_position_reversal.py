import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}; P=pd.DataFrame(P).sort_index()
R=P.pct_change(); lo=P.rolling(60,min_periods=40).min(); hi=P.rolling(60,min_periods=40).max(); pos=(P-lo)/(hi-lo)
# Oversold assets favored, but penalize extreme volatility; lag avoids lookahead
F=((0.5-pos)/(R.rolling(40,min_periods=25).std()*np.sqrt(40))).shift(1)
cut=pd.Timestamp('2034-07-05'); rows=[]
for dt in F.loc[:cut].index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('last',P.index.max().date(),'period',r.date.min().date(),r.date.max().date(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(A))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.loc[:cut].notna().sum(axis=1).mean()/len(A),4),'turnover',round(F.loc[:cut].rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True); r.to_csv('scripts/artifacts/miner_1_20340706_range_position_reversal_ic.csv',index=False); F.loc[:cut].to_csv('scripts/artifacts/miner_1_20340706_range_position_reversal_signal.csv')
