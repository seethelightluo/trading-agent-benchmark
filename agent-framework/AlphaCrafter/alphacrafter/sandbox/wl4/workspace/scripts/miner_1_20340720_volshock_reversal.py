import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for a in A}; P=pd.concat(P,axis=1).sort_index().loc[:'2034-07-20']; R=P.pct_change(); v=R.rolling(5,min_periods=4).std()/R.rolling(60,min_periods=40).std(); F=(-v).shift(1)
rows=[]
for d in F.index:
 y=P.shift(-10).loc[d]/P.loc[d]-1;q=pd.concat([F.loc[d],y],axis=1).dropna()
 if len(q)>=8: rows.append((d,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']);s=r.ic
print('period',r.date.min().date(),r.date.max().date(),'dates',len(r),'avgN',round(r.n.mean(),2));print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s)));print('recent',k,'ICIR',round(q.mean()/q.std(ddof=1),6),'IC',round(q.mean(),6))
print('coverage',round(F.notna().sum(axis=1).mean()/15,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4));os.makedirs('scripts/artifacts',exist_ok=True);r.to_csv('scripts/artifacts/miner_1_20340720_volshock_reversal_ic.csv',index=False);F.to_csv('scripts/artifacts/miner_1_20340720_volshock_reversal_signal.csv')