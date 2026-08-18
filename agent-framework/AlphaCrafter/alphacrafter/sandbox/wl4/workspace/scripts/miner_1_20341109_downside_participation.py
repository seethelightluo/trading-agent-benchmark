import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}).sort_index().loc[:'2034-11-08']
r=P.pct_change(); market=r.mean(axis=1)
# Defensive downside participation: inverse fraction of negative market days in trailing 40 sessions.
neg=(r.lt(0).astype(float)).rolling(40,min_periods=30).mean(); mneg=market.lt(0).astype(float).rolling(40,min_periods=30).mean()
F=(1-neg.div(mneg.replace(0,np.nan),axis=0)).shift(1)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
R=pd.DataFrame(rows,columns=['date','n','ic']); s=R.ic
print('period',R.date.min().date(),R.date.max().date(),'dates',len(R),'avgN',round(R.n.mean(),2),'assets',len(A))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True); R.to_csv('scripts/artifacts/miner_1_20341109_downside_participation_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_1_20341109_downside_participation_signal.csv')
