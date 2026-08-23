import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in A}
P=pd.DataFrame({a:D[a].close.astype(float) for a in A}).sort_index().loc[:'2035-12-19']
V=pd.DataFrame({a:D[a].volume.astype(float) for a in A}).reindex(P.index)
R=P.pct_change(); mom=P.pct_change(20); vol=R.rolling(20,min_periods=15).std()
vr=V/(V.rolling(60,min_periods=30).median())
# directional momentum whose strength is confirmed by abnormal but winsorized volume
confirm=np.sign(mom)*np.minimum(vr,3.0).clip(lower=0.25)
F=(mom/vol.replace(0,np.nan)*confirm).shift(1)
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
os.makedirs('scripts/artifacts',exist_ok=True)
r.to_csv('scripts/artifacts/miner_1_20351220_volume_confirmed_momentum_ic.csv',index=False)
F.to_csv('scripts/artifacts/miner_1_20351220_volume_confirmed_momentum_signal.csv')
