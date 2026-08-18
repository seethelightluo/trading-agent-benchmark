import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}; V={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 P[a]=d.close.astype(float); V[a]=d.volume.astype(float).replace(0,np.nan)
P=pd.DataFrame(P); V=pd.DataFrame(V).reindex(P.index)
R=P.pct_change(); volsur=(V.rolling(5,min_periods=3).mean()/V.rolling(40,min_periods=20).mean()-1)
# directional 20d return confirmed by abnormal volume, normalized by range volatility
rv=R.rolling(20,min_periods=15).std(); rng=((P.diff()/P.shift(1)).abs()).rolling(20,min_periods=15).mean()
F=(R.rolling(20,min_periods=15).sum()/(rv*np.sqrt(20)))*np.sign(R.rolling(20,min_periods=15).sum())*0.5 + (R.rolling(20,min_periods=15).sum()*volsur/(rng*np.sqrt(20)))
F=F.shift(1)
cut=P.index[-11]; rows=[]
for dt in F.loc[:cut].index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('last',P.index.max().date(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(A),'period',r.date.min().date(),r.date.max().date())
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.loc[:cut].notna().sum(axis=1).mean()/len(A),4),'turnover',round(F.loc[:cut].rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True); r.to_csv('scripts/artifacts/miner_1_20340706_volume_range_confirmation_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_1_20340706_volume_range_confirmation_signal.csv')
