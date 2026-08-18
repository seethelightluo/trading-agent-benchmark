import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
P=pd.DataFrame(P).sort_index().loc[:'2034-12-20']
# Cross-asset trend: 60d return scaled by trailing 20d risk, lagged to prevent lookahead.
r60=P.pct_change(60); vol20=P.pct_change().rolling(20,min_periods=15).std()*np.sqrt(252)
base=r60/vol20
# VIX stress gate: suppress risk-seeking trend in high-volatility regimes, without dropping defensive assets.
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(P.index).ffill()
vr=(v-v.rolling(252,min_periods=126).mean())/v.rolling(252,min_periods=126).std()
gate=(1-0.35*np.clip(vr,-2,2)).clip(0.3,1.7)
F=base.mul(gate,axis=0).shift(1)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('last',P.index.max().date(),'period',r.date.min().date(),r.date.max().date(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(A))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True);r.to_csv('scripts/artifacts/miner_1_20341221_macro_conditioned_momentum_ic.csv',index=False);F.to_csv('scripts/artifacts/miner_1_20341221_macro_conditioned_momentum_signal.csv')
