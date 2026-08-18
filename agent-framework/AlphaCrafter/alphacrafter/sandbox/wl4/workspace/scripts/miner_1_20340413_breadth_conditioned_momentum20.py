import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
P=pd.DataFrame(P).sort_index().loc[:'2034-04-13']
R=P.pct_change(); vol=R.rolling(40,min_periods=25).std()*np.sqrt(252)
r20=P.pct_change(20)
breadth=r20.median(axis=1)
# Trend-following in positive-breadth regimes, contrarian in negative-breadth regimes; lag all inputs.
raw=r20.div(vol).mul(np.where(breadth.values>=0,1.0,-1.0),axis=0)
F=(raw.sub(raw.median(axis=1),axis=0)).shift(1)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1
 z=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min(),r.date.max(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(P.columns))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(k); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(P.columns),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True)
r.to_csv('scripts/artifacts/miner_1_20340413_breadth_conditioned_momentum20_ic.csv',index=False)
F.to_csv('scripts/artifacts/miner_1_20340413_breadth_conditioned_momentum20_signal.csv')
