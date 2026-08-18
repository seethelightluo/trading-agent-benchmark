import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for a in assets:
 f=f'../persistent/stock_data/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); P[a]=d.close.astype(float)
P=pd.DataFrame(P).sort_index().loc[:'2033-07-07']; R=P.pct_change(); nA=len(P.columns)
# Candidate: volatility-scaled medium-term trend, conditioned by lagged defensive breadth.
# Breadth is deliberately a regime state, not a future-dependent trade filter.
defs=[x for x in ['XAU','US10Y','CN10Y'] if x in P]; cyc=[x for x in ['WTI','COPPER','BTC','ETH','SOX','NDX'] if x in P]
trend=P.pct_change(20); vol=R.rolling(40,min_periods=20).std()*np.sqrt(20)
base=(trend/vol).replace([np.inf,-np.inf],np.nan)
db=(R[defs].rolling(20,min_periods=10).sum().gt(0).mean(axis=1)-R[cyc].rolling(20,min_periods=10).sum().gt(0).mean(axis=1))
# smooth and lag; multiplier remains positive and bounded
reg=(1+0.8*db.rolling(10,min_periods=5).mean()).clip(.35,1.65).shift(1)
F=base.shift(1).mul(reg,axis=0)
rows=[]
for dt in F.index:
 fut=P.shift(-10).loc[dt]/P.loc[dt]-1
 z=pd.concat([F.loc[dt],fut],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min(),r.date.max(),'dates',len(r),'avgN',r.n.mean(),'assets',nA)
print('IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean())
for k in [260,520,780]:
 q=s.tail(k); print('recent',k,'IC',q.mean(),'ICIR',q.mean()/q.std())
print('coverage',F.notna().sum(axis=1).mean()/nA,'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
os.makedirs('scripts/artifacts',exist_ok=True)
r.to_csv('scripts/artifacts/miner_3_20330707_defensive_breadth_voltrend_ic.csv',index=False)
F.to_csv('scripts/artifacts/miner_3_20330707_defensive_breadth_voltrend_signal.csv')
