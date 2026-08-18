import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f): P[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').close.astype(float)
P=pd.DataFrame(P).sort_index().loc[:'2033-11-23']; R=P.pct_change()
# Family: relative intermediate momentum whose sign is conditioned on lagged cross-asset breadth.
base=P.pct_change(60).sub(P.pct_change(60).median(axis=1),axis=0)
breadth=(P.pct_change(20)>0).mean(axis=1)
# In stressed/low breadth regimes, invert momentum (reversal); otherwise retain trend.
reg=(breadth>=0.5).astype(float).replace(0,-1)
F=base.mul(reg,axis=0).shift(1)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; z=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min(),r.date.max(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(P.columns))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(k); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'dates',len(q))
print('coverage',round(F.notna().sum(axis=1).mean()/len(P.columns),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True); r.to_csv('scripts/artifacts/miner_2_20331124_regime_conditioned_momentum_10d_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_2_20331124_regime_conditioned_momentum_10d_signal.csv')
