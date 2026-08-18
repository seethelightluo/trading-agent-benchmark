import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for a in A:
 f=f'../persistent/stock_data/{a}.csv'
 if os.path.exists(f):
  x=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); D[a]=x.close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:'2033-07-07']; R=P.pct_change(); defs=['XAU','US10Y','CN10Y']; cyc=['WTI','COPPER','BTC','ETH','SOX','NDX']
# Candidate: 10d residual reversal, amplified when lagged defensive breadth dominates cyclical breadth.
ret=P.pct_change(10); cs=ret.sub(ret.mean(axis=1),axis=0); vol=R.rolling(30,min_periods=15).std()*np.sqrt(10)
breadth=(R[defs].rolling(20,min_periods=10).sum().gt(0).mean(axis=1)-R[cyc].rolling(20,min_periods=10).sum().gt(0).mean(axis=1)).rolling(10,min_periods=5).mean()
# positive bounded regime multiplier; reversal remains interpretable and all inputs lagged
F=(-cs/vol).shift(1).mul((1+0.7*breadth).clip(.45,1.55).shift(1),axis=0)
rows=[]
for dt in F.index:
 fut=P.shift(-10).loc[dt]/P.loc[dt]-1; z=pd.concat([F.loc[dt],fut],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min(),r.date.max(),'dates',len(r),'avgN',r.n.mean(),'assets',len(P.columns))
print('IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean())
for k in [260,520,780]:
 q=s.tail(k); print('recent',k,'IC',q.mean(),'ICIR',q.mean()/q.std())
print('coverage',F.notna().sum(axis=1).mean()/len(P.columns),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
os.makedirs('scripts/artifacts',exist_ok=True); r.to_csv('scripts/artifacts/miner_3_20330707_defensive_breadth_reversal_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_3_20330707_defensive_breadth_reversal_signal.csv')
