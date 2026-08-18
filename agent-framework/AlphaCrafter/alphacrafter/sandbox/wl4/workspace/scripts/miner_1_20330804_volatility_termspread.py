import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for a in assets:
 f=f'../persistent/stock_data/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); P[a]=d.close.astype(float)
P=pd.DataFrame(P).sort_index().loc[:'2033-08-03']; R=P.pct_change()
# Volatility term-structure compression: assets whose short risk is low versus long risk.
v10=R.rolling(10,min_periods=8).std(); v60=R.rolling(60,min_periods=40).std()
F=(-(v10/v60-1)).shift(1)
rows=[]
for dt in F.index:
 fut=P.shift(-10).loc[dt]/P.loc[dt]-1
 z=pd.concat([F.loc[dt],fut],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min(),r.date.max(),'dates',len(r),'avgN',r.n.mean(),'assets',len(P.columns))
print('IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean())
for k in [260,520,780]:
 q=s.tail(k); print('recent',k,'IC',q.mean(),'ICIR',q.mean()/q.std(),'dates',len(q))
print('coverage',F.notna().sum(axis=1).mean()/len(P.columns),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
os.makedirs('scripts/artifacts',exist_ok=True)
r.to_csv('scripts/artifacts/miner_1_20330804_volatility_termspread_ic.csv',index=False)
F.to_csv('scripts/artifacts/miner_1_20330804_volatility_termspread_signal.csv')
