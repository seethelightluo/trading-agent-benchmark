import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for a in assets:
 f=f'../persistent/stock_data/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); P[a]=d.close.astype(float)
P=pd.DataFrame(P).sort_index().loc[:'2033-08-03']
R=P.pct_change()
# Low-volatility leadership: inverse realized risk, with a modest reversal overlay.
vol=R.rolling(30,min_periods=20).std()
rev=-P.pct_change(10)
F=(0.7*(-vol).rank(axis=1,pct=True)+0.3*rev.rank(axis=1,pct=True)).shift(1)
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
r.to_csv('scripts/artifacts/miner_1_20330804_lowvol_dispersion_reversal_ic.csv',index=False)
F.to_csv('scripts/artifacts/miner_1_20330804_lowvol_dispersion_reversal_signal.csv')
