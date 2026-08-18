import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}; V={}
for a in assets:
 f=f'../persistent/stock_data/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
  P[a]=d.close.astype(float)
  if 'volume' in d: V[a]=d.volume.astype(float)
P=pd.DataFrame(P).sort_index().loc[:'2033-09-28']
V=pd.DataFrame(V).reindex(P.index).reindex(columns=P.columns)
R=P.pct_change()
# Contrarian 60-day move excluding the latest 5 sessions, volatility scaled.
raw=(P.pct_change(60)-P.pct_change(5))
vol=R.rolling(40,min_periods=25).std()*np.sqrt(252)
base=-raw/vol
# Unusual relative volume confirms that the intermediate move was sentiment-driven.
rv=V/V.rolling(40,min_periods=25).mean()
F=(base*(1+0.35*np.log(rv.clip(lower=0.25,upper=4)))).shift(1)
rows=[]
for dt in F.index:
 fut=P.shift(-10).loc[dt]/P.loc[dt]-1
 z=pd.concat([F.loc[dt],fut],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min(),r.date.max(),'dates',len(r),'avgN',r.n.mean(),'assets',len(P.columns))
print('IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean())
for k in [120,260,520,780]:
 q=s.tail(k); print('recent',k,'IC',q.mean(),'ICIR',q.mean()/q.std())
print('coverage',F.notna().sum(axis=1).mean()/len(P.columns),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
os.makedirs('scripts/artifacts',exist_ok=True)
r.to_csv('scripts/artifacts/miner_3_20330929_volume_confirmed_intermediate_reversal_ic.csv',index=False)
F.to_csv('scripts/artifacts/miner_3_20330929_volume_confirmed_intermediate_reversal_signal.csv')
