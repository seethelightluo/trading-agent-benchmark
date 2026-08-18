import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for a in assets:
 f=f'../persistent/stock_data/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); P[a]=d.close.astype(float)
P=pd.DataFrame(P).sort_index().loc[:'2033-09-28']; R=P.pct_change()
r20=P.pct_change(20); v20=R.rolling(20,min_periods=15).std(); v60=R.rolling(60,min_periods=40).std()
# Trend strength amplified during compression, lagged one day.
F=(r20/v20*(v20/v60)).shift(1)
for h in [1,5,10,20]:
 rows=[]
 fut=P.shift(-h)/P-1
 for dt in F.index:
  z=pd.concat([F.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
 print('H',h,'period',r.date.min().date(),r.date.max().date(),'dates',len(r),'avgN',round(r.n.mean(),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4))
 for k in [120,260,520]:
  q=s.tail(k); print(' recent',k,round(q.mean(),6),round(q.mean()/q.std(),6))
print('coverage',round(F.notna().sum(axis=1).mean()/len(P.columns),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'assets',len(P.columns))
F.to_csv('scripts/artifacts/miner_2_20330929_volcompression_trend_signal.csv')
