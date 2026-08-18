import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for a in assets:
 f=f'../persistent/stock_data/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
  P[a]=d.close.astype(float)
P=pd.DataFrame(P).sort_index().loc[:'2033-09-14']
R=P.pct_change(); mkt=R.mean(axis=1)
# Trend strength is residualized from common market movement, then conditioned on broad cross-asset participation.
ret20=P.pct_change(20); vol30=R.rolling(30,min_periods=20).std()*np.sqrt(252)
breadth=ret20.gt(0).mean(axis=1)
# smooth breadth and use only lagged information; neutral around 50%, amplify agreement, suppress mixed regimes
bg=(breadth.rolling(10,min_periods=8).mean()-0.5)*2
F=(ret20/vol30).mul((1+0.7*bg),axis=0).shift(1)
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
r.to_csv('scripts/artifacts/miner_3_20330915_breadth_gated_trend_ic.csv',index=False)
F.to_csv('scripts/artifacts/miner_3_20330915_breadth_gated_trend_signal.csv')
