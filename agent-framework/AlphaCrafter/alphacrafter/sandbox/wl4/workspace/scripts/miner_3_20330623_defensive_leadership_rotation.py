import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; prices={}
for a in assets:
 f=f'{base}/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
  prices[a]=d.close.astype(float)
P=pd.DataFrame(prices).sort_index(); R=P.pct_change()
# Orthogonal defensive-leadership rotation: short horizon relative strength,
# signed by lagged breadth of defensive assets versus cyclicals. All inputs lagged.
short=P.pct_change(20); long=P.pct_change(60)
defs=['XAU','US10Y','CN10Y']; cyc=['WTI','COPPER','BTC','ETH','SOX','NDX']
lead=(short[defs].mean(axis=1)-short[cyc].mean(axis=1)).rolling(10,min_periods=5).mean()
# asset-specific signal: 20d-vs-60d acceleration, with a common regime multiplier
acc=(short-long).shift(1)
gate=(lead.rolling(40,min_periods=20).rank(pct=True)-0.5).shift(1)
F=acc.mul(1+0.8*gate,axis=0)
rows=[]
for dt in F.index:
 fut=P.shift(-10).loc[dt]/P.loc[dt]-1
 z=pd.concat([F.loc[dt],fut],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('dates',len(r),'avgN',r.n.mean(),'assets',len(P.columns),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean())
for n in [260,520,780]:
 q=s.tail(n); print('recent',n,'IC',q.mean(),'ICIR',q.mean()/q.std())
print('coverage',F.notna().sum(axis=1).mean()/len(assets),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
os.makedirs('scripts/artifacts',exist_ok=True)
r.to_csv('scripts/artifacts/miner_3_20330623_defensive_leadership_rotation_ic.csv',index=False)
F.to_csv('scripts/artifacts/miner_3_20330623_defensive_leadership_rotation_signal.csv')
