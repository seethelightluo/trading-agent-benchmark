import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for a in assets:
 f=f'{base}/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
  px[a]=d.close.astype(float)
P=pd.DataFrame(px).sort_index(); ret=P.pct_change()
# reward-to-downside-risk: intermediate horizon trend divided by recent downside volatility
mom=P.pct_change(20)
down=ret.where(ret<0,0).pow(2).rolling(40,min_periods=20).mean().pow(.5)
F=(mom/(down+1e-8)).shift(1)
rows=[]
for dt in F.index:
 fut=P.shift(-10).loc[dt]/P.loc[dt]-1
 z=pd.concat([F.loc[dt],fut],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(P.columns),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),4),'hit',round((s>0).mean(),4))
for n in [260,520,780]:
 q=s.tail(n); print('recent',n,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),4),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(assets),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True)
r.to_csv('scripts/artifacts/miner_2_20330818_downside_risk_momentum_ic.csv',index=False)
F.to_csv('scripts/artifacts/miner_2_20330818_downside_risk_momentum_signal.csv')
