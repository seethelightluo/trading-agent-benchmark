import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for a in assets:
 f=f'{base}/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
  px[a]=d.close.astype(float)
P=pd.DataFrame(px).sort_index(); r10=P.pct_change(10)
disp=r10.std(axis=1).rolling(20,min_periods=10).mean()
gate=disp.rolling(120,min_periods=60).rank(pct=True).shift(1)
base_signal=-r10.sub(r10.mean(axis=1),axis=0)
F=base_signal.mul(0.5+gate,axis=0).shift(1)
rows=[]
for dt in F.index:
 fut=P.shift(-10).loc[dt]/P.loc[dt]-1
 z=pd.concat([F.loc[dt],fut],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('dates',len(r),'avgN',r.n.mean(),'assets',len(P.columns),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean())
for n in [260,520,780]:
 q=s.tail(n); print('recent',n,'IC',q.mean(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean())
print('coverage',F.notna().sum(axis=1).mean()/len(assets),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
os.makedirs('scripts/artifacts',exist_ok=True)
r.to_csv('scripts/artifacts/miner_2_20330804_dispersion_gated_reversal_ic.csv',index=False)
F.to_csv('scripts/artifacts/miner_2_20330804_dispersion_gated_reversal_signal.csv')
