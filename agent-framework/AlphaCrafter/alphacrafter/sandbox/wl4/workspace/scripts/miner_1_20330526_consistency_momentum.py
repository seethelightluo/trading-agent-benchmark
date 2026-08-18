import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; closes={}
for a in assets:
 f=f'{base}/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); closes[a]=d.close.astype(float)
P=pd.DataFrame(closes).reindex(columns=assets); R=P.pct_change()
# Consistency-weighted momentum: lagged 20-session return, weighted by fraction of positive days.
ret20=P/P.shift(20)-1
cons=(R.gt(0).rolling(20,min_periods=15).mean()-0.5)*2
vol=R.rolling(40,min_periods=25).std()*np.sqrt(20)
F=(ret20*cons/vol).shift(1)
rows=[]
for h in [5,10,20,30]:
 for dt in F.index:
  z=pd.concat([F.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in [5,10,20,30]:
 q=r[r.h==h]; s=q.ic
 print('horizon',h,'dates',len(q),'avgN',round(q.n.mean(),3),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),6))
 for n in [260,520,780]:
  t=s.tail(n); print(' recent',n,round(t.mean(),6),round(t.mean()/t.std(),6))
print('coverage',round(F.notna().sum(axis=1).mean()/len(assets),6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
os.makedirs('scripts/artifacts',exist_ok=True); r.to_csv('scripts/artifacts/miner_1_20330526_consistency_momentum_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_1_20330526_consistency_momentum_signal.csv')
