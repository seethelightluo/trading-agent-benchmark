import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for a in assets:
 f=f'../persistent/stock_data/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); P[a]=pd.to_numeric(d.close,errors='coerce')
P=pd.DataFrame(P).sort_index().loc[:'2033-08-17']; R=P.pct_change(); vol=R.rolling(20,min_periods=15).std(); F=(P.pct_change(60)/vol).shift(1)
rows=[]
for dt in F.index:
 fut=P.shift(-10).loc[dt]/P.loc[dt]-1; z=pd.concat([F.loc[dt],fut],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min().date(),r.date.max().date(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(P.columns)); print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4))
for k in [260,520,780]:
 q=s.tail(k); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
print('coverage',round(F.notna().sum(axis=1).mean()/len(P.columns),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6)); os.makedirs('scripts/artifacts',exist_ok=True); r.to_csv('scripts/artifacts/miner_1_20330818_voladjusted_momentum_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_1_20330818_voladjusted_momentum_signal.csv')
