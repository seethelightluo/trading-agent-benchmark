import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-07-25')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
# Defensive volatility compression: recent realized vol relative to its slower baseline; lower is preferred.
v10=r.rolling(10,min_periods=8).std(); v40=r.rolling(40,min_periods=24).std(); f=-(v10/(v40+1e-8)).shift(1)
print('factor vol_compression_defensive universe',len(U),'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20]:
 I=[];N=[];D=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   I.append(spearmanr(q.f,q.y).statistic);N.append(len(q));D.append(px.index[i])
 a=np.array(I); print(h,'dates',len(a),'avgN',round(np.mean(N),2),'IC',round(a.mean(),6),'dailyICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'annICIR',round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(252),6),'hit',round((a>0).mean(),4))
 for label,mask in [('2020-25',np.array(D)<pd.Timestamp('2026-01-01')),('2026+',np.array(D)>=pd.Timestamp('2026-01-01')),('2028+',np.array(D)>=pd.Timestamp('2028-01-01')),('2029YTD',np.array(D)>=pd.Timestamp('2029-01-01'))]:
  z=a[mask]
  if len(z): print(' ',label,len(z),round(z.mean(),6),round(z.mean()/(z.std(ddof=1)+1e-12),6))
rank=f.rank(axis=1,pct=True); print('coverage',round(f.notna().sum().sum()/f.size,6),'turnover',round(rank.diff().abs().mean(axis=1).mean(),6))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20290726_vol_compression_signal.csv',index=False)
