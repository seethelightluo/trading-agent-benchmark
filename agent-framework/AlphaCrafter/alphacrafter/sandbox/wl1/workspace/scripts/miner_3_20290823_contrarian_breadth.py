import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2029-08-22')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}; idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
fast=.6*(r.rolling(5,min_periods=3).mean()>0).astype(float)+.4*(r.rolling(10,min_periods=5).mean()>0).astype(float); slow=.6*(r.rolling(20,min_periods=10).mean()>0).astype(float)+.4*(r.rolling(40,min_periods=20).mean()>0).astype(float)
# Contrarian breadth deceleration: favor assets whose recent directional breadth has weakened versus its slower baseline
f=(-(fast-slow)).sub((-(fast-slow)).mean(axis=1),axis=0).shift(1); print('factor contrarian_breadth_deceleration universe',15,'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20]:
 I=[];N=[];D=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1: I.append(spearmanr(q.f,q.y).statistic);N.append(len(q));D.append(px.index[i])
 a=np.array(I); d=pd.DatetimeIndex(D); print(h,'dates',len(a),'avgN',round(np.mean(N),2),'IC',round(a.mean(),6),'dailyICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'annICIR',round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(252),6),'hit',round((a>0).mean(),4))
 for label,mask in [('2020-25',d<pd.Timestamp('2026-01-01')),('2026+',d>=pd.Timestamp('2026-01-01')),('2028+',d>=pd.Timestamp('2028-01-01')),('2029YTD',d>=pd.Timestamp('2029-01-01'))]:
  z=a[mask]
  if len(z): print(' ',label,len(z),round(z.mean(),6),round(z.mean()/(z.std(ddof=1)+1e-12),6))
print('coverage',round(f.notna().sum().sum()/f.size,6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6)); f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20290823_contrarian_breadth_signal.csv',index=False)
