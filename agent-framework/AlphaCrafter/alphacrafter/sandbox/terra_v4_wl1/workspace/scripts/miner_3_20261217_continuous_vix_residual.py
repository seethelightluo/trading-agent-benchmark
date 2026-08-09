import pandas as pd, numpy as np
from scipy.stats import spearmanr
import os
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames=[]
for s in syms:
 p='../persistent/stock_data/'+s+'.csv'
 d=pd.read_csv(p,parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 d['r1']=d.close.pct_change()
 d['r7']=d.close.shift(1)/d.close.shift(8)-1
 d['vol20']=d.r1.shift(1).rolling(20,min_periods=15).std()
 frames.append(d[['date','close','r7','vol20']].assign(symbol=s))
x=pd.concat(frames,ignore_index=True)
# lagged cross-sectional residual reversal, with continuous VIX intensity
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date')
v=v[v.date<=END].copy(); vc='close' if 'close' in v else 'value'
v['vlag']=v[vc].shift(1); v['med60']=v['vlag'].rolling(60,min_periods=30).median(); v['intensity']=(v['vlag']/v['med60']-1).clip(-0.75,2.0)
x=x.merge(v[['date','intensity']],on='date',how='left')
x['resid']=x['r7']-x.groupby('date')['r7'].transform('median')
x['factor']=-x['resid']/x['vol20']*(1+x['intensity'].clip(lower=0).fillna(0))
for h in [1,5,10]: x['y'+str(h)]=x.groupby('symbol').close.shift(-h)/x.close-1
rows=[]
for (dt),g in x.groupby('date'):
 for h in [1,5,10]:
  z=g.dropna(subset=['factor','y'+str(h)])
  if len(z)>=8 and z.factor.nunique()>1 and z['y'+str(h)].nunique()>1: rows.append((dt,h,len(z),spearmanr(z.factor,z['y'+str(h)]).statistic))
r=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in [1,5,10]:
 q=r[r.h==h]; print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
 if h==1:
  for a,b in [(2020,2022),(2023,2024),(2025,2026)]:
   t=q[q.date.dt.year.between(a,b)]; print('REG',a,b,len(t),round(t.ic.mean(),6),round(t.ic.mean()/t.ic.std(ddof=1),6))
v=x.dropna(subset=['factor']); print('coverage',round(len(v)/len(x),4),'avg_valid',round(v.groupby('date').size().mean(),2))
p=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('turnover',round(p.diff().abs().mean(axis=1).mean(),6))
x.to_csv('scripts/miner_3_20261217_continuous_vix_residual_signal.csv',index=False)
