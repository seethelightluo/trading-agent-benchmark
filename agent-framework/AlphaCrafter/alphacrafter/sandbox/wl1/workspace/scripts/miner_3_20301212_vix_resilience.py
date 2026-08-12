import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs=[]
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 q=d.drop_duplicates('date'); xs.append(q.set_index(pd.to_datetime(q.date)).close.rename(s))
p=pd.concat(xs,axis=1).sort_index().ffill(); r=p.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv');v['date']=pd.to_datetime(v.date);v=v.set_index('date').close.reindex(p.index).ffill(); vr=v.pct_change()
# VIX-shock resilience: momentum plus inverse rolling sensitivity to daily VIX shocks.
# Covariance/beta is lagged and cross-sectionally ranked, so lower beta means resilience.
cov=r.rolling(60,min_periods=40).cov(vr); vv=vr.rolling(60,min_periods=40).var(); beta=cov.div(vv,axis=0)
mom=(p.pct_change(20)/(r.rolling(40,min_periods=25).std()*np.sqrt(40)+1e-9)).rank(axis=1,pct=True)
res=(-beta).rank(axis=1,pct=True)
shock=(vr>vr.rolling(252,min_periods=100).quantile(.7)).astype(float)
signal=(mom*(1-.35*shock.values[:,None]) + res*(.35*shock.values[:,None])).shift(1)
rows=[]
for i,dt in enumerate(p.index[:-21]):
 for h in [1,5,10,20]:
  z=pd.concat([signal.iloc[i].rename('x'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:rows.append((dt,h,z.x.corr(z.y,method='spearman')))
o=pd.DataFrame(rows,columns=['date','h','ic']);print('dates',o.date.nunique(),'assets',p.shape[1],'avgN',p.shape[1])
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first();print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
print('coverage %.4f turnover %.6f'%(signal.notna().mean().mean(),signal.rank(axis=1,pct=True).diff().abs().mean().mean()));signal.to_csv('scripts/miner_3_20301212_vix_resilience_signal.csv')
