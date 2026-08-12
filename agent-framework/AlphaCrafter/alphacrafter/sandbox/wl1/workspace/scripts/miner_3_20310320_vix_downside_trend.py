import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
v=pd.read_csv('../persistent/index_data/VIX.csv');v['date']=pd.to_datetime(v['date']); v=v.set_index('date')['close'].astype(float)
P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):
  d=d.copy();d['date']=pd.to_datetime(d['date']);P[s]=d.drop_duplicates('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(P).sort_index().ffill(); r=np.log(p).diff();
tr=np.log(p).diff(20); dn=r.where(r<0,0).rolling(40,min_periods=20).std(); base=tr/(dn+1e-8)
vr=np.log(v).diff(); vz=(vr-vr.rolling(60,min_periods=40).mean())/(vr.rolling(60,min_periods=40).std()+1e-8)
state=1+0.6*vz.clip(lower=0).clip(upper=2); raw=base.div(state,axis=0)
sig=raw.shift(1).rank(axis=1,pct=True); rows=[]
for dt in sig.index:
 if dt not in p.index: continue
 for h in [1,5,10,20]:
  pos=p.index.get_loc(dt)
  if pos+h>=len(p): continue
  y=p.iloc[pos+h]/p.iloc[pos]-1; z=pd.concat([sig.loc[dt].rename('x'),y.rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((pd.Timestamp(dt),h,z.x.corr(z.y,method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['date','h','ic','n']);o['date']=pd.to_datetime(o.date)
print('dates',o.date.nunique(),'assets',p.shape[1],'avgN',o.groupby('date').n.first().mean(),'coverage',sig.notna().mean().mean(),'turnover',sig.diff().abs().mean().mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first();print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for yr,g in o[o.h==10].groupby(o[o.h==10].date.dt.year): print('year',yr,'IC %.6f ICIR %.6f n=%d'%(g.ic.mean(),g.ic.mean()/g.ic.std(ddof=1),len(g)))
sig.to_csv('scripts/miner_3_20310320_vix_downside_trend_signal.csv')
