import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):
  d.date=pd.to_datetime(d.date);P[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill();r=np.log(p).diff(); v=r.rolling(20,min_periods=10).std()
vix=pd.read_csv('../persistent/index_data/VIX.csv');vix.date=pd.to_datetime(vix.date); vv=vix.set_index('date').close.reindex(p.index).ffill(); z=(vv/vv.rolling(120,min_periods=40).median()).clip(.7,2)
# one-day shock reversal gated by elevated VIX and cross-sectional dispersion
D=r.rolling(20,min_periods=10).std().mean(axis=1); ds=D/D.rolling(120,min_periods=40).median()
g=((z>1.05)&(ds>1.0)).astype(float)
sig=(-r/(v+1e-8)).mul(g,axis=0).shift(1).rank(axis=1,pct=True)
rows=[]
for dt in sig.index:
 i=p.index.get_loc(dt)
 for h in [1,5,10,20]:
  if i+h>=len(p):continue
  q=pd.concat([sig.loc[dt].rename('x'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8:rows.append((dt,h,q.x.corr(q.y,method='spearman'),len(q)))
o=pd.DataFrame(rows,columns=['date','h','ic','n']);print('dates',o.date.nunique(),'assets',p.shape[1],'avgN',o.groupby('date').n.first().mean(),'coverage',sig.notna().mean().mean(),'turnover',sig.diff().abs().mean().mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first().dropna();print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
print('recent',[(a,b, (g2:=o[(o.h==1)&(o.date.dt.year.between(a,b))].ic.dropna()).mean(),g2.mean()/g2.std(ddof=1),len(g2)) for a,b in [(2020,2022),(2023,2025),(2026,2028),(2029,2030),(2031,2031)]])
sig.to_csv('scripts/miner_2_20310807_vix_dispersion_shock_signal.csv')
