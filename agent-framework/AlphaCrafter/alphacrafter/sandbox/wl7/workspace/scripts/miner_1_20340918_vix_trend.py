import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in U:
 d=None
 try:d=get_index_daily_data(s,days=5200)
 except:pass
 if d is None:
  try:d=get_stock_daily_data(s,days=5200)
  except:pass
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);fs[s]=d.set_index('date').close.astype(float)
px=pd.DataFrame(fs).sort_index().ffill(); r=px.pct_change(); mom=px.pct_change(20); v=r.rolling(20).std()
try: vd=get_index_daily_data('VIX',days=5200)
except: vd=None
if vd is None: vd=get_stock_daily_data('VIX',days=5200)
vx=vd.assign(date=pd.to_datetime(vd.date)).set_index('date').close.astype(float).reindex(px.index).ffill()
# Trend signal only in subdued VIX, with neutral zero outside regime.
f=mom.where(vx < vx.rolling(60).median(),0)/v
out=[]
for h in [1,5,10,20]:
 y=px.shift(-h)/px-1; a=[];ds=[];ns=[]
 for dt in px.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:
   z=spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic;a.append(z);ds.append(dt);ns.append(ok.sum())
 a=np.array(a);print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d avgN %.2f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),len(a),np.mean(ns)))
 for aa,bb in [(2020,2022),(2023,2026),(2027,2030),(2031,2034)]:
  z=np.array([x for x,d in zip(a,ds) if aa<=d.year<=bb]);
  if len(z)>1:print(' ',aa,bb,len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340918_vix_trend_signal.csv',index=False)
