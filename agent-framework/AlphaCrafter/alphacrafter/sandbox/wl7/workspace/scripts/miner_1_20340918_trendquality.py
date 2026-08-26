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
px=pd.DataFrame(fs).sort_index().ffill(); ret=px.pct_change(); vol=ret.rolling(20).std()
# Smooth risk-adjusted medium trend, lag one completed session for forecast.
f=px.pct_change(30)/vol
vals={h:[] for h in [1,5,10,20]}; dates={h:[] for h in vals}; ns={h:[] for h in vals}
for h in vals:
 y=px.shift(-h)/px-1
 for dt in px.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:
   q=spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic
   vals[h].append(q);dates[h].append(dt);ns[h].append(ok.sum())
 print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d avgN %.2f'%(np.nanmean(vals[h]),np.nanmean(vals[h])/np.nanstd(vals[h],ddof=1),(np.array(vals[h])>0).mean(),len(vals[h]),np.mean(ns[h])))
 for a,b in [(2020,2022),(2023,2026),(2027,2030),(2031,2034)]:
  z=np.array([v for v,d in zip(vals[h],dates[h]) if a<=d.year<=b]);
  if len(z)>1: print(' ',a,b,len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
r=f.rank(axis=1,pct=True);print('coverage %.4f turnover %.4f'%(f.notna().sum(axis=1).mean()/15,r.diff().abs().mean(axis=1).mean()))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340918_trendquality_signal.csv',index=False)
