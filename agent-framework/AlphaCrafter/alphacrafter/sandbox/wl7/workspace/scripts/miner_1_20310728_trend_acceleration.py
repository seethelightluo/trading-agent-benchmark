import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Volatility-adjusted trend acceleration: recent 10d return minus its prior 10d return,
# normalized by lagged 40d volatility. This captures improving/decelerating cross-asset trend.
r10=r.rolling(10,min_periods=10).sum(); prior10=r10.shift(10)
v40=r.rolling(40,min_periods=30).std()*np.sqrt(40)
sig=((r10-prior10)/(v40+1e-12)).shift(1)
# cross-sectional rank centers scales and avoids raw-unit domination
sig=sig.rank(axis=1,pct=True).sub(.5)
y={h:P.shift(-h)/P-1 for h in [1,5,10,20]}
def test(h):
 vals=[]; ns=[]; dates=[]
 for dt in sig.index:
  vv=sig.loc[dt].notna()&y[h].loc[dt].notna()
  if vv.sum()>=8:
   vals.append(sig.loc[dt,vv].corr(y[h].loc[dt,vv],method='spearman')); ns.append(int(vv.sum())); dates.append(dt)
 return pd.Series(vals,index=pd.to_datetime(dates)),ns
for h in [1,5,10,20]:
 a,ns=test(h); print('h',h,'dates',len(a),'avg_n %.2f'%np.mean(ns),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
a,ns=test(1)
print('rows',len(P),'assets',len(P.columns),'coverage %.5f turnover %.5f'%((sig.notna()).mean().mean(),sig.diff().abs().mean().mean()))
for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]: print('regime',i,j,'IC %.8f ICIR %.8f'%(a.iloc[i:j].mean(),a.iloc[i:j].mean()/a.iloc[i:j].std(ddof=1)))
pd.DataFrame({'date':a.index,'ic':a.values,'n':ns}).to_csv('scripts/miner_1_20310728_trend_acceleration_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20310728_trend_acceleration_signal.csv',index=False)
