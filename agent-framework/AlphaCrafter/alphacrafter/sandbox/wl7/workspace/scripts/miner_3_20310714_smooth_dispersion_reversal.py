import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Smooth volatility-scaled short reversal, with continuous dispersion emphasis.
r3=r.rolling(3,min_periods=3).sum(); v20=r.rolling(20,min_periods=15).std()*np.sqrt(20)
base=-(r3/(v20+1e-12))
disp=r.std(axis=1).rolling(20,min_periods=15).mean()
# Cross-sectional dispersion z-score uses only prior observations, bounded to avoid concentration.
dmu=disp.rolling(120,min_periods=60).mean(); dsd=disp.rolling(120,min_periods=60).std()
weight=(1+0.5*((disp-dmu)/(dsd+1e-12)).clip(-1,1)).shift(1)
sig=base.mul(weight,axis=0).shift(1)
sig=sig.rank(axis=1,pct=True).sub(.5)
def test(h):
 y=P.shift(-h)/P-1; vals=[]; ns=[]; dates=[]
 for dt in sig.index:
  vv=sig.loc[dt].notna()&y.loc[dt].notna()
  if vv.sum()>=8:
   vals.append(sig.loc[dt,vv].corr(y.loc[dt,vv],method='spearman')); ns.append(int(vv.sum())); dates.append(dt)
 a=pd.Series(vals,index=pd.to_datetime(dates)); return a,ns
for h in [1,5,10,20]:
 a,ns=test(h); print('h',h,'dates',len(a),'avg_n %.2f'%np.mean(ns),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
a,ns=test(1)
print('rows',len(P),'assets',len(P.columns),'coverage %.5f turnover %.5f'%((sig.notna()).mean().mean(),sig.diff().abs().mean().mean()))
for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]: print('regime',i,j,'IC %.8f'%a.iloc[i:j].mean())
pd.DataFrame({'date':a.index,'ic':a.values,'n':ns}).to_csv('scripts/miner_3_20310714_smooth_dispersion_reversal_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20310714_smooth_dispersion_reversal_signal.csv',index=False)
