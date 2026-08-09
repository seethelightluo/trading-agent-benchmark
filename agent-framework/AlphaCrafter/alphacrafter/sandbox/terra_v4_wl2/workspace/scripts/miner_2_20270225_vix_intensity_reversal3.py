import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None and len(d): return d
  except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index()
r=px.pct_change(); fwd=px.shift(-1)/px-1
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(px.index).ffill()
vr=v.pct_change(); med=vr.rolling(60,min_periods=30).median(); mad=(vr-med).abs().rolling(60,min_periods=30).median()
shock=((vr-med)/(1.4826*mad.replace(0,np.nan))).clip(lower=0,upper=4).fillna(0)
base=-r.rolling(3).sum(); f=base.mul(1+shock,axis=0)
f=f.sub(f.median(axis=1),axis=0)
rows=[]
for dt in f.index:
 z=pd.DataFrame({'a':f.loc[dt],'b':fwd.loc[dt]}).dropna()
 if len(z)>=8 and z.a.nunique()>1: rows.append((dt,z.a.corr(z.b,method='spearman'),len(z)))
d=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for label,x in [('all',d),('2020-22',d.loc['2020':'2022']),('2023-24',d.loc['2023':'2024']),('2025-26',d.loc['2025':'2026']),('2027',d.loc['2027':])]:
 ic=x.ic.mean(); print(label,'dates',len(x),'avgN',round(x.n.mean(),2),'IC',round(ic,6),'ICIR',round(ic/x.ic.std(ddof=1),6),'hit',round((x.ic>0).mean(),4))
print('coverage',f.notna().sum().sum()/(len(U)*len(f)))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_2_20270225_vix_intensity_reversal3.csv',index=False)
print('artifact rows',len(out))
