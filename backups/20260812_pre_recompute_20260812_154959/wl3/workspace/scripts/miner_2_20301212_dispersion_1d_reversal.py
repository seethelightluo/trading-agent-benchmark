import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try:d=get_stock_daily_data(s,5000)
 except:d=None
 if d is None or len(d)<300:
  try:d=get_index_daily_data(s,5000)
  except:d=None
 if d is not None and len(d):
  d=d.copy();d['date']=pd.to_datetime(d['date']);D[s]=d.set_index('date')['close'].astype(float).rename(s)
px=pd.concat(D,axis=1).sort_index();r=px.pct_change(); vol=r.rolling(20).std()*np.sqrt(252)
# One-day relative reversal, only in elevated cross-sectional dispersion; inverse-volatility risk scaling.
r1=r; disp=r1.std(axis=1); gate=disp>disp.rolling(120,min_periods=60).median()
sig=-(r1.sub(r1.median(axis=1),axis=0))/vol.replace(0,np.nan);sig=sig.where(gate,0.)
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');q=a.ic
print('daily',q.mean(),q.mean()/q.std(),'hit',(q>0).mean(),'dates',len(q),'median_n',a.n.median(),'coverage',a.n.sum()/(len(a)*15),'gate_frac',gate.mean())
for name,sl in [('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028-30',slice('2028','2030'))]:
 x=a.loc[sl,'ic'];print(name,len(x),x.mean(),x.mean()/x.std() if len(x)>1 else np.nan)
for h in [3,5,10]:
 vals=[];y=px.pct_change(h).shift(-h)
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 x=pd.Series(vals);print('horizon',h,x.mean(),x.mean()/x.std(),len(x))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20301212_dispersion_1d_reversal_signal.csv',index=False)
