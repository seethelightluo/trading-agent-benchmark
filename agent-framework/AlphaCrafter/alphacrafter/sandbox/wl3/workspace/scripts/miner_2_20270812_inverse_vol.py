import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=3000)
   if d is not None and len(d)>=100:
    d=d.copy(); d['date']=pd.to_datetime(d['date']); return d.set_index('date')['close'].astype(float)
  except Exception: pass
p={s:g(s) for s in U}; p={s:x for s,x in p.items() if x is not None}
P=pd.concat(p,axis=1).sort_index().ffill(); R=P.pct_change()
# Standalone inverse realized-volatility preference, cross-sectional median demeaned and clipped.
vol=R.rolling(20,min_periods=15).std()*np.sqrt(252)
f=(1/(vol+1e-8)).sub((1/(vol+1e-8)).median(axis=1),axis=0).clip(-6,6)
ics={}
for h in [1,3,5,10]:
 fr=P.shift(-h)/P-1;v=[];ds=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1]));ds.append(dt);ns.append(len(z))
 ic=pd.Series(v,index=ds).dropna();ics[h]=ic
 print('H',h,'obs',len(ic),'avgN',round(float(np.mean(ns)),3),'IC',round(float(ic.mean()),6),'ICIR',round(float(ic.mean()/ic.std(ddof=1)*np.sqrt(len(ic))),6),'hit',round(float((ic>0).mean()),4))
 if h==1: print('coverage',round(float(f.notna().mean().mean()),6),'turnover',round(float(f.rank(pct=True).diff().abs().mean(axis=1).mean()),6))
if 1 in ics:
 for lab,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-27','2025-01-01','2027-08-11')]:
  x=ics[1][(ics[1].index>=a)&(ics[1].index<=b)]
  print('REG',lab,len(x),round(float(x.mean()),6),round(float(x.mean()/x.std(ddof=1)*np.sqrt(len(x))),4))
out=f.iloc[-1].rename('signal').to_frame();out.index.name='symbol';out.to_csv('scripts/miner_2_20270812_inverse_vol_signal.csv')
print('cutoff',f.dropna(how='all').index.max(),'dates',len(P),'instruments',len(P.columns))
