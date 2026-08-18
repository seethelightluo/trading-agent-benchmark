import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
    try: d=get_stock_daily_data(s,days=4000)
    except Exception: d=None
    if d is None:
      try: d=get_index_daily_data(s,days=4000)
      except Exception: d=None
    if d is None or len(d)<150:return None
    d=d.copy();d['date']=pd.to_datetime(d['date']);d=d.sort_values('date').set_index('date')
    return pd.to_numeric(d['close'],errors='coerce').replace(0,np.nan)
P={s:load(s) for s in U};P={s:x for s,x in P.items() if x is not None};px=pd.DataFrame(P).sort_index();r=px.pct_change()
ret20=px/px.shift(20)-1; neg=r.where(r<0,0).rolling(20,min_periods=12).std();bread=(r>0).rolling(20,min_periods=12).mean();f=(ret20/(neg*np.sqrt(20)+1e-8)*bread).shift(1)
ics={h:[] for h in [1,5,10,20]};dates=[];ns=[];turn=[]
for i in range(len(px)-21):
 vals=f.iloc[i]
 for h in ics:
  z=pd.concat([vals,px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:ics[h].append(z.iloc[:,0].corr(z.iloc[:,1]))
 ns.append(vals.notna().sum());dates.append(px.index[i])
 if i: 
  z=pd.concat([f.iloc[i-1],vals],axis=1).dropna()
  if len(z)>=8:turn.append((z.iloc[:,0].rank()-z.iloc[:,1].rank()).abs().mean()/len(z))
print('dates',len(dates),'avg_n',np.mean(ns),'min_n',min(ns),'coverage',np.mean(ns)/15,'avg_turnover',np.mean(turn))
for h,x in ics.items():
 x=np.array(x);print(h,'n',len(x),'IC',np.nanmean(x),'ICIR',np.nanmean(x)/np.nanstd(x,ddof=1),'hit',np.mean(x>0),'recent250',np.nanmean(x[-250:]))
for lo,hi in [('2020','2023'),('2023','2025'),('2025','2029')]:
 x=[v for d,v in zip(dates,ics[10]) if lo<=str(d)[:4]<hi];print(lo+'-'+hi,'n',len(x),'ic10',np.mean(x))
