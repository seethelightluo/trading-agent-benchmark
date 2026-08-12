import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0: d=get_index_daily_data(s,4000)
 return d
rows=[]
for s in U:
 d=fetch(s)
 if d is None: continue
 d=d.copy();d['date']=pd.to_datetime(d['date']);d=d.set_index('date').sort_index();rng=(d.high-d.low).replace(0,np.nan);atr=(d.high-d.low).rolling(20,min_periods=10).mean();f=((d.close-d.open)/rng).clip(-3,3)*(rng/atr).clip(0,4);r=d.close.shift(-1)/d.close-1
 for z in pd.concat([f.rename('f'),r.rename('r')],axis=1).dropna().itertuples(): rows.append((z[0],s,z[1],z[2]))
x=pd.DataFrame(rows,columns=['date','symbol','f','r'])
def calc(q):
 out={}
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: out[dt]=g.f.corr(g.r,method='spearman')
 return pd.Series(out).sort_index().dropna()
a=calc(x);print('dates',len(a),'avg instruments',x.groupby('date').size().mean(),'coverage',len(x)/(x.date.nunique()*15),'IC %.8f ICIR %.5f hit %.4f'%(a.mean(),a.mean()/a.std(),(a>0).mean()))
for lo,hi in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2027-12-31'),('2028','2028-12-31')]:
 z=a[(a.index>=pd.Timestamp(lo))&(a.index<=pd.Timestamp(hi))];print(lo,hi,len(z),z.mean(),z.mean()/z.std() if len(z)>1 else np.nan)
for h in [1,3,5,10]:
 rr=[]
 for s in U:
  d=fetch(s)
  if d is None:continue
  d=d.copy();d['date']=pd.to_datetime(d.date);d=d.set_index('date').sort_index();rng=(d.high-d.low).replace(0,np.nan);atr=(d.high-d.low).rolling(20,min_periods=10).mean();f=((d.close-d.open)/rng).clip(-3,3)*(rng/atr).clip(0,4);r=d.close.shift(-h)/d.close-1;q=pd.DataFrame({'f':f,'r':r}).dropna();q=q.reset_index(names='date');rr.append(q)
 q=pd.concat(rr);ii=calc(q);print('h',h,'IC',ii.mean(),'ICIR',ii.mean()/ii.std(),'n',len(ii))
x.to_csv('scripts/miner_3_20281116_candle_pressure_signal.csv',index=False)
