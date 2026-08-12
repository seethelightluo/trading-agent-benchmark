import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0: d=get_index_daily_data(s,4000)
 return d

def make(s,h):
 d=fetch(s)
 if d is None or len(d)<40:return None
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index()
 ret=d.close.pct_change(); vol=ret.rolling(20,min_periods=15).std()
 # Short-horizon overshoot reversal, scaled by ex-ante realized volatility
 f=-(d.close.pct_change(5)/(vol*np.sqrt(5))).replace([np.inf,-np.inf],np.nan)
 r=d.close.shift(-h)/d.close-1
 return pd.DataFrame({'f':f,'r':r}).dropna().reset_index()
def calc(q):
 out={}
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:out[dt]=g.f.corr(g.r,method='spearman')
 return pd.Series(out).sort_index().dropna()
allq=[]
for s in U:
 q=make(s,1)
 if q is not None:q['symbol']=s;allq.append(q)
x=pd.concat(allq,ignore_index=True);a=calc(x)
print('factor=5d volatility-scaled reversal')
print('dates',len(a),'instruments',x.symbol.nunique(),'avg_valid',x.groupby('date').size().mean(),'coverage',len(x)/(x.date.nunique()*15))
print('h1 IC %.8f ICIR %.5f hit %.4f'%(a.mean(),a.mean()/a.std(),(a>0).mean()))
for lo,hi in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2027-12-31'),('2028','2028-12-31'),('2028-08-01','2028-11-29')]:
 z=a[(a.index>=pd.Timestamp(lo))&(a.index<=pd.Timestamp(hi))];print('regime',lo,hi,'n',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std() if len(z)>1 else np.nan)
for h in [3,5,10]:
 qs=[]
 for s in U:
  q=make(s,h)
  if q is not None:qs.append(q)
 z=calc(pd.concat(qs,ignore_index=True));print('h',h,'IC',z.mean(),'ICIR',z.mean()/z.std(),'n',len(z))
print('signal rows',len(x))
x.to_csv('scripts/miner_3_20281130_volscaled_reversal5_signal.csv',index=False)
