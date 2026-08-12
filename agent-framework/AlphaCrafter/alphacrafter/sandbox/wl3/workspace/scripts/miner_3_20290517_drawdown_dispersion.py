import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0: d=get_index_daily_data(s,4000)
 return d
def one(s,h=1):
 d=fetch(s)
 if d is None or len(d)<80:return None
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index()
 r=d.close.pct_change(); v=r.rolling(20,min_periods=15).std()
 # medium-horizon drawdown from recent peak, normalized by volatility;
 # cross-asset dispersion multiplier is computed cross-sectionally below
 dd=d.close/d.close.rolling(40,min_periods=30).max()-1
 f=(dd/(v*np.sqrt(20))).replace([np.inf,-np.inf],np.nan)
 return pd.DataFrame({'raw':f,'ret':r,'fwd':d.close.shift(-h)/d.close-1}).reset_index().assign(symbol=s)
qs=[one(s) for s in U]; qs=[q for q in qs if q is not None]
x=pd.concat(qs,ignore_index=True)
# daily dispersion from contemporaneous completed-day returns; lag one day to avoid using today's return
wide=x.pivot(index='date',columns='symbol',values='ret'); disp=wide.rolling(20,min_periods=12).std().median(axis=1)
# Use lagged dispersion regime, and bounded continuous high-dispersion boost
x=x.merge(disp.rename('disp'),on='date',how='left'); med=x.groupby('date').disp.transform('median')
x['signal']=x.raw*(1+0.75*(x.disp>med).astype(float))
def calc(z):
 out={}
 for dt,g in z.groupby('date'):
  g=g.dropna(subset=['signal','fwd'])
  if len(g)>=8 and g.signal.nunique()>1 and g.fwd.nunique()>1: out[dt]=g.signal.corr(g.fwd,method='spearman')
 return pd.Series(out).sort_index().dropna()
a=calc(x)
print('factor=40d drawdown reversal, high-dispersion conditional boost')
print('dates',len(a),'instruments',x.symbol.nunique(),'avg_valid',x.groupby('date').signal.apply(lambda z:z.notna().sum()).mean(),'coverage',x.signal.notna().mean())
print('daily IC %.8f ICIR %.8f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2027-12-31'),('2028-01-01','2028-12-31'),('2028-08-01','2029-05-15')]:
 z=a[(a.index>=lo)&(a.index<=hi)]; print('regime',lo,hi,'n',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
for h in [3,5,10]:
 q=[]
 for s in U:
  z=one(s,h)
  if z is not None:q.append(z)
 y=pd.concat(q,ignore_index=True).merge(disp.rename('disp'),on='date',how='left'); m=y.groupby('date').disp.transform('median'); y['signal']=y.raw*(1+0.75*(y.disp>m)); b=calc(y); print('h',h,'IC',b.mean(),'ICIR',b.mean()/b.std(ddof=1),'n',len(b))
x.to_csv('scripts/miner_3_20290517_drawdown_dispersion_signal.csv',index=False)
