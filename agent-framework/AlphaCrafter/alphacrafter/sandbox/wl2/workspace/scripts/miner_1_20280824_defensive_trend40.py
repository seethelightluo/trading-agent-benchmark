import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<120: d=get_index_daily_data(s,days=3000)
 if d is not None: D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); lr=np.log(p).diff();
# Smooth 40-day residual trend, conditioned only when defensive basket leads; lagged one bar.
defs=[x for x in ['XAU','US10Y','CN10Y'] if x in p]
asset40=np.log(p).diff(40); cross=asset40.median(axis=1); defensive=asset40[defs].mean(axis=1)
lead=(defensive-cross).rolling(10,min_periods=5).mean().clip(lower=0)
vol=lr.rolling(60,min_periods=40).std()*np.sqrt(40)
f=asset40.sub(cross,axis=0).div(vol).mul(1+lead.clip(upper=0.25),axis=0).shift(1)
print('rows',len(f),'instruments',len(U),'coverage',round(f.notna().sum(axis=1).mean()/len(U),4))
for h in [1,5,10,20]:
 y=np.log(p).shift(-h)-np.log(p); a=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): a.append(q);ns.append(len(z));dates.append(dt)
 a=np.asarray(a); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2027-12-31'),('2028-01-01','2028-08-23')]:
  x=a[np.array([(d>=pd.Timestamp(lo))&(d<=pd.Timestamp(hi)) for d in dates])]
  if len(x)>10: print(' ',lo[:4],len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
print('turnover10',round((f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1)>0.05).mean(),4))
f.to_csv('scripts/miner_1_20280824_defensive_trend40_signal.csv')
