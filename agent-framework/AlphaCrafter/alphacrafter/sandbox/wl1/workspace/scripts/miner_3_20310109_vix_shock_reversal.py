import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date); v=pd.to_numeric(v.set_index('date').close,errors='coerce')
P={}
for s in U:
 try:d=get_index_daily_data(s,4100)
 except FileNotFoundError:d=get_stock_daily_data(s,4100)
 d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index(); r=np.log(p).diff(); vv=np.log(v.reindex(p.index).ffill()).diff(); shock=(vv-vv.rolling(60).mean())/(vv.rolling(60).std()+1e-12)
# defensive conditional reversal: reverse recent 3d return more strongly during VIX shocks
base=-r.rolling(3).sum(); sig=base.mul((1+0.75*shock.clip(lower=0)),axis=0).shift(1).rank(axis=1,pct=True)
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; a=[];ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=pd.Series(a).dropna();print(h,len(a),np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
print('turnover',sig.diff().abs().mean(axis=1).mean())
