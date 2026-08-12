import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is None or len(d)<120: d=get_index_daily_data(s,days=3200)
 if d is not None and len(d)>0: D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); lr=np.log(p).diff()
# Volatility-normalized trend with agreement between medium and long horizons.
# The signal is lagged one completed bar; positive return is rewarded only when 20d and 60d trends agree.
m20=np.log(p).diff(20); m60=np.log(p).diff(60); vol=lr.rolling(20,min_periods=15).std()*np.sqrt(20)
ag=np.sign(m20*m60)
f=(m20/vol)*ag
f=f.sub(f.median(axis=1),axis=0).shift(1)
print('rows',len(p),'assets',len(D),'coverage',round(f.notna().mean().mean(),4))
for h in [1,3,5,10,20]:
 y=np.log(p).shift(-h)-np.log(p); vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): vals.append(q); ns.append(len(z))
 a=np.asarray(vals)
 print('h',h,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
q=f.rank(axis=1,pct=True); print('turnover',round((q.diff().abs().mean(axis=1)/2).mean(),6))
f.to_csv('scripts/miner_3_20280921_agreement_voltrend_signal.csv')
