import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is None or len(d)<120: d=get_index_daily_data(s,days=3200)
 if d is not None: D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Trend efficiency: signed 20d move relative to realized path length, cross-sectionally centered.
move=np.log(p).diff(20); path=r.abs().rolling(20,min_periods=15).sum(); eff=move/path
f=eff.sub(eff.median(axis=1),axis=0).shift(1)
print('rows',len(p),'assets',len(D),'factor coverage',round(f.notna().mean().mean(),4))
for h in [1,3,5,10,20]:
 y=np.log(p).shift(-h)-np.log(p); vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): vals.append(q);ns.append(len(z))
 a=np.asarray(vals); print('h',h,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
# 10d rank turnover
q=f.rank(axis=1,pct=True); turn=(q.diff().abs().mean(axis=1)/2).mean(); print('turnover',round(turn,6))
f.to_csv('scripts/miner_3_20280907_efficiency_trend_signal.csv')
