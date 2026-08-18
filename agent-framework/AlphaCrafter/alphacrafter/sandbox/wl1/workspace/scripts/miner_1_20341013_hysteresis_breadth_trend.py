import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None:d=get_index_daily_data(s,4000)
 if d is not None and len(d):D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();breadth=(r.rolling(60).mean()>0).mean(axis=1)
z=1; ss=[]
for b in breadth:
 if np.isfinite(b):
  if b>.55:z=1
  elif b<.45:z=-1
 ss.append(z)
state=pd.Series(ss,index=p.index);factor=r.rolling(20).sum().shift(1).mul(state.shift(1),axis=0);fwd=p.shift(-10).div(p).sub(1);ics=[];ds=[];ns=[]
for dt in p.index:
 x=factor.loc[dt];y=fwd.loc[dt];ok=x.notna()&y.notna()
 if ok.sum()>=8:ics.append(x[ok].corr(y[ok],method='spearman'));ds.append(dt);ns.append(ok.sum())
a=pd.Series(ics,index=pd.to_datetime(ds)).dropna();print('dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4));print('IC10',round(a.mean(),6),'dailyICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4),'turnover',round((factor.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.2).mean(),4))
for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 q=a.loc[lo:hi];print(lo,round(q.mean(),6),round(q.mean()/q.std(),6),len(q))
for h in [5,10,20,40]:
 yy=p.shift(-h).div(p).sub(1);vv=[]
 for dt in p.index:
  x=factor.loc[dt];y=yy.loc[dt];ok=x.notna()&y.notna()
  if ok.sum()>=8:vv.append(x[ok].corr(y[ok],method='spearman'))
 print('decay',h,round(pd.Series(vv).dropna().mean(),6))
out=factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal');out.to_csv('scripts/miner_1_20341013_hysteresis_breadth_trend_signal.csv',index=False)
