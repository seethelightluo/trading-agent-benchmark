import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);raw[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(raw).sort_index().ffill(limit=3); r=np.log(p).diff()
shock=r.rolling(3,min_periods=3).sum(); vol=r.rolling(60,min_periods=30).std(); base=-shock/(vol*np.sqrt(3))
# Condition reversal on cross-asset dispersion: high dispersion receives stronger reversal weight.
disp=r.rolling(5,min_periods=3).std().mean(axis=1)
dz=(disp-disp.rolling(120,min_periods=60).mean())/disp.rolling(120,min_periods=60).std()
weight=(1+0.35*dz.clip(-2,2)).clip(0.25,2.0)
f=base.mul(weight,axis=0).shift(1); f=f.sub(f.mean(axis=1),axis=0)
fr=np.log(p.shift(-1)/p); vals=[];dates=[];ns=[]
for dt in f.index:
 ok=f.loc[dt].notna()&fr.loc[dt].notna()
 if ok.sum()>=8: vals.append(f.loc[dt,ok].corr(fr.loc[dt,ok],method='spearman'));dates.append(dt);ns.append(ok.sum())
z=pd.Series(vals,index=dates).dropna();print('dates',len(z),'avg_n',round(np.mean(ns),2),'IC',round(z.mean(),7),'ICIR',round(z.mean()/z.std(ddof=1),7),'hit',round((z>0).mean(),4))
for h in [3,5,10]:
 y=np.log(p.shift(-h)/p); vv=[];dd=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8: vv.append(f.loc[dt,ok].corr(y.loc[dt,ok],method='spearman'));dd.append(dt)
 q=pd.Series(vv,index=dd).dropna();print('H',h,len(q),round(q.mean(),7),round(q.mean()/q.std(ddof=1),7))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
f.to_csv('scripts/miner_1_20290809_dispersion_conditioned_shock_signal.csv')
