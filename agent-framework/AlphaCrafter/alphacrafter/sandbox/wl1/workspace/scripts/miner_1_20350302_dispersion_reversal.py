import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];xs={}
for s in U:
 try:d=get_index_daily_data(s,days=6000)
 except:
  try:d=get_stock_daily_data(s,days=6000)
  except:d=None
 if d is not None and len(d):xs[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float).sort_index()
p=pd.DataFrame(xs).sort_index().ffill();r=p.pct_change();cs=r.rolling(5).std().mean(axis=1);act=(cs>cs.rolling(60,min_periods=40).median()).astype(float);base=-(r.rolling(5).sum().shift(1))/r.rolling(20).std().shift(1);f=base.mul(act.shift(1),axis=0)
print('instruments',len(xs),'rows',len(p),'validf',f.notna().sum(axis=1).mean())
for h in [5,10,20,40]:
 fr=p.pct_change(h).shift(-h);vals=[];dates=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),fr.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1:vals.append(z.f.corr(z.r,method='spearman'));dates.append(dt);ns.append(len(z))
 q=pd.Series(vals,index=dates).dropna();print('H',h,'dates',len(q),'avgN',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
 if h==10:
  for a,b in [('2020','2024'),('2025','2029'),('2030','2035')]:
   z=q.loc[a:b];print('regime',a,len(z),z.mean(),z.mean()/z.std(ddof=1))
print('turnover',f.rank(axis=1,pct=True).diff().abs().stack().mean())
