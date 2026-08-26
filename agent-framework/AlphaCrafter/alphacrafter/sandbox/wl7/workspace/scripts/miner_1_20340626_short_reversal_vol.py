import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index();r=p.pct_change();v=r.rolling(30).std(); f=(-(p.pct_change(5))/(v*np.sqrt(30))).shift(1); y=p.shift(-10)/p-1
q=[]; ns=[];ds=[]
for dt in p.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
q=pd.Series(q,index=ds).dropna();print('H10 IC %.8f ICIR %.8f hit %.4f dates %d avgN %.2f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q),np.mean(ns)))
for h in [1,5,20]:
 yy=p.shift(-h)/p-1;a=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(a);print('H%d IC %.8f ICIR %.8f'%(h,a.mean(),a.mean()/a.std(ddof=1)))
for n in [180,500,750]:
 z=q.iloc[-n:];print('RECENT%d %.8f %.8f'%(n,z.mean(),z.mean()/z.std(ddof=1)))
print('coverage %.6f turnover %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
