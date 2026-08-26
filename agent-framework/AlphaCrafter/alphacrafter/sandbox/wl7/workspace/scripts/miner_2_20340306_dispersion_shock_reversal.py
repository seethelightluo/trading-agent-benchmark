import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];xs={}
for s in U:
 d=None
 try:d=get_index_daily_data(s,4000)
 except:pass
 if d is None or len(d)<250:
  try:d=get_stock_daily_data(s,4000)
  except:d=None
 if d is not None and len(d)>250:xs[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
p=pd.DataFrame(xs).sort_index().ffill();r=p.pct_change();v10=r.rolling(10).std();v60=r.rolling(60).std();disp=r.rolling(5).std().mean(axis=1)
base=(v10/v60).clip(.5,2);f=-p.pct_change(3).div(v10).mul(base,axis=0).mul((disp/disp.rolling(60).median()).clip(.7,1.8),axis=0)
rows=[]
for i in range(len(p)-1):
 q=pd.concat([f.iloc[i],(p.iloc[i+1]/p.iloc[i]-1)],axis=1).dropna()
 if len(q)>=8:rows.append((p.index[i],q.iloc[:,0].corr(q.iloc[:,1],method='spearman'),len(q)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');a=x.ic.dropna();print('assets',len(xs),'dates',len(x),'avg_n',x.n.mean(),'coverage',len(x)/(len(p)-1));print('IC',a.mean(),'std',a.std(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean(),'recent180',a.tail(180).mean(),'recent500',a.tail(500).mean());print('turnover',f.rank(pct=True).diff().abs().mean().mean())
for lo,hi in [('2020','2022'),('2023','2026'),('2027','2030'),('2031','2034')]:
 z=a.loc[lo:hi];print(lo,hi,len(z),z.mean(),z.mean()/z.std())
for h in [5,10]:
 rr=p.shift(-h)/p-1;z=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i],rr.iloc[i]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 z=pd.Series(z).dropna();print('H',h,len(z),z.mean(),z.mean()/z.std())
f.rank(pct=True).to_csv('scripts/miner_2_20340306_dispersion_shock_reversal_signal.csv')
