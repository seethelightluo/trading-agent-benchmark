import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x): D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Cross-sectional shock reversal: fade a 10-day volatility-normalized move only on high dispersion days.
rv=r.rolling(20).std(); shock=r.rolling(10).sum().div(rv.rolling(10).mean()+1e-8)
disp=r.rolling(5).std().mean(axis=1)
gate=(disp>disp.rolling(120,min_periods=60).quantile(.60)).astype(float)
f=(-shock).mul(gate,axis=0).shift(1)
rows=[]
for h in [5,10,20]:
 y=p.pct_change(h).shift(-h); rows=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(a)>=8: rows.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); mu=z.ic.mean(); sd=z.ic.std()
 print('H',h,'assets',len(D),'dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15,'IC',mu,'ICIR',mu/sd,'hit',(z.ic>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
 if h==10:
  for label,lo,hi in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030-32','2030-01-01','2032-12-31'),('2033-34','2033-01-01','2034-01-06')]: print(label,len(z.loc[lo:hi]),z.loc[lo:hi].ic.mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20340106_dispersion_shock10_signal.csv',index=False)
