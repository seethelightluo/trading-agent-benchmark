import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x): D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Downside asymmetry quality: favor assets whose medium-term return is achieved with relatively mild downside volatility.
down=r.clip(upper=0).rolling(40).std(); up=r.clip(lower=0).rolling(40).std(); ret=r.rolling(40).sum()
f=ret.div(down+1e-8).sub(0.5*ret.div(up+1e-8)).shift(1)
for h in [5,10,20]:
 y=p.pct_change(h).shift(-h); rows=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   c=a.iloc[:,0].corr(a.iloc[:,1])
   if np.isfinite(c): rows.append((dt,c,len(a)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');mu=z.ic.mean();sd=z.ic.std()
 print('H',h,'assets',len(D),'dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15,'IC',mu,'ICIR',mu/sd,'hit',(z.ic>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
 if h==10:
  for label,lo,hi in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030-32','2030-01-01','2032-12-31'),('2033-34','2033-01-01','2034-01-06')]: print(label,len(z.loc[lo:hi]),z.loc[lo:hi].ic.mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340106_downside_quality_signal.csv',index=False)
