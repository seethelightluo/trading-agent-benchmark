import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try:x=get_stock_daily_data(s,days=3000)
 except Exception:x=None
 if x is not None and len(x): D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff(); ret20=r.rolling(20).sum(); v=r.rolling(20).std(); disp=v.median(1); cut=disp.rolling(252,min_periods=126).quantile(.8)
# Defensive flight-to-quality: rank assets by 20d residual strength, only during severe broad stress
stress=(ret20.median(1)<0)&(disp>cut); f=(ret20.sub(ret20.median(1),axis=0)/v).where(stress,0).shift(1)
for h in [10]:
 fw=np.log(p).shift(-h)-np.log(p); vals=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
 a=pd.Series(vals); print('dates',len(a),'avg_n',15,'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean(),'coverage',f.notna().mean().mean(),'stress_frac',stress.mean())
