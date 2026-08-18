import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,4000)
 if x is None:x=get_index_daily_data(s,4000)
 if x is not None:D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change(); disp=r.rolling(5).std().mean(axis=1)
for qtile in [.6,.7,.8]:
 for look in [3,5]:
  gate=disp>disp.rolling(120).quantile(qtile)
  mom=p.pct_change(look); sig=-mom.sub(mom.median(axis=1),axis=0).where(gate,0).shift(1); rows=[]
  for d in sig.index:
   z=pd.concat([sig.loc[d],p.shift(-10).loc[d]/p.loc[d]-1],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].std()>0:rows.append(z.iloc[:,0].corr(z.iloc[:,1]))
  a=np.array(rows);print('q',qtile,'look',look,'n',len(a),'ic',a.mean(),'icir',a.mean()/a.std(),'hit',(a>0).mean())
