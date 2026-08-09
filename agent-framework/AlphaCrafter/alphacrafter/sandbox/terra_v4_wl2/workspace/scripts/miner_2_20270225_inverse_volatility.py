import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
def run():
 ds=[]
 for s in U:
  x=get_stock_daily_data(symbol=s,days=2100)
  if x is None or len(x)<80: continue
  x=x.sort_values('date'); c=x.close.astype(float).values; r=c[1:]/c[:-1]-1
  for i in range(20,len(r)-1): ds.append((x.date.iloc[i+1],s,1/max(np.std(r[i-19:i+1]),.001),c[i+1]/c[i]-1))
 a=pd.DataFrame(ds,columns=['date','s','sig','f']); ic=a.groupby('date').apply(lambda z:z.sig.corr(z.f) if len(z)>=8 else np.nan).dropna()
 print('dates',len(ic),'inst',a.s.nunique(),'avg',a.groupby('date').size().mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean())
for h in [3,5,10]: pass
run()
