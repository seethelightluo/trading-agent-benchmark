import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={s:get_stock_daily_data(s,4000) for s in U}; c=pd.DataFrame({s:x.set_index('date').close for s,x in p.items() if x is not None}).sort_index().ffill(); r=c.pct_change(); out=[]
for i in range(35,len(c)-10):
 f=-r.iloc[i-21:i].std(); y=c.iloc[i+10]/c.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).dropna()
 if len(z)>=8: out.append((c.index[i],len(z),z.f.corr(z.y)))
d=pd.DataFrame(out,columns=['date','n','ic']).dropna(); print('dates',len(d),'avg_n',d.n.mean(),'coverage',d.n.mean()/15,'IC',d.ic.mean(),'ICIR',d.ic.mean()/d.ic.std(),'hit',(d.ic>0).mean())
for a,b in [('2020','2023'),('2024','2025'),('2026','2027'),('2028','2028'),('2029','2029')]:
 x=d[(d.date.astype(str)>=a)&(d.date.astype(str)<=b)]; print(a,len(x),x.ic.mean(),x.ic.mean()/x.ic.std() if len(x)>1 else np.nan)
