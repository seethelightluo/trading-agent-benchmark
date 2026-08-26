import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  z=x[['date','close']].copy();z.date=pd.to_datetime(z.date);D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.drop_duplicates('date').set_index('date').close.reindex(p.index).ffill()
rows=[]
# 30d trend/20d volatility, only when VIX is below its trailing 60d median.
for i,t in enumerate(p.index):
 if i<70 or i+10>=len(p) or not np.isfinite(v.iloc[i]) or v.iloc[i]>=v.iloc[i-60:i].median():continue
 sig=(p.iloc[i]/p.iloc[i-30]-1)/(r.iloc[i-19:i+1].std()*np.sqrt(20))
 f=p.iloc[i+10]/p.iloc[i]-1;q=pd.concat([sig,f],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1:rows.append((t,len(q),q.iloc[:,0].rank().corr(q.iloc[:,1].rank())))
A=pd.DataFrame(rows,columns=['date','n','ic']);x=A.ic.to_numpy();print('range',p.index.min().date(),p.index.max().date(),'assets',len(D),'gated_dates',len(A),'mean_n',A.n.mean(),'coverage',A.n.mean()/15)
print('IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1)*np.sqrt(252),'hit',(x>0).mean())
for label,lo,hi in [('2020-23','2020','2024'),('2024-26','2024','2027'),('2027-29','2027','2030'),('2028-29','2028','2030')]:
 y=A[(A.date>=lo)&(A.date<hi)].ic.to_numpy();print(label,len(y),np.mean(y) if len(y) else np.nan,np.mean(y)/np.std(y,ddof=1)*np.sqrt(252) if len(y)>1 else np.nan)
