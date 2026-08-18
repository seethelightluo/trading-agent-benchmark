import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,4000); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
dates=sorted(set.intersection(*[set(x.index) for x in px.values()])); rows=[]
for d in dates:
 vals={}
 for s in U:
  r=px[s].loc[:d].pct_change().dropna()
  if len(r)>=121:
   x=r.iloc[-65:-5]; vals[s]=x.sum()/(x.std()*np.sqrt(len(x))+1e-12)
 sig=pd.Series(vals); sig=sig-sig.median()
 for s in U:
  if s in sig:
   f=px[s].loc[px[s].index>d]
   if len(f)>=10: rows.append((d,s,float(sig[s]),float(f.iloc[9]/px[s].loc[d]-1)))
df=pd.DataFrame(rows,columns=['date','s','sig','fut']); out='scripts/miner_2_20330722_skip5_momentum_signal.csv'; df.to_csv(out,index=False)
print('dates',len(dates),'used',df.date.nunique(),'avg_n',round(df.groupby('date').size().mean(),3),'coverage',round(len(df)/(len(dates)*15),4),'artifact',out)
ic=df.groupby('date').apply(lambda x:x.sig.corr(x.fut),include_groups=False).dropna(); print('H 10 IC',round(df.sig.corr(df.fut),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'dates',len(ic))
for a,b in [('2020','2025'),('2026','2029'),('2030','2033')]:
 z=ic[(ic.index>=a)&(ic.index<=b+'-12-31')]; print('regime',a+'-'+b,'n',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
