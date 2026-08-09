import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for s in U}
# Acceleration: recent 5d return relative to its prior 15d average daily return.
raw={s:(px[s].pct_change(5)-px[s].pct_change(20).shift(5)*.25) for s in U}
rows=[]
for d in sorted(set().union(*[set(x.index) for x in px.values()])):
 for h in [1,5,10]:
  f=[];y=[]
  for s in U:
   if d not in px[s].index: continue
   i=px[s].index.get_loc(d); z=raw[s].get(d,np.nan)
   if i+h<len(px[s]) and np.isfinite(z): f.append(z);y.append(px[s].iloc[i+h]/px[s].iloc[i]-1)
  if len(f)>=8 and np.ptp(f)>0: rows.append((d,h,spearmanr(f,y).statistic,len(f)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 x=df[df.h==h]; print('H',h,'dates',len(x),'avgN',round(x.n.mean(),2),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(ddof=1),6),'hit',round((x.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
  z=x.set_index('date').loc[lo:hi].ic; print(lo,len(z),round(z.mean(),6) if len(z) else None)
# artifact for admission horizon
out=[]
for d in sorted(set().union(*[set(x.index) for x in px.values()])):
 for s in U:
  out.append((d,s,raw[s].get(d,np.nan)))
pd.DataFrame(out,columns=['date','asset','signal']).to_csv('../persistent/factor_signals_miner_2_20270225_acceleration.csv',index=False)
