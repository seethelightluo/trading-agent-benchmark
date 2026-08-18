import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,4000); d['date']=pd.to_datetime(d.date); px[s]=d.set_index('date')['close'].astype(float)
dates=sorted(set.intersection(*[set(x.index) for x in px.values()])); rows=[]
for d in dates:
 vals={}
 for s in U:
  r=px[s].loc[:d].pct_change().dropna()
  if len(r)<121: continue
  # skip most recent 5 days to avoid short-term reversal contamination
  x=r.iloc[-65:-5]; vals[s]=x.sum()/(x.std()*np.sqrt(len(x))+1e-12)
 sig=pd.Series(vals); sig=sig-sig.median()
 for s in U:
  if pd.isna(sig.get(s,np.nan)): continue
  f=px[s].loc[px[s].index>d]
  if len(f)>=10:
   for H in [1,3,5,10]: rows.append((d,s,float(sig[s]),H,float(f.iloc[H-1]/px[s].loc[d]-1)))
df=pd.DataFrame(rows,columns=['date','s','sig','h','fut']); out='scripts/miner_2_20330708_skip5_momentum_signal.csv'; df.to_csv(out,index=False)
print('dates',len(dates),'used',df.date.nunique(),'avg_n',round(df.groupby('date').size().mean()/4,3),'coverage',round(df.s.count()/(len(dates)*15),4),'artifact',out)
for H,g in df.groupby('h'):
 ic=g.groupby('date').apply(lambda x:x.sig.corr(x.fut),include_groups=False).dropna(); print('H',H,'IC',round(g.sig.corr(g.fut),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'dates',len(ic))
 for a,b in [('2020','2025'),('2026','2029'),('2030','2033')]:
  z=ic[(ic.index>=a)&(ic.index<=b+'-12-31')]; print('regime',a+'-'+b,'n',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
